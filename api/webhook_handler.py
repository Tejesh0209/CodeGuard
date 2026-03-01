import hmac
import hashlib
import json
import os
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from api.github_client import fetch_pr_diff
from orchestration.graph import codeguard_graph

load_dotenv()

router = APIRouter()
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def verify_signature(payload: bytes, signature: str) -> bool:
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


async def process_pr(payload: dict):
    try:
        repo_name = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]
        pr_title  = payload["pull_request"]["title"]
        pr_author = payload["pull_request"]["user"]["login"]

        print(f"\n{'='*60}")
        print(f"🔍 CodeGuard — New PR Detected")
        print(f"   Repo   : {repo_name}")
        print(f"   PR #   : {pr_number}")
        print(f"   Title  : {pr_title}")
        print(f"   Author : {pr_author}")
        print(f"{'='*60}")

        # Step 1 — fetch diff
        diff_chunks = await fetch_pr_diff(repo_name, pr_number)
        print(f"\n📂 Files changed: {len(diff_chunks)}")

        # Step 2 — build initial state
        initial_state = {
            "repo_name"       : repo_name,
            "pr_number"       : pr_number,
            "pr_title"        : pr_title,
            "pr_author"       : pr_author,
            "diff_chunks"     : diff_chunks,
            "style_review"    : None,
            "security_review" : None,
            "perf_review"     : None,
            "severity_level"  : "LOW",
            "should_autofix"  : False,
            "final_report"    : None,
            "messages"        : []
        }

        # Step 3 — run LangGraph
        config = {"configurable": {"thread_id": f"pr-{repo_name}-{pr_number}"}}
        result = await codeguard_graph.ainvoke(initial_state, config=config)

        print(f"\nCodeGuard review complete for PR #{pr_number}")
        return result

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


@router.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    payload_bytes = await request.body()
    signature     = request.headers.get("X-Hub-Signature-256", "")

    if not verify_signature(payload_bytes, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event   = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(payload_bytes)

    if event == "pull_request":
        action = payload.get("action", "")
        if action in ["opened", "synchronize", "reopened"]:
            print(f"⚡ PR event received: action={action}")
            background_tasks.add_task(process_pr, payload)
            return {
                "status": "processing",
                "pr"    : payload["pull_request"]["number"]
            }

    return {"status": "ignored", "event": event}