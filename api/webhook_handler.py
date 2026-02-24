import hmac
import hashlib
import json
import os
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from api.github_client import fetch_pr_diff

load_dotenv()

router = APIRouter()
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET","")

def verify_signature(payload: bytes, signature: str) -> bool:
    "Verify the weebhook coming from github not from someone else"
    if not signature or not signature.startswith("sha256="):
        return False
    excepted = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={excepted}",signature)
async def process_pr(payload: dict):
    """
    Background task — runs after we return 200 to GitHub.
    GitHub expects a fast response, so heavy work goes here.
    """
    repo_name = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]
    pr_title  = payload["pull_request"]["title"]
    pr_author = payload["pull_request"]["user"]["login"]

    print(f"\n{'='*60}")
    print(f"CodeGuard — New PR Detected")
    print(f"   Repo   : {repo_name}")
    print(f"   PR #   : {pr_number}")
    print(f"   Title  : {pr_title}")
    print(f"   Author : {pr_author}")
    print(f"{'='*60}")

    # Fetch the diff from GitHub API
    diff_chunks = await fetch_pr_diff(repo_name, pr_number)

    print(f"\nFiles changed: {len(diff_chunks)}")
    for chunk in diff_chunks:
        print(f"\n {chunk['filename']}")
        print(f"     Status    : {chunk['status']}")
        print(f"     Additions : +{chunk['additions']}")
        print(f"     Deletions : -{chunk['deletions']}")
        if chunk["patch"]:
            preview = chunk["patch"].split("\n")[:8]
            print(f"     Patch preview:")
            for line in preview:
                print(f"       {line}")

    print(f"\n Diff extraction complete — ready for agent layer\n")
    return diff_chunks


@router.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    payload_bytes = await request.body()
    signature     = request.headers.get("X-Hub-Signature-256", "")

    # Step 1 — verify it really came from GitHub
    if not verify_signature(payload_bytes, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event   = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(payload_bytes)

    # Step 2 — only handle PR events
    if event == "pull_request":
        action = payload.get("action", "")
        if action in ["opened", "synchronize", "reopened"]:
            print(f"⚡ PR event received: action={action}")
            # Step 3 — process in background, return fast to GitHub
            background_tasks.add_task(process_pr, payload)
            return {
                "status": "processing",
                "pr": payload["pull_request"]["number"]
            }

    return {"status": "ignored", "event": event}

