import os
import asyncio
from dotenv import load_dotenv
from agents.agent import Agent
from agents.run import Runner
from agents.tool import function_tool
from api.github_client import get_installation_token
import httpx

load_dotenv()

# ── Tools the agent can call ─────────────────────────────────────

@function_tool
async def get_file_contents(repo_name: str, file_path: str, branch: str = "main") -> str:
    """Read the current contents of a file from the GitHub repository."""
    token = await get_installation_token(repo_name)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.github.com/repos/{repo_name}/contents/{file_path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept"       : "application/vnd.github+json"
            },
            params={"ref": branch}
        )
        if resp.status_code == 200:
            import base64
            data    = resp.json()
            content = base64.b64decode(data["content"]).decode("utf-8")
            return f"FILE: {file_path}\nSHA: {data['sha']}\n\n{content}"
        return f"ERROR: Could not fetch {file_path} — status {resp.status_code}"


@function_tool
async def create_branch(repo_name: str, branch_name: str, from_branch: str = "main") -> str:
    """Create a new branch in the repository for the auto-fix."""
    token = await get_installation_token(repo_name)
    async with httpx.AsyncClient() as client:
        # Get SHA of base branch
        ref_resp = await client.get(
            f"https://api.github.com/repos/{repo_name}/git/ref/heads/{from_branch}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept"       : "application/vnd.github+json"
            }
        )
        if ref_resp.status_code != 200:
            return f"ERROR: Could not get base branch SHA — {ref_resp.status_code}"

        sha = ref_resp.json()["object"]["sha"]

        # Create new branch
        create_resp = await client.post(
            f"https://api.github.com/repos/{repo_name}/git/refs",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept"       : "application/vnd.github+json"
            },
            json={
                "ref": f"refs/heads/{branch_name}",
                "sha": sha
            }
        )
        if create_resp.status_code == 201:
            return f"SUCCESS: Branch '{branch_name}' created from '{from_branch}'"
        elif create_resp.status_code == 422:
            return f"SUCCESS: Branch '{branch_name}' already exists"
        return f"ERROR: Could not create branch — {create_resp.status_code} {create_resp.text}"


@function_tool
async def commit_fix(
    repo_name   : str,
    file_path   : str,
    new_content : str,
    commit_message: str,
    branch_name : str,
    file_sha    : str
) -> str:
    """Commit the fixed file content to the auto-fix branch."""
    import base64
    token = await get_installation_token(repo_name)

    encoded = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")

    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"https://api.github.com/repos/{repo_name}/contents/{file_path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept"       : "application/vnd.github+json"
            },
            json={
                "message" : commit_message,
                "content" : encoded,
                "sha"     : file_sha,
                "branch"  : branch_name
            }
        )
        if resp.status_code in [200, 201]:
            return f"SUCCESS: Committed fix to {file_path} on branch {branch_name}"
        return f"ERROR: Commit failed — {resp.status_code} {resp.text}"


@function_tool
async def open_draft_pr(
    repo_name   : str,
    branch_name : str,
    base_branch : str,
    pr_title    : str,
    pr_body     : str
) -> str:
    """Open a Draft Pull Request with the auto-fix changes."""
    token = await get_installation_token(repo_name)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.github.com/repos/{repo_name}/pulls",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept"       : "application/vnd.github+json"
            },
            json={
                "title" : pr_title,
                "body"  : pr_body,
                "head"  : branch_name,
                "base"  : base_branch,
                "draft" : True
            }
        )
        if resp.status_code == 201:
            pr_url = resp.json()["html_url"]
            pr_num = resp.json()["number"]
            return f"SUCCESS: Draft PR #{pr_num} created — {pr_url}"
        return f"ERROR: Could not create PR — {resp.status_code} {resp.text}"


@function_tool
def validate_python_syntax(code: str) -> str:
    """Check if Python code has valid syntax before committing."""
    import ast
    try:
        ast.parse(code)
        return "VALID: Python syntax is correct"
    except SyntaxError as e:
        return f"INVALID: SyntaxError at line {e.lineno}: {e.msg}"


# ── Agent Definition ─────────────────────────────────────────────

AUTOFIX_INSTRUCTIONS = """You are CodeGuard's Auto-Fix Agent.

You receive a list of CRITICAL and HIGH severity issues found in a PR.
Your job is to automatically fix these issues by:
1. Reading the affected file using get_file_contents
2. Generating the corrected version of the file
3. Validating the syntax with validate_python_syntax
4. Creating a fix branch with create_branch
5. Committing the fix with commit_fix
6. Opening a Draft PR with open_draft_pr

IMPORTANT RULES:
- Only fix CRITICAL and HIGH severity issues
- Never break existing functionality
- Keep fixes minimal — change only what's needed to fix the issue
- Always validate Python syntax before committing
- If syntax is invalid after fix, try again with corrected fix
- Use branch name format: codeguard/autofix-pr-{pr_number}
- PR title format: [CodeGuard Auto-Fix] Fix {n} issues in PR #{pr_number}

Common fixes:
  SQL Injection (CWE-89):
    BEFORE: cursor.execute(f"SELECT * WHERE id={user_id}")
    AFTER:  cursor.execute("SELECT * WHERE id=?", (user_id,))

  Hardcoded credentials (CWE-798):
    BEFORE: password = "admin123"
    AFTER:  password = os.getenv("ADMIN_PASSWORD")
            (also add: import os at top if not present)

  N+1 Query:
    BEFORE: for user in users: db.query(orders WHERE user_id=user.id)
    AFTER:  JOIN query fetching all users and orders at once

Once the Draft PR is created, report:
  - Which issues were fixed
  - The PR URL
  - Any issues that could not be safely auto-fixed
"""

autofix_agent = Agent(
    name         = "AutoFixAgent",
    instructions = AUTOFIX_INSTRUCTIONS,
    model        = "gpt-4o",
    tools        = [
        get_file_contents,
        create_branch,
        commit_fix,
        open_draft_pr,
        validate_python_syntax
    ]
)


# ── Runner Function ──────────────────────────────────────────────

async def run_autofix(
    repo_name   : str,
    pr_number   : int,
    final_report: dict
) -> dict:
    """
    Run the Auto-Fix Agent on CRITICAL and HIGH issues.
    Called from LangGraph when severity == CRITICAL.
    """
    security_issues = final_report.get("security_issues", [])
    perf_issues     = final_report.get("perf_issues", [])
    style_issues    = final_report.get("style_issues", [])
    arch_issues     = final_report.get("arch_issues", [])

    all_issues   = security_issues + perf_issues + style_issues + arch_issues
    fixable      = [i for i in all_issues if i.get("severity") in ["CRITICAL", "HIGH"]]

    if not fixable:
        return {"status": "skipped", "reason": "no CRITICAL or HIGH issues"}

    print(f"\n   Auto-Fix Agent starting — {len(fixable)} issues to fix")

    # Format issues for the agent
    issues_text = "\n".join([
        f"- [{i.get('severity')}] {i.get('file')}:{i.get('line')} — "
        f"{i.get('message')} | Fix: {i.get('suggestion')}"
        for i in fixable
    ])
    prompt = f"""
    Fix the following CRITICAL and HIGH issues found in PR #{pr_number} of {repo_name}:

    {issues_text}

    Repository: {repo_name}
    Branch to read files from: main
    Branch to create for fix: codeguard/autofix-pr-{pr_number}
    Base branch: main

    IMPORTANT: When calling get_file_contents, always pass:
    - repo_name = "{repo_name}"
    - file_path = the filename (e.g. "app.py")
    - branch = "main"

    Work through each issue:
    1. Call get_file_contents("{repo_name}", "app.py", "main")
    2. Apply the fix to the content
    3. Call validate_python_syntax with the fixed content
    4. Call create_branch("{repo_name}", "codeguard/autofix-pr-{pr_number}", "main")
    5. Call commit_fix with the fixed content
    6. Call open_draft_pr once all fixes are committed
    """

    result = await Runner.run(autofix_agent, prompt)

    print(f"\n   Auto-Fix complete:")
    print(f"   {result.final_output}")

    return {
        "status"  : "completed",
        "summary" : result.final_output,
        "fixed"   : len(fixable)
    }
