import os
import time
import jwt
import httpx
from dotenv import load_dotenv

load_dotenv()

APP_ID           = os.getenv("GITHUB_APP_ID")
PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH", "./github_app.pem")


def generate_jwt() -> str:
    """
    Generate a short-lived JWT using our private key.
    GitHub uses this to verify we are App ID 2900288.
    """
    with open(PRIVATE_KEY_PATH, "r") as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "iat": now - 60,        # issued 60s ago (handles clock drift)
        "exp": now + (10 * 60), # expires in 10 minutes
        "iss": APP_ID           # our App ID
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


async def get_installation_token(repo_full_name: str) -> str:
    """
    Exchange our JWT for a short-lived installation token.
    This token is what we use to actually call GitHub API.
    """
    app_jwt = generate_jwt()
    owner   = repo_full_name.split("/")[0]  # "Tejesh0209"

    async with httpx.AsyncClient() as client:

        # Find which installation ID belongs to our target owner
        resp = await client.get(
            "https://api.github.com/app/installations",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            }
        )
        resp.raise_for_status()
        installations = resp.json()

        installation_id = None
        for inst in installations:
            if inst["account"]["login"].lower() == owner.lower():
                installation_id = inst["id"]
                break

        if not installation_id:
            raise ValueError(f"No installation found for {owner}")

        # Exchange for an installation access token
        token_resp = await client.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            }
        )
        token_resp.raise_for_status()
        return token_resp.json()["token"]


async def fetch_pr_diff(repo_full_name: str, pr_number: int) -> list[dict]:
    """
    Fetch all changed files + their diffs for a given PR.
    Returns a list of chunks — one per file changed.
    """
    token = await get_installation_token(repo_full_name)

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            }
        )
        resp.raise_for_status()
        files = resp.json()

    chunks = []
    for f in files:
        chunks.append({
            "filename"  : f["filename"],
            "status"    : f["status"],        # added / modified / removed
            "additions" : f["additions"],
            "deletions" : f["deletions"],
            "patch"     : f.get("patch", ""), # the actual diff lines
        })

    return chunks