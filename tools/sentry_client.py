import os
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()

SENTRY_AUTH_TOKEN = os.getenv("SENTRY_AUTH_TOKEN", "")
SENTRY_ORG        = os.getenv("SENTRY_ORG", "")
SENTRY_PROJECT    = os.getenv("SENTRY_PROJECT", "")


class SentryClient:
    def __init__(self):
        self.enabled = all([SENTRY_AUTH_TOKEN, SENTRY_ORG, SENTRY_PROJECT])
        self.base_url = "https://sentry.io/api/0"
        if not self.enabled:
            print("Sentry not configured — will use simulated data")

    def get_recent_errors(self, filename: str = None, limit: int = 5) -> list:
        """
        Fetch recent production errors from Sentry.
        If filename given, filter errors related to that file.
        Used to give agents production context during review.
        """
        if not self.enabled:
            # Return realistic simulated data for development
            return self._simulated_errors(filename)

        try:
            url = (
                f"{self.base_url}/projects/{self.SENTRY_ORG}"
                f"/{self.SENTRY_PROJECT}/issues/"
                f"?limit={limit}&query=is:unresolved"
            )
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {SENTRY_AUTH_TOKEN}"}
            )
            with urllib.request.urlopen(req) as resp:
                issues = json.loads(resp.read())

            errors = []
            for issue in issues:
                if filename and filename not in str(issue.get("culprit", "")):
                    continue
                errors.append({
                    "title"    : issue.get("title", ""),
                    "culprit"  : issue.get("culprit", ""),
                    "count"    : issue.get("count", 0),
                    "last_seen": issue.get("lastSeen", ""),
                    "level"    : issue.get("level", "error")
                })
            return errors

        except Exception as e:
            print(f"   Sentry error: {e}")
            return self._simulated_errors(filename)

    def format_for_context(self, errors: list) -> str:
        """Format Sentry errors as readable context for LLM agents."""
        if not errors:
            return "No recent production errors found."

        lines = ["Recent production errors (from Sentry):"]
        for e in errors:
            lines.append(
                f"  - [{e['level'].upper()}] {e['title']} "
                f"| {e['count']} occurrences | last: {e['last_seen']}"
            )
        return "\n".join(lines)

    def _simulated_errors(self, filename: str = None) -> list:
        """Realistic simulated Sentry errors for development."""
        errors = [
            {
                "title"    : "OperationalError: no such table: users",
                "culprit"  : "app.py in get_users",
                "count"    : 47,
                "last_seen": "2 hours ago",
                "level"    : "error"
            },
            {
                "title"    : "TypeError: unsupported operand type(s)",
                "culprit"  : "app.py in calc",
                "count"    : 12,
                "last_seen": "1 day ago",
                "level"    : "error"
            },
            {
                "title"    : "sqlite3.ProgrammingError: Incorrect number of bindings",
                "culprit"  : "app.py in get_admin",
                "count"    : 8,
                "last_seen": "3 hours ago",
                "level"    : "warning"
            }
        ]
        if filename:
            return [e for e in errors if filename in e["culprit"]]
        return errors
