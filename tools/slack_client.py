import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_CHANNEL     = os.getenv("SLACK_CHANNEL", "#code-review")


class SlackClient:
    def __init__(self):
        self.enabled = bool(SLACK_WEBHOOK_URL)
        if not self.enabled:
            print("Slack not configured — alerts will be simulated")

    def send_alert(self, message: str, severity: str = "LOW") -> dict:
        """Send a severity-based alert to Slack."""
        severity_emoji = {
            "CRITICAL": ":rotating_light:",
            "HIGH"    : ":red_circle:",
            "MEDIUM"  : ":large_yellow_circle:",
            "LOW"     : ":large_green_circle:"
        }.get(severity, ":white_circle:")

        payload = {
            "channel"  : SLACK_CHANNEL,
            "text"     : f"{severity_emoji} *[CodeGuard]* {message}",
            "username" : "CodeGuard",
            "icon_emoji": ":robot_face:"
        }

        if not self.enabled:
            print(f"   [SIMULATED] Slack alert: {message[:80]}")
            return {"status": "simulated"}

        try:
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                SLACK_WEBHOOK_URL,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                return {"status": "sent", "code": resp.getcode()}
        except Exception as e:
            print(f"   Slack error: {e}")
            return {"status": "error", "error": str(e)}

    def send_review_summary(
        self,
        repo_name   : str,
        pr_number   : int,
        pr_title    : str,
        final_report: dict
    ) -> dict:
        """Send a formatted review summary to Slack."""
        summary  = final_report.get("summary", {})
        severity = final_report.get("severity", "LOW")
        approved = final_report.get("approved", False)

        verdict = "APPROVED" if approved else "CHANGES REQUESTED"

        message = (
            f"{verdict} | PR #{pr_number} in {repo_name}\n"
            f"*{pr_title}*\n"
            f"Severity: {severity} | "
            f"Issues: {summary.get('total_issues', 0)} total "
            f"({summary.get('critical', 0)} critical, "
            f"{summary.get('high', 0)} high)\n"
            f"Scores: Style {summary.get('style_score', 0)}/10 | "
            f"Security {summary.get('security_score', 0)}/10 | "
            f"Perf {summary.get('perf_score', 0)}/10 | "
            f"Arch {summary.get('arch_score', 0)}/10"
        )

        return self.send_alert(message, severity)
