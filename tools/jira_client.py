import os
from dotenv import load_dotenv

load_dotenv()

JIRA_URL         = os.getenv("JIRA_URL", "")
JIRA_EMAIL       = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN   = os.getenv("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "CG")


class JiraClient:
    def __init__(self):
        self.enabled = all([JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN])
        if self.enabled:
            from tools.jira_client import JIRA
            self.client = JIRA(
                server=JIRA_URL,
                basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN)
            )
            print("JIRA client connected")
        else:
            print("JIRA not configured — tickets will be simulated")

    def create_ticket(
        self,
        summary     : str,
        description : str,
        issue_type  : str = "Bug",
        priority    : str = "High",
        labels      : list = None
    ) -> dict:
        """
        Create a JIRA ticket for a code review finding.
        Falls back to simulation if JIRA not configured.
        """
        if not self.enabled:
            # Simulate for development
            ticket_id = f"{JIRA_PROJECT_KEY}-{hash(summary) % 1000}"
            print(f"   🎫 [SIMULATED] JIRA ticket: {ticket_id}")
            print(f"      Summary: {summary[:60]}...")
            return {
                "key"    : ticket_id,
                "url"    : f"https://jira.example.com/browse/{ticket_id}",
                "status" : "simulated"
            }

        # Real JIRA ticket creation
        issue_dict = {
            "project"    : {"key": JIRA_PROJECT_KEY},
            "summary"    : summary[:255],    # JIRA has 255 char limit
            "description": description,
            "issuetype"  : {"name": issue_type},
            "priority"   : {"name": priority},
            "labels"     : labels or ["codeguard", "automated"]
        }

        issue = self.client.create_issue(fields=issue_dict)
        print(f"JIRA ticket created: {issue.key}")

        return {
            "key"    : issue.key,
            "url"    : f"{JIRA_URL}/browse/{issue.key}",
            "status" : "created"
        }

    def create_tickets_from_review(
        self,
        repo_name   : str,
        pr_number   : int,
        pr_title    : str,
        final_report: dict
    ) -> list[dict]:
        """
        Create JIRA tickets for all HIGH and CRITICAL issues.
        LOW and MEDIUM issues don't get tickets — too noisy.
        """
        tickets  = []
        pr_url   = f"https://github.com/{repo_name}/pull/{pr_number}"

        # Collect all HIGH+ issues from all agents
        all_issues = (
            final_report.get("security_issues", []) +
            final_report.get("perf_issues",     []) +
            final_report.get("style_issues",    []) +
            final_report.get("arch_issues",     [])
        )

        high_plus = [
            i for i in all_issues
            if i.get("severity") in ["HIGH", "CRITICAL"]
        ]

        print(f"\n🎫 Creating JIRA tickets for {len(high_plus)} HIGH+ issues...")

        for issue in high_plus:
            severity = issue.get("severity", "HIGH")
            priority = "Highest" if severity == "CRITICAL" else "High"

            summary = (
                f"[CodeGuard] [{severity}] {issue.get('message', '')[:100]}"
            )

            description = f"""
CodeGuard automated review found a {severity} issue in PR #{pr_number}.

*Pull Request:* [{pr_title}|{pr_url}]
*File:* {issue.get('file', 'N/A')}
*Line:* {issue.get('line', 'N/A')}
*Category:* {issue.get('category', issue.get('principle', 'N/A'))}

*Issue:*
{issue.get('message', 'N/A')}

*Suggested Fix:*
{issue.get('suggestion', 'N/A')}

_This ticket was automatically created by CodeGuard._
"""
            ticket = self.create_ticket(
                summary     = summary,
                description = description,
                issue_type  = "Bug" if severity == "CRITICAL" else "Task",
                priority    = priority,
                labels      = ["codeguard", severity.lower(), "pr-review"]
            )
            tickets.append(ticket)

        return tickets