import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from agents.agent import Agent
from agents.run import Runner
from agents.tool import function_tool
from tools.slack_client import SlackClient
from tools.sentry_client import SentryClient
from agents.mcp import MCPServerStdio
from agents.mcp.server import MCPServerStdioParams
import httpx

load_dotenv()

NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "")
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_APP_PASSWORD", "")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://codeguard:codeguard123@localhost:5432/codeguard_db"
)
DD_API_KEY   = os.getenv("DD_API_KEY", "")
DD_APP_KEY   = os.getenv("DD_APP_KEY", "")
DD_SITE      = os.getenv("DD_SITE", "datadoghq.com")


# ── Tools ────────────────────────────────────────────────────────

@function_tool
def send_slack_alert(message: str, severity: str) -> str:
    """Send an alert to the Slack channel for the given severity level."""
    client = SlackClient()
    result = client.send_alert(message, severity)
    return f"Slack alert sent: {result.get('status')}"


@function_tool
def send_slack_summary(
    repo_name   : str,
    pr_number   : int,
    pr_title    : str,
    total_issues: int,
    critical    : int,
    high        : int,
    severity    : str,
    approved    : bool
) -> str:
    """Send a full PR review summary to the Slack channel."""
    client = SlackClient()
    final_report = {
        "severity": severity,
        "approved": approved,
        "summary" : {
            "total_issues": total_issues,
            "critical"    : critical,
            "high"        : high
        }
    }
    result = client.send_review_summary(repo_name, pr_number, pr_title, final_report)
    return f"Slack summary sent: {result.get('status')}"


@function_tool
def get_sentry_errors(filename: str) -> str:
    """Fetch recent production errors from Sentry for a given filename."""
    client = SentryClient()
    errors = client.get_recent_errors(filename=filename)
    return client.format_for_context(errors)


@function_tool
def get_all_sentry_errors() -> str:
    """Fetch all recent production errors from Sentry."""
    client = SentryClient()
    errors = client.get_recent_errors()
    return client.format_for_context(errors)


@function_tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification. Used for CRITICAL severity alerts."""
    if not SMTP_USER or not SMTP_PASS:
        return "SKIPPED: SMTP_USER or SMTP_APP_PASSWORD not configured in .env"
    try:
        msg = MIMEMultipart()
        msg["From"]    = SMTP_USER
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return f"SUCCESS: Email sent to {to}"
    except Exception as e:
        return f"ERROR: Failed to send email - {e}"


@function_tool
def query_datadog(service: str, metric_type: str = "error_rate") -> str:
    """Query DataDog for service metrics. metric_type: 'error_rate' or 'p99_latency'."""
    if not DD_API_KEY or not DD_APP_KEY:
        return "SKIPPED: DD_API_KEY or DD_APP_KEY not configured in .env"
    try:
        import time
        now  = int(time.time())
        past = now - 3600  # last 1 hour

        if metric_type == "p99_latency":
            query = f"p99:trace.web.request{{service:{service}}}"
        else:
            query = f"sum:trace.web.request.errors{{service:{service}}}.as_rate()"

        resp = httpx.get(
            f"https://api.{DD_SITE}/api/v1/query",
            headers={
                "DD-API-KEY":         DD_API_KEY,
                "DD-APPLICATION-KEY": DD_APP_KEY
            },
            params={"from": past, "to": now, "query": query}
        )
        if resp.status_code == 200:
            data   = resp.json()
            series = data.get("series", [])
            if series and series[0].get("pointlist"):
                points = series[0]["pointlist"]
                latest = points[-1][1] if points else "no data"
                return f"{metric_type} for {service} (last 1h): {latest}"
            return f"No {metric_type} data found for service '{service}'"
        return f"ERROR: DataDog API returned {resp.status_code}"
    except Exception as e:
        return f"ERROR: DataDog query failed - {e}"


# ── Instructions ─────────────────────────────────────────────────

NOTIFICATION_INSTRUCTIONS = f"""You are CodeGuard's Notification Agent.

Your job is to:
1. Send Slack alerts based on review severity
2. Check Sentry for production errors related to changed files
3. Query DataDog for latency and error rate metrics on changed services
4. Send email notifications for CRITICAL issues

Rules:
- ALWAYS send a Slack summary for every review
- For CRITICAL severity: send a Slack ALERT first, then the summary
- For CRITICAL severity: also send an email to {NOTIFY_EMAIL} using send_email
- Always check Sentry for production errors on files with CRITICAL issues
- Query DataDog using query_datadog for services touched by the PR
- Include Sentry error counts AND DataDog metrics in notifications

DataDog queries to run for CRITICAL/HIGH:
  - query_datadog(service, "error_rate") for error rate in the last 1 hour
  - query_datadog(service, "p99_latency") for P99 latency in the last 1 hour
  - Include these metrics in the Slack message and email

Email format for CRITICAL:
  Subject: [CodeGuard CRITICAL] PR #{{pr_number}} in {{repo}} needs immediate attention
  Body: List all critical issues, files affected, Sentry errors,
        DataDog metrics, and suggested fixes.

Slack message format:
  Severity emoji + verdict + PR info + issue counts + agent scores + metrics
"""


# ── Runner ───────────────────────────────────────────────────────

async def run_notifications(
    repo_name   : str,
    pr_number   : int,
    pr_title    : str,
    pr_author   : str,
    final_report: dict
) -> dict:
    """
    Run the Notification Agent after every review.
    Sends Slack alerts, checks Sentry, queries DataDog, emails on CRITICAL.
    """
    summary  = final_report.get("summary", {})
    severity = final_report.get("severity", "LOW")
    approved = final_report.get("approved", False)

    critical_files = list(set([
        i.get("file", "")
        for i in (
            final_report.get("security_issues", []) +
            final_report.get("style_issues",    []) +
            final_report.get("perf_issues",     []) +
            final_report.get("arch_issues",     [])
        )
        if i.get("severity") == "CRITICAL" and i.get("file")
    ]))

    print(f"\n   Notification Agent starting...")
    print(f"   Severity: {severity} | Critical files: {critical_files}")

    prompt = f"""
        Send notifications for this CodeGuard review:

        Repository     : {repo_name}
        PR Number      : #{pr_number}
        PR Title       : {pr_title}
        PR Author      : {pr_author}
        Severity       : {severity}
        Approved       : {approved}
        Total Issues   : {summary.get('total_issues', 0)}
        Critical       : {summary.get('critical', 0)}
        High           : {summary.get('high', 0)}
        Medium         : {summary.get('medium', 0)}
        Low            : {summary.get('low', 0)}
        Style Score    : {summary.get('style_score', 0)}/10
        Security Score : {summary.get('security_score', 0)}/10
        Perf Score     : {summary.get('perf_score', 0)}/10
        Arch Score     : {summary.get('arch_score', 0)}/10

        Files with CRITICAL issues: {', '.join(critical_files) if critical_files else 'none'}

        Please:
        1. {'Send a CRITICAL Slack alert first, then ' if severity == 'CRITICAL' else ''}Send a Slack summary
        2. {'Check Sentry for errors in: ' + ', '.join(critical_files) if critical_files else 'Check Sentry for any recent errors'}
        3. {'Query DataDog for error rate and P99 latency of affected services' if severity in ['CRITICAL', 'HIGH'] else 'Skip DataDog for this severity'}
        4. {'Send an email to ' + NOTIFY_EMAIL + ' with full CRITICAL details including Sentry + DataDog data' if severity == 'CRITICAL' else 'No email needed for this severity'}
        """

    postgres_mcp = MCPServerStdio(
        params=MCPServerStdioParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres", DATABASE_URL]
        ),
        name="postgresql-mcp"
    )

    async with postgres_mcp:
        agent = Agent(
            name         = "NotificationAgent",
            instructions = NOTIFICATION_INSTRUCTIONS,
            model        = "gpt-4o",
            tools        = [
                send_slack_alert,
                send_slack_summary,
                get_sentry_errors,
                get_all_sentry_errors,
                send_email,
                query_datadog
            ],
            mcp_servers  = [postgres_mcp]
        )
        result = await Runner.run(agent, prompt)

    print(f"\n   Notifications complete:")
    print(f"   {result.final_output[:200]}")

    return {
        "status"  : "completed",
        "summary" : result.final_output,
        "severity": severity
    }