import asyncio
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from orchestration.state import CodeGuardState
from agents.style_agent import StyleAgent
from agents.security_agent import SecurityAgent
from agents.performance_agent import PerformanceAgent
from agents.arch_agent import ArchAgent
from tools.jira_client import JiraClient


# ── Agent Nodes ──────────────────────────────────────────────────

async def parallel_review_node(state: CodeGuardState) -> dict:
    """Run all 4 agents in parallel."""
    print("\n🚀 Running all 4 agents in parallel...")

    style_result, security_result, perf_result, arch_result = await asyncio.gather(
        StyleAgent().review(state["diff_chunks"]),
        SecurityAgent().review(state["diff_chunks"]),
        PerformanceAgent().review(state["diff_chunks"]),
        ArchAgent().review(state["diff_chunks"])
    )

    has_critical = security_result.get("has_critical", False)
    risk_score   = security_result.get("risk_score", 0)

    if has_critical:
        severity = "CRITICAL"
    elif risk_score >= 7:
        severity = "HIGH"
    elif risk_score >= 4:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "style_review"    : style_result,
        "security_review" : security_result,
        "perf_review"     : perf_result,
        "arch_review"     : arch_result,
        "severity_level"  : severity,
        "should_autofix"  : has_critical,
        "messages"        : [
            f"Style     : {style_result.get('overall_score')}/10",
            f"Security  : {security_result.get('risk_score')}/10 risk",
            f"Perf      : {perf_result.get('perf_score')}/10",
            f"Arch      : {arch_result.get('arch_score')}/10",
            f"Severity  : {severity}"
        ]
    }


async def supervisor_node(state: CodeGuardState) -> dict:
    """
    Supervisor decides routing based on severity.
    Sets next_action field to guide conditional edges.
    """
    severity = state.get("severity_level", "LOW")
    print(f"\n🎯 Supervisor: severity={severity}")

    if severity == "CRITICAL":
        print("   → Routing to: JIRA ticket + aggregator")
        next_action = "critical"
    elif severity == "HIGH":
        print("   → Routing to: JIRA ticket + aggregator")
        next_action = "high"
    else:
        print("   → Routing to: aggregator only")
        next_action = "normal"

    return {
        "next_action" : next_action,
        "messages"    : [f"Supervisor decision: {next_action}"]
    }


async def jira_node(state: CodeGuardState) -> dict:
    """Create JIRA tickets for HIGH and CRITICAL issues."""
    print("\nCreating JIRA tickets...")

    final_report = state.get("final_report") or _build_report(state)
    jira         = JiraClient()
    tickets      = jira.create_tickets_from_review(
        repo_name    = state["repo_name"],
        pr_number    = state["pr_number"],
        pr_title     = state["pr_title"],
        final_report = final_report
    )

    return {
        "jira_tickets" : tickets,
        "messages"     : [f"Created {len(tickets)} JIRA tickets"]
    }


async def aggregator_node(state: CodeGuardState) -> dict:
    """Merge all findings into unified final report."""
    print("\nAggregating findings...")

    report = _build_report(state)

    print(f"\n{'='*60}")
    print(f"FINAL REPORT")
    print(f"{'='*60}")
    print(f"   Total Issues : {report['summary']['total_issues']}")
    print(f"   Critical  : {report['summary']['critical']}")
    print(f"   High      : {report['summary']['high']}")
    print(f"   Medium    : {report['summary']['medium']}")
    print(f"   Low       : {report['summary']['low']}")
    print(f"   Approved  : {report['approved']}")
    print(f"{'='*60}\n")

    return {
        "final_report" : report,
        "messages"     : [f"Final: {report['summary']['total_issues']} issues"]
    }


async def post_comment_node(state: CodeGuardState) -> dict:
    """Post unified review comment back to GitHub PR."""
    from api.github_client import post_pr_comment

    report  = state.get("final_report", {})
    comment = _format_github_comment(state, report)

    print(f"\nPosting review comment to GitHub PR #{state['pr_number']}...")
    await post_pr_comment(
        repo_name  = state["repo_name"],
        pr_number  = state["pr_number"],
        comment    = comment
    )
    print(f"Comment posted!")

    return {
        "messages": ["GitHub comment posted"]
    }


# ── Helper Functions ─────────────────────────────────────────────

def _build_report(state: CodeGuardState) -> dict:
    style_issues    = state.get("style_review",    {}).get("issues", []) or []
    security_issues = state.get("security_review", {}).get("issues", []) or []
    perf_issues     = state.get("perf_review",     {}).get("issues", []) or []
    arch_issues     = state.get("arch_review",     {}).get("issues", []) or []

    all_issues = style_issues + security_issues + perf_issues + arch_issues
    critical   = [i for i in all_issues if i.get("severity") == "CRITICAL"]
    high       = [i for i in all_issues if i.get("severity") == "HIGH"]
    medium     = [i for i in all_issues if i.get("severity") == "MEDIUM"]
    low        = [i for i in all_issues if i.get("severity") == "LOW"]

    return {
        "repo"            : state["repo_name"],
        "pr_number"       : state["pr_number"],
        "pr_title"        : state["pr_title"],
        "severity"        : state.get("severity_level", "LOW"),
        "approved"        : len(critical) == 0 and len(high) == 0,
        "summary"         : {
            "total_issues"   : len(all_issues),
            "critical"       : len(critical),
            "high"           : len(high),
            "medium"         : len(medium),
            "low"            : len(low),
            "style_score"    : state.get("style_review",    {}).get("overall_score", 0) or 0,
            "security_score" : state.get("security_review", {}).get("risk_score",    0) or 0,
            "perf_score"     : state.get("perf_review",     {}).get("perf_score",    0) or 0,
            "arch_score"     : state.get("arch_review",     {}).get("arch_score",    0) or 0,
        },
        "style_issues"    : style_issues,
        "security_issues" : security_issues,
        "perf_issues"     : perf_issues,
        "arch_issues"     : arch_issues
    }


def _format_github_comment(state: CodeGuardState, report: dict) -> str:
    summary  = report.get("summary", {})
    approved = report.get("approved", False)
    verdict  = "APPROVED" if approved else "CHANGES REQUESTED"
    severity = report.get("severity", "LOW")

    severity_emoji = {
        "CRITICAL": "🚨", "HIGH": "🔴",
        "MEDIUM"  : "🟡", "LOW" : "🟢"
    }.get(severity, "⚪")

    comment = f"""## CodeGuard Automated Review

### {verdict} {severity_emoji} Severity: {severity}

| Agent | Score | Issues |
|-------|-------|--------|
| Style | {summary.get('style_score', 0)}/10 | {len(report.get('style_issues', []))} |
| Security | {summary.get('security_score', 0)}/10 risk | {len(report.get('security_issues', []))} |
| Performance | {summary.get('perf_score', 0)}/10 | {len(report.get('perf_issues', []))} |
| Architecture | {summary.get('arch_score', 0)}/10 | {len(report.get('arch_issues', []))} |

### Summary
- Critical: **{summary.get('critical', 0)}**
- High: **{summary.get('high', 0)}**
- Medium: **{summary.get('medium', 0)}**
- Low: **{summary.get('low', 0)}**
- Total: **{summary.get('total_issues', 0)}**
"""

    # Add critical issues prominently
    critical_issues = [
        i for i in (
            report.get("security_issues", []) +
            report.get("perf_issues",     []) +
            report.get("style_issues",    []) +
            report.get("arch_issues",     [])
        )
        if i.get("severity") == "CRITICAL"
    ]

    if critical_issues:
        comment += "\n### Critical Issues (Must Fix)\n"
        for issue in critical_issues:
            comment += f"""
> **{issue.get('file')}** line {issue.get('line')}
> {issue.get('message')}
>  Fix: {issue.get('suggestion')}
"""

    # Add high issues
    high_issues = [
        i for i in (
            report.get("security_issues", []) +
            report.get("perf_issues",     []) +
            report.get("style_issues",    []) +
            report.get("arch_issues",     [])
        )
        if i.get("severity") == "HIGH"
    ]

    if high_issues:
        comment += "\n### High Issues (Should Fix)\n"
        for issue in high_issues[:5]:  # max 5 to avoid huge comments
            comment += f"- **{issue.get('file')}:{issue.get('line')}** — {issue.get('message')}\n"

    comment += "\n---\n*Reviewed by [CodeGuard](https://github.com/Tejesh0209/CodeGuard)*"
    return comment


# ── Conditional Routing ──────────────────────────────────────────

def route_after_supervisor(state: CodeGuardState) -> Literal["jira_node", "aggregator"]:
    next_action = state.get("next_action", "normal")
    if next_action in ["critical", "high"]:
        return "jira_node"
    return "aggregator"


# ── Build Graph ──────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(CodeGuardState)

    # Add all nodes
    graph.add_node("parallel_review", parallel_review_node)
    graph.add_node("supervisor",      supervisor_node)
    graph.add_node("jira_node",       jira_node)
    graph.add_node("aggregator",      aggregator_node)
    graph.add_node("post_comment",    post_comment_node)

    # Entry point
    graph.set_entry_point("parallel_review")

    # Edges
    graph.add_edge("parallel_review", "supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "jira_node"  : "jira_node",
            "aggregator" : "aggregator"
        }
    )

    # After JIRA → always go to aggregator
    graph.add_edge("jira_node",    "aggregator")
    graph.add_edge("aggregator",   "post_comment")
    graph.add_edge("post_comment", END)

    checkpointer = MemorySaver()
    compiled     = graph.compile(checkpointer=checkpointer)
    print("LangGraph compiled — 5 nodes, conditional routing")
    return compiled


codeguard_graph = build_graph()