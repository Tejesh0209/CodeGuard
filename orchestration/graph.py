import asyncio
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from orchestration.state import CodeGuardState
from agents.style_agent import StyleAgent
from agents.security_agent import SecurityAgent
from agents.performance_agent import PerformanceAgent


# ── Agent Nodes ──────────────────────────────────────────────────

async def style_node(state: CodeGuardState) -> dict:
    """Style Agent node."""
    print("\n🎨 Style node started")
    agent  = StyleAgent()
    review = await agent.review(state["diff_chunks"])
    return {
        "style_review" : review,
        "messages"     : [f"Style review complete — score: {review.get('overall_score')}/10"]
    }


async def security_node(state: CodeGuardState) -> dict:
    """Security Agent node."""
    print("\n🔒 Security node started")
    agent  = SecurityAgent()
    review = await agent.review(state["diff_chunks"])
    return {
        "security_review" : review,
        "has_critical"    : review.get("has_critical", False),
        "messages"        : [f"Security review complete — risk: {review.get('risk_score')}/10"]
    }


async def perf_node(state: CodeGuardState) -> dict:
    """Performance Agent node."""
    print("\n⚡ Performance node started")
    agent  = PerformanceAgent()
    review = await agent.review(state["diff_chunks"])
    return {
        "perf_review" : review,
        "messages"    : [f"Performance review complete — score: {review.get('perf_score')}/10"]
    }


async def parallel_review_node(state: CodeGuardState) -> dict:
    """
    Run all 3 agents in PARALLEL using asyncio.gather.
    All agents start at the same time — total time = slowest agent
    instead of sum of all agents.
    """
    print("\nRunning all agents in parallel...")

    # asyncio.gather runs all 3 coroutines simultaneously
    style_result, security_result, perf_result = await asyncio.gather(
        StyleAgent().review(state["diff_chunks"]),
        SecurityAgent().review(state["diff_chunks"]),
        PerformanceAgent().review(state["diff_chunks"])
    )

    # Determine overall severity
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
        "severity_level"  : severity,
        "should_autofix"  : has_critical,
        "messages"        : [
            f"Style     : {style_result.get('overall_score')}/10",
            f"Security  : {security_result.get('risk_score')}/10 risk",
            f"Perf      : {perf_result.get('perf_score')}/10",
            f"Severity  : {severity}"
        ]
    }


async def aggregator_node(state: CodeGuardState) -> dict:
    """
    Merge findings from all agents into a single final report.
    """
    print("\nAggregating findings...")

    style_issues    = state.get("style_review",    {}).get("issues", [])
    security_issues = state.get("security_review", {}).get("issues", [])
    perf_issues     = state.get("perf_review",     {}).get("issues", [])

    # Count by severity
    all_issues = style_issues + security_issues + perf_issues
    critical   = [i for i in all_issues if i.get("severity") == "CRITICAL"]
    high       = [i for i in all_issues if i.get("severity") == "HIGH"]
    medium     = [i for i in all_issues if i.get("severity") == "MEDIUM"]
    low        = [i for i in all_issues if i.get("severity") == "LOW"]

    final_report = {
        "repo"       : state["repo_name"],
        "pr_number"  : state["pr_number"],
        "pr_title"   : state["pr_title"],
        "severity"   : state.get("severity_level", "LOW"),
        "summary"    : {
            "total_issues"    : len(all_issues),
            "critical"        : len(critical),
            "high"            : len(high),
            "medium"          : len(medium),
            "low"             : len(low),
            "style_score"     : state.get("style_review",    {}).get("overall_score", 0),
            "security_score"  : state.get("security_review", {}).get("risk_score",    0),
            "perf_score"      : state.get("perf_review",     {}).get("perf_score",    0),
        },
        "style_issues"    : style_issues,
        "security_issues" : security_issues,
        "perf_issues"     : perf_issues,
        "approved"        : len(critical) == 0 and len(high) == 0
    }

    print(f"\n{'='*60}")
    print(f"📊 FINAL REPORT")
    print(f"{'='*60}")
    print(f"   Total Issues : {len(all_issues)}")
    print(f"   🚨 Critical  : {len(critical)}")
    print(f"   🔴 High      : {len(high)}")
    print(f"   🟡 Medium    : {len(medium)}")
    print(f"   🟢 Low       : {len(low)}")
    print(f"   ✅ Approved  : {final_report['approved']}")
    print(f"{'='*60}\n")

    return {
        "final_report" : final_report,
        "messages"     : [f"📊 Final report: {len(all_issues)} total issues"]
    }


# ── Conditional Routing ──────────────────────────────────────────

def route_after_review(state: CodeGuardState) -> Literal["aggregator", "aggregator"]:
    """
    Route after parallel review.
    Currently always goes to aggregator.
    Day 7: CRITICAL → autofix node instead.
    """
    return "aggregator"


# ── Build the Graph ──────────────────────────────────────────────

def build_graph():
    graph = StateGraph(CodeGuardState)

    # Add nodes
    graph.add_node("parallel_review", parallel_review_node)
    graph.add_node("aggregator",      aggregator_node)

    # Set entry point
    graph.set_entry_point("parallel_review")

    # Add edges
    graph.add_edge("parallel_review", "aggregator")
    graph.add_edge("aggregator",      END)

    # Compile with in-memory checkpointer
    checkpointer = MemorySaver()
    compiled     = graph.compile(checkpointer=checkpointer)

    print("LangGraph compiled successfully")
    return compiled


# Singleton — build once, reuse
codeguard_graph = build_graph()