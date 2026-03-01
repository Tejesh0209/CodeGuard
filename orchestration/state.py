from typing import TypedDict, List, Annotated, Optional
import operator


class CodeGuardState(TypedDict):
    # ── Input ────────────────────────────────────────────
    repo_name   : str
    pr_number   : int
    pr_title    : str
    pr_author   : str
    diff_chunks : List[dict]

    # ── Agent Findings ───────────────────────────────────
    style_review    : Optional[dict]
    security_review : Optional[dict]
    perf_review     : Optional[dict]
    arch_review     : Optional[dict]

    # ── Control Flow ─────────────────────────────────────
    severity_level  : str
    should_autofix  : bool
    next_action     : str   # "normal", "high", "critical"

    # ── Output ───────────────────────────────────────────
    final_report    : Optional[dict]
    jira_tickets    : Optional[List[dict]]
    github_comment  : Optional[str]

    # ── Message Log ──────────────────────────────────────
    messages : Annotated[List[str], operator.add]