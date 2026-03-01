# orchestration/state.py
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

    # ── Control Flow ─────────────────────────────────────
    severity_level  : str   # LOW, MEDIUM, HIGH, CRITICAL
    should_autofix  : bool

    # ── Output ───────────────────────────────────────────
    final_report    : Optional[dict]

    # ── Message Log (accumulates across nodes) ───────────
    messages : Annotated[List[str], operator.add]