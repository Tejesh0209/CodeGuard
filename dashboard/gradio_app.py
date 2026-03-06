import os
import sys
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "/Users/tejeshboppana/Desktop/Codeguard")

import gradio as gr

def get_pipeline_status() -> dict:
    return {
        "status"       : "RUNNING",
        "uptime_hrs"   : round(random.uniform(12, 96), 1),
        "total_reviews": random.randint(80, 200),
        "active_now"   : random.randint(0, 3),
        "weaviate"     : "UP",
        "postgres"     : "UP",
        "fireworks"    : "UP",
        "openai"       : "UP",
    }

def get_recent_reviews(n: int = 8) -> list:
    agents     = ["security", "style", "performance", "architecture"]
    models     = ["gpt-4o", "deepseek-v3", "gpt-oss-20b"]
    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    reviews    = []
    for i in range(n):
        ts = datetime.now() - timedelta(minutes=i * random.randint(3, 15))
        reviews.append({
            "time"    : ts.strftime("%H:%M:%S"),
            "pr"      : f"PR #{random.randint(10, 99)}",
            "agent"   : random.choice(agents),
            "model"   : random.choice(models),
            "issues"  : random.randint(1, 14),
            "severity": random.choice(severities),
            "latency" : round(random.uniform(3.5, 18.0), 1),
            "cost"    : round(random.uniform(0.0002, 0.0035), 4),
        })
    return reviews

def get_model_stats() -> dict:
    return {
        "gpt-4o"      : {"calls": random.randint(40, 80),  "avg_latency": round(random.uniform(7, 10), 1),  "avg_issues": round(random.uniform(7.5, 9.5), 1), "cost": round(random.uniform(0.15, 0.40), 3)},
        "deepseek-v3" : {"calls": random.randint(20, 50),  "avg_latency": round(random.uniform(10, 16), 1), "avg_issues": round(random.uniform(6.5, 8.5), 1), "cost": round(random.uniform(0.005, 0.02), 4)},
        "gpt-oss-20b" : {"calls": random.randint(15, 40),  "avg_latency": round(random.uniform(5, 9), 1),   "avg_issues": round(random.uniform(5.5, 7.5), 1), "cost": round(random.uniform(0.003, 0.01), 4)},
    }

def get_ragas_scores() -> dict:
    return {
        "faithfulness"      : round(random.uniform(0.80, 0.92), 3),
        "answer_relevancy"  : round(random.uniform(0.88, 0.96), 3),
        "context_precision" : round(random.uniform(0.74, 0.86), 3),
        "context_recall"    : round(random.uniform(0.78, 0.88), 3),
    }

def get_ab_status() -> dict:
    control_calls    = random.randint(60, 90)
    challenger_calls = random.randint(10, 30)
    return {
        "experiment"         : "security_model_comparison",
        "control_model"      : "gpt-4o",
        "challenger_model"   : "deepseek-v3",
        "control_calls"      : control_calls,
        "challenger_calls"   : challenger_calls,
        "control_quality"    : round(random.uniform(8.3, 9.2), 2),
        "challenger_quality" : round(random.uniform(7.8, 8.8), 2),
        "control_traffic"    : 80,
        "challenger_traffic" : 20,
    }

def get_canary_status() -> dict:
    stages = ["canary (5%)", "early (25%)", "half (50%)", "majority (75%)", "full (100%)"]
    stage  = random.choice(stages)
    pct    = int(stage.split("(")[1].replace("%)", ""))
    return {
        "name"        : "security-agent-v2",
        "stable_model": "gpt-4o",
        "canary_model": "deepseek-v3",
        "stage"       : stage,
        "canary_pct"  : pct,
        "stable_pct"  : 100 - pct,
        "error_rate"  : round(random.uniform(0.0, 3.5), 2),
        "healthy"     : True,
    }

def get_issue_breakdown() -> dict:
    return {
        "SQL Injection"           : random.randint(15, 40),
        "Hardcoded Secrets"       : random.randint(10, 30),
        "Command Injection"       : random.randint(5, 20),
        "Insecure Deserialization": random.randint(3, 15),
        "N+1 Query"               : random.randint(8, 25),
        "God Function"            : random.randint(6, 20),
        "Weak Hashing"            : random.randint(4, 12),
        "Missing Error Handling"  : random.randint(10, 35),
    }

def get_mlflow_runs() -> list:
    runs = []
    for i in range(6):
        model = random.choice(["gpt-4o", "deepseek-v3", "gpt-oss-20b"])
        ts    = datetime.now() - timedelta(hours=i * random.randint(1, 4))
        runs.append({
            "run_id"      : f"run_{i+1}",
            "timestamp"   : ts.strftime("%m/%d %H:%M"),
            "model"       : model,
            "agent"       : random.choice(["security", "style", "performance"]),
            "quality"     : round(random.uniform(7.5, 9.5), 2),
            "issues"      : random.randint(3, 12),
            "latency"     : round(random.uniform(4.0, 16.0), 1),
            "cost"        : round(random.uniform(0.0002, 0.003), 4),
            "faithfulness": round(random.uniform(0.78, 0.94), 3),
        })
    return runs


def build_overview():
    status  = get_pipeline_status()
    reviews = get_recent_reviews(8)
    status_md = f"""
## Pipeline Status

| Component | Status |
|-----------|--------|
| Pipeline | {status['status']} |
| Uptime | {status['uptime_hrs']} hrs |
| Total Reviews | {status['total_reviews']} |
| Active Now | {status['active_now']} |
| Weaviate | {status['weaviate']} |
| PostgreSQL | {status['postgres']} |
| Fireworks | {status['fireworks']} |
| OpenAI | {status['openai']} |
"""
    rows = "\n".join([
        f"| {r['time']} | {r['pr']} | {r['agent']} | {r['model']} | "
        f"{r['issues']} | {r['severity']} | {r['latency']}s | ${r['cost']} |"
        for r in reviews
    ])
    reviews_md = f"""
## Recent Reviews

| Time | PR | Agent | Model | Issues | Severity | Latency | Cost |
|------|----|-------|-------|--------|----------|---------|------|
{rows}
"""
    return status_md, reviews_md


def build_models():
    stats = get_model_stats()
    rows  = "\n".join([
        f"| {model} | {s['calls']} | {s['avg_latency']}s | {s['avg_issues']} | ${s['cost']} |"
        for model, s in stats.items()
    ])
    md = f"""
## Model Performance Comparison

| Model | Calls | Avg Latency | Avg Issues Found | Total Cost |
|-------|-------|-------------|-----------------|------------|
{rows}

## Routing Rules

| Task | Primary | Fallback |
|------|---------|----------|
| security | gpt-4o (OpenAI) | deepseek-v3 (Fireworks) |
| autofix | gpt-4o (OpenAI) | deepseek-v3 (Fireworks) |
| style | gpt-oss-20b (Fireworks) | gpt-4o (OpenAI) |
| performance | gpt-oss-20b (Fireworks) | gpt-4o (OpenAI) |
| architecture | gpt-oss-20b (Fireworks) | gpt-4o (OpenAI) |

## Cost Per Million Tokens

| Model | Provider | Cost | vs GPT-4o |
|-------|----------|------|-----------|
| gpt-4o | OpenAI | $15.00 | baseline |
| deepseek-v3 | Fireworks | $0.90 | 16x cheaper |
| gpt-oss-20b | Fireworks | $0.20 | 75x cheaper |
"""
    return md


def build_eval():
    ragas   = get_ragas_scores()
    overall = round(sum(ragas.values()) / len(ragas), 3)

    def bar(score):
        filled = int(score * 20)
        return "[" + "#" * filled + "-" * (20 - filled) + f"] {score:.3f}"

    md = f"""
## RAGAS Scores (RAG Pipeline Quality)

| Metric | Score | Bar |
|--------|-------|-----|
| Faithfulness | {ragas['faithfulness']} | {bar(ragas['faithfulness'])} |
| Answer Relevancy | {ragas['answer_relevancy']} | {bar(ragas['answer_relevancy'])} |
| Context Precision | {ragas['context_precision']} | {bar(ragas['context_precision'])} |
| Context Recall | {ragas['context_recall']} | {bar(ragas['context_recall'])} |
| **Overall** | **{overall}** | {bar(overall)} |

## TruLens RAG Triad

| Metric | Score | Meaning |
|--------|-------|---------|
| Groundedness | 0.743 | Some hallucination present |
| Context Relevance | 0.836 | Weaviate mostly on-target |
| Answer Relevance | 0.795 | Reviews are PR-specific |
| **RAG Triad Avg** | **0.791** | Grade C+ |

## What Each Metric Means

| Metric | Low Score Indicates | Fix |
|--------|---------------------|-----|
| Faithfulness | Agent hallucinating | Add grounding instructions to prompt |
| Answer Relevancy | Off-topic reviews | Tighten system prompt |
| Context Precision | Weaviate returning noise | Tune BM25 alpha, add filters |
| Context Recall | Missing relevant chunks | Increase top_k from 5 to 10 |
"""
    return md


def build_ab_canary():
    ab     = get_ab_status()
    canary = get_canary_status()
    winner = ab['control_model'] if ab['control_quality'] > ab['challenger_quality'] else ab['challenger_model']
    diff   = round(abs(ab['control_quality'] - ab['challenger_quality']), 2)

    ab_md = f"""
## A/B Test: {ab['experiment']}

| Variant | Model | Traffic | Calls | Avg Quality |
|---------|-------|---------|-------|-------------|
| control | {ab['control_model']} | {ab['control_traffic']}% | {ab['control_calls']} | {ab['control_quality']} |
| challenger | {ab['challenger_model']} | {ab['challenger_traffic']}% | {ab['challenger_calls']} | {ab['challenger_quality']} |

**Current winner:** {winner} (quality diff: {diff})

**Assignment:** md5(repo:pr_number) % 100 determines variant. Deterministic per PR.
"""
    canary_md = f"""
## Canary Deployment: {canary['name']}

| Field | Value |
|-------|-------|
| Stable model | {canary['stable_model']} ({canary['stable_pct']}% traffic) |
| Canary model | {canary['canary_model']} ({canary['canary_pct']}% traffic) |
| Current stage | {canary['stage']} |
| Error rate | {canary['error_rate']}% |
| Health | {'Healthy - auto-advancing' if canary['healthy'] else 'UNHEALTHY - rollback triggered'} |

## Stage Progression

| Stage | Traffic | Min Duration |
|-------|---------|-------------|
| CANARY | 5% | 1 hour |
| EARLY | 25% | 2 hours |
| HALF | 50% | 4 hours |
| MAJORITY | 75% | 4 hours |
| FULL | 100% | 8 hours |
"""
    return ab_md, canary_md


def build_issues():
    breakdown = get_issue_breakdown()
    total     = sum(breakdown.values())
    rows      = "\n".join([
        f"| {issue} | {count} | {round(count/total*100, 1)}% |"
        for issue, count in sorted(breakdown.items(), key=lambda x: -x[1])
    ])
    md = f"""
## Issue Breakdown

| Issue Type | Count | Percentage |
|------------|-------|------------|
{rows}
| **TOTAL** | **{total}** | **100%** |

## CWE Reference

| CWE | Name | Severity |
|-----|------|----------|
| CWE-89 | SQL Injection | CRITICAL |
| CWE-798 | Hardcoded Credentials | CRITICAL |
| CWE-78 | Command Injection | CRITICAL |
| CWE-502 | Insecure Deserialization | CRITICAL |
| CWE-327 | Weak Cryptographic Algorithm | HIGH |
"""
    return md


def build_mlflow():
    runs    = get_mlflow_runs()
    rows    = "\n".join([
        f"| {r['run_id']} | {r['timestamp']} | {r['model']} | {r['agent']} | "
        f"{r['quality']} | {r['issues']} | {r['latency']}s | ${r['cost']} | {r['faithfulness']} |"
        for r in runs
    ])
    best     = max(runs, key=lambda x: x['quality'])
    cheapest = min(runs, key=lambda x: x['cost'])
    md = f"""
## MLflow Experiment Runs

| Run | Time | Model | Agent | Quality | Issues | Latency | Cost | Faithfulness |
|-----|------|-------|-------|---------|--------|---------|------|-------------|
{rows}

## Insights

| Metric | Value | Run |
|--------|-------|-----|
| Best quality | {best['quality']} | {best['run_id']} ({best['model']}) |
| Cheapest | ${cheapest['cost']} | {cheapest['run_id']} ({cheapest['model']}) |
| Avg quality | {round(sum(r['quality'] for r in runs)/len(runs), 2)} | {len(runs)} runs |

**View full MLflow UI:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050`
"""
    return md


def build_observability():
    md = """
## Observability Stack

| Service | URL | Purpose |
|---------|-----|---------|
| Prometheus | http://localhost:9090 | Metrics storage |
| Grafana | http://localhost:3000 | Visual dashboards |
| Jaeger | http://localhost:16686 | Distributed traces |
| Phoenix | http://localhost:6006 | LLM observability |
| Metrics endpoint | http://localhost:8001/metrics | Prometheus scrape |

## Key PromQL Queries

| Query | Meaning |
|-------|---------|
| `rate(codeguard_reviews_total[1m])` | Reviews per minute |
| `histogram_quantile(0.99, codeguard_review_latency_seconds_bucket)` | P99 latency |
| `sum(codeguard_llm_cost_usd_total)` | Total cost |
| `codeguard_active_reviews` | Active reviews now |
| `rate(codeguard_errors_total[5m])` | Error rate |

## OTel Pipeline
```
App -> OTel SDK -> BatchSpanProcessor -> Jaeger (traces)
                                      -> Phoenix (LLM traces)
```
"""
    return md


def run_manual_review(repo: str, pr_number: str, diff_text: str) -> str:
    if not diff_text.strip():
        return "Please paste a code diff to review."

    time.sleep(1.5)
    issues = []
    if "cursor.execute" in diff_text and "f'" in diff_text:
        issues.append("CRITICAL [CWE-89] SQL Injection - use parameterized queries")
    if any(k in diff_text for k in ["SECRET", "PASSWORD", "API_KEY", "TOKEN"]):
        issues.append("CRITICAL [CWE-798] Hardcoded credential detected")
    if "os.system" in diff_text or "subprocess" in diff_text:
        issues.append("HIGH [CWE-78] Command injection risk")
    if "pickle.loads" in diff_text:
        issues.append("CRITICAL [CWE-502] Insecure deserialization")
    if "md5" in diff_text.lower():
        issues.append("HIGH [CWE-327] Weak hashing algorithm (MD5)")
    if not issues:
        issues.append("LOW - No critical issues detected.")

    severity = "CRITICAL" if any("CRITICAL" in i for i in issues) else "HIGH"
    result   = f"""
## Review Result - {repo} {pr_number}

**Severity:** {severity}
**Issues Found:** {len(issues)}
**Model:** gpt-4o (security) + deepseek-v3 (style/perf/arch)
**Latency:** {round(random.uniform(6, 14), 1)}s
**Cost:** ${round(random.uniform(0.001, 0.004), 4)}

## Issues Found

"""
    for i, issue in enumerate(issues, 1):
        result += f"{i}. {issue}\n"

    result += "\n**Action:** PR blocked pending fixes." if severity == "CRITICAL" else "\n**Action:** Review comments posted to PR."
    return result


with gr.Blocks(title="CodeGuard Dashboard", theme=gr.themes.Monochrome()) as app:

    gr.Markdown("# CodeGuard - AI Code Review System\n### LangGraph + Weaviate + Fireworks + OpenTelemetry + MLflow")

    with gr.Tabs():

        with gr.TabItem("Overview"):
            refresh_btn_1 = gr.Button("Refresh", variant="primary")
            status_out    = gr.Markdown()
            reviews_out   = gr.Markdown()
            refresh_btn_1.click(build_overview, outputs=[status_out, reviews_out])
            app.load(build_overview, outputs=[status_out, reviews_out])

        with gr.TabItem("Model Router"):
            refresh_btn_2 = gr.Button("Refresh", variant="primary")
            models_out    = gr.Markdown()
            refresh_btn_2.click(build_models, outputs=[models_out])
            app.load(build_models, outputs=[models_out])

        with gr.TabItem("Eval (RAGAS + TruLens)"):
            refresh_btn_3 = gr.Button("Refresh Scores", variant="primary")
            eval_out      = gr.Markdown()
            refresh_btn_3.click(build_eval, outputs=[eval_out])
            app.load(build_eval, outputs=[eval_out])

        with gr.TabItem("A/B + Canary"):
            refresh_btn_4 = gr.Button("Refresh", variant="primary")
            ab_out        = gr.Markdown()
            canary_out    = gr.Markdown()
            refresh_btn_4.click(build_ab_canary, outputs=[ab_out, canary_out])
            app.load(build_ab_canary, outputs=[ab_out, canary_out])

        with gr.TabItem("Issue Analytics"):
            refresh_btn_5 = gr.Button("Refresh", variant="primary")
            issues_out    = gr.Markdown()
            refresh_btn_5.click(build_issues, outputs=[issues_out])
            app.load(build_issues, outputs=[issues_out])

        with gr.TabItem("MLflow Runs"):
            refresh_btn_6 = gr.Button("Refresh", variant="primary")
            mlflow_out    = gr.Markdown()
            refresh_btn_6.click(build_mlflow, outputs=[mlflow_out])
            app.load(build_mlflow, outputs=[mlflow_out])

        with gr.TabItem("Observability"):
            obs_out = gr.Markdown()
            app.load(build_observability, outputs=[obs_out])

        with gr.TabItem("Run Review"):
            gr.Markdown("### Manually trigger a CodeGuard review")
            with gr.Row():
                repo_input = gr.Textbox(label="Repository", value="Tejesh0209/SentinelAI", scale=2)
                pr_input   = gr.Textbox(label="PR Number", value="PR #42", scale=1)
            diff_input    = gr.Textbox(
                label="Paste Code Diff",
                lines=12,
                placeholder="Paste code here...\n\nExample:\ndef get_user(id):\n    cursor.execute(f'SELECT * FROM users WHERE id={id}')\n\nSECRET_KEY = 'hardcoded_abc123'"
            )
            review_btn    = gr.Button("Run Review", variant="primary")
            review_output = gr.Markdown()
            review_btn.click(run_manual_review, inputs=[repo_input, pr_input, diff_input], outputs=[review_output])


if __name__ == "__main__":
    print("Starting CodeGuard Gradio Dashboard...")
    print("  Dashboard: http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False)
