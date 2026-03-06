"""
Prometheus — Metrics Collection & Alerting

WHY PROMETHEUS:
  OTel/Jaeger/Phoenix handle TRACES (what happened in one request).
  Prometheus handles METRICS (aggregate numbers over time).

  Traces answer: "Why was PR #42 review slow?"
  Metrics answer: "How many reviews per minute? What's p99 latency?
                   How many CRITICAL issues found today?"

  Prometheus scrapes your app on a schedule (every 15s).
  Grafana reads from Prometheus and draws dashboards.

METRIC TYPES:
  Counter:   only goes up (total reviews, total errors)
  Gauge:     can go up or down (active reviews, queue size)
  Histogram: distribution (latency buckets: <1s, <5s, <10s, <30s)
  Summary:   percentiles (p50, p90, p99 latency)
"""
import time
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
    CollectorRegistry,
    REGISTRY
)


# ── Define all CodeGuard metrics ──────────────────────────────────

# COUNTERS (never decrease)
REVIEWS_TOTAL = Counter(
    "codeguard_reviews_total",
    "Total number of PR reviews processed",
    labelnames=["agent", "model", "provider", "severity"]
)

ISSUES_FOUND = Counter(
    "codeguard_issues_found_total",
    "Total number of issues found across all reviews",
    labelnames=["agent", "severity", "issue_type"]
)

ERRORS_TOTAL = Counter(
    "codeguard_errors_total",
    "Total number of errors in review pipeline",
    labelnames=["agent", "error_type"]
)

LLM_TOKENS_USED = Counter(
    "codeguard_llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["model", "provider", "token_type"]  # token_type: input/output
)

LLM_COST_USD = Counter(
    "codeguard_llm_cost_usd_total",
    "Total LLM cost in USD",
    labelnames=["model", "provider"]
)

# GAUGES (can go up and down)
ACTIVE_REVIEWS = Gauge(
    "codeguard_active_reviews",
    "Number of reviews currently in progress"
)

WEAVIATE_CHUNKS = Gauge(
    "codeguard_weaviate_chunks_total",
    "Total chunks stored in Weaviate",
    labelnames=["repo"]
)

CANARY_TRAFFIC_PCT = Gauge(
    "codeguard_canary_traffic_percent",
    "Current canary deployment traffic percentage",
    labelnames=["deployment_name", "variant"]
)

AB_TEST_CALLS = Gauge(
    "codeguard_ab_test_calls",
    "Number of calls per A/B test variant",
    labelnames=["experiment", "variant"]
)

# HISTOGRAMS (latency distributions)
REVIEW_LATENCY = Histogram(
    "codeguard_review_latency_seconds",
    "PR review latency in seconds",
    labelnames=["agent", "model"],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120]
)

LLM_LATENCY = Histogram(
    "codeguard_llm_latency_seconds",
    "LLM API call latency in seconds",
    labelnames=["model", "provider"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30]
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "codeguard_rag_retrieval_latency_seconds",
    "Weaviate retrieval latency in seconds",
    labelnames=["search_type"],  # hybrid, dense, bm25
    buckets=[0.1, 0.5, 1, 2, 5]
)

# SUMMARIES (percentiles)
QUALITY_SCORE = Summary(
    "codeguard_review_quality_score",
    "Review quality score distribution",
    labelnames=["agent", "model"]
)

RAGAS_SCORE = Summary(
    "codeguard_ragas_score",
    "RAGAS evaluation score distribution",
    labelnames=["metric"]  # faithfulness, relevancy, precision, recall
)


class CodeGuardMetrics:
    """
    Helper class for recording CodeGuard metrics.
    Used by agents, webhook handler, eval pipeline.
    """

    @staticmethod
    def record_review_start():
        """Call when a PR review begins."""
        ACTIVE_REVIEWS.inc()

    @staticmethod
    def record_review_complete(
        agent    : str,
        model    : str,
        provider : str,
        severity : str,
        latency  : float,
        issues   : int,
        quality  : float,
        tokens_in : int = 0,
        tokens_out: int = 0,
        cost_usd  : float = 0.0
    ):
        """Call when a PR review completes."""
        ACTIVE_REVIEWS.dec()

        # Count this review
        REVIEWS_TOTAL.labels(
            agent=agent, model=model,
            provider=provider, severity=severity
        ).inc()

        # Record latency
        REVIEW_LATENCY.labels(agent=agent, model=model).observe(latency)

        # Record quality
        QUALITY_SCORE.labels(agent=agent, model=model).observe(quality)

        # Record token usage
        if tokens_in:
            LLM_TOKENS_USED.labels(
                model=model, provider=provider, token_type="input"
            ).inc(tokens_in)
        if tokens_out:
            LLM_TOKENS_USED.labels(
                model=model, provider=provider, token_type="output"
            ).inc(tokens_out)

        # Record cost
        if cost_usd:
            LLM_COST_USD.labels(model=model, provider=provider).inc(cost_usd)

    @staticmethod
    def record_issues(agent: str, severity: str, issue_type: str, count: int = 1):
        """Record issues found by an agent."""
        ISSUES_FOUND.labels(
            agent=agent, severity=severity, issue_type=issue_type
        ).inc(count)

    @staticmethod
    def record_error(agent: str, error_type: str):
        """Record an error in the pipeline."""
        ERRORS_TOTAL.labels(agent=agent, error_type=error_type).inc()

    @staticmethod
    def record_llm_call(model: str, provider: str, latency: float):
        """Record individual LLM API call."""
        LLM_LATENCY.labels(model=model, provider=provider).observe(latency)

    @staticmethod
    def record_rag_retrieval(search_type: str, latency: float):
        """Record Weaviate retrieval latency."""
        RAG_RETRIEVAL_LATENCY.labels(search_type=search_type).observe(latency)

    @staticmethod
    def record_ragas_scores(scores: dict):
        """Record RAGAS evaluation scores."""
        for metric, score in scores.items():
            if metric != "overall_rag_score":
                RAGAS_SCORE.labels(metric=metric).observe(score)

    @staticmethod
    def update_canary(deployment: str, variant: str, traffic_pct: float):
        """Update canary traffic percentage gauge."""
        CANARY_TRAFFIC_PCT.labels(
            deployment_name=deployment, variant=variant
        ).set(traffic_pct)

    @staticmethod
    def update_ab_calls(experiment: str, variant: str, calls: int):
        """Update A/B test call counts."""
        AB_TEST_CALLS.labels(
            experiment=experiment, variant=variant
        ).set(calls)


def start_metrics_server(port: int = 8001):
    """
    Start Prometheus metrics HTTP server.
    Prometheus scrapes: http://localhost:8001/metrics

    Format:
      # HELP codeguard_reviews_total Total number of PR reviews processed
      # TYPE codeguard_reviews_total counter
      codeguard_reviews_total{agent="security",model="gpt-4o",...} 42.0
    """
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
    print(f"  Scrape endpoint: http://localhost:{port}/metrics")
    print(f"  Prometheus UI:   http://localhost:9090")
    print(f"  Grafana UI:      http://localhost:3000 (admin/codeguard)")


# ── Global singleton ──────────────────────────────────────────────
metrics = CodeGuardMetrics()


if __name__ == "__main__":
    print("Starting Prometheus metrics demo...")

    # Start metrics HTTP server
    start_metrics_server(port=8001)

    # Simulate 10 reviews worth of metrics
    print("\nSimulating 10 PR reviews...")
    import random

    for i in range(10):
        metrics.record_review_start()
        time.sleep(0.1)

        model    = random.choice(["gpt-4o", "deepseek-v3"])
        provider = "openai" if model == "gpt-4o" else "fireworks"
        agent    = random.choice(["security", "style", "performance", "architecture"])
        issues   = random.randint(2, 12)
        latency  = random.uniform(4.0, 15.0)
        severity = "CRITICAL" if issues > 8 else "HIGH" if issues > 4 else "LOW"

        metrics.record_review_complete(
            agent=agent, model=model, provider=provider,
            severity=severity, latency=latency,
            issues=issues, quality=random.uniform(7.5, 9.5),
            tokens_in=random.randint(1500, 3000),
            tokens_out=random.randint(500, 1500),
            cost_usd=0.003 if model == "gpt-4o" else 0.0002
        )
        metrics.record_issues(agent, severity, "sql_injection", random.randint(1, 3))
        metrics.record_llm_call(model, provider, latency)
        metrics.record_rag_retrieval("hybrid", random.uniform(0.2, 1.5))

    metrics.record_ragas_scores({
        "faithfulness"      : 0.847,
        "answer_relevancy"  : 0.912,
        "context_precision" : 0.783,
        "context_recall"    : 0.821
    })

    metrics.update_canary("security-agent-v2", "stable", 75.0)
    metrics.update_canary("security-agent-v2", "canary", 25.0)
    metrics.update_ab_calls("security_model_comparison", "control",    12)
    metrics.update_ab_calls("security_model_comparison", "challenger",  3)

    print("\nMetrics available at http://localhost:8001/metrics")
    print("Keeping server alive for 60 seconds...")
    print("Open Prometheus: http://localhost:9090")
    print("Query: codeguard_reviews_total")

    time.sleep(60)
