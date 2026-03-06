"""
Start the full observability stack and run a demo.
"""
import os, sys, time
sys.path.insert(0, "/Users/tejeshboppana/Desktop/Codeguard")
os.chdir("/Users/tejeshboppana/Desktop/Codeguard")

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("CODEGUARD OBSERVABILITY STACK")
print("=" * 60)

# ── 1. Prometheus metrics server ──────────────────────────────────
print("\n[1/4] Starting Prometheus metrics server...")
from observability.prometheus_metrics import metrics, start_metrics_server
start_metrics_server(port=8001)

# ── 2. Phoenix tracer ─────────────────────────────────────────────
print("\n[2/4] Initializing Phoenix tracer...")
from observability.phoenix_tracer import phoenix_tracer

# ── 3. OpenTelemetry ──────────────────────────────────────────────
print("\n[3/4] Setting up OpenTelemetry...")
from observability.otel_setup import setup_telemetry
tracer = setup_telemetry()

# ── 4. Simulate metrics ───────────────────────────────────────────
print("\n[4/4] Generating demo metrics...")
import random

for i in range(5):
    model    = random.choice(["gpt-4o", "deepseek-v3"])
    provider = "openai" if model == "gpt-4o" else "fireworks"
    agent    = random.choice(["security", "style", "performance"])
    issues   = random.randint(2, 12)
    latency  = random.uniform(4.0, 15.0)
    severity = "CRITICAL" if issues > 8 else "HIGH"

    metrics.record_review_start()
    time.sleep(0.05)
    metrics.record_review_complete(
        agent=agent, model=model, provider=provider,
        severity=severity, latency=latency,
        issues=issues, quality=random.uniform(7.5, 9.5),
        tokens_in=2000, tokens_out=1000,
        cost_usd=0.003 if model == "gpt-4o" else 0.0002
    )
    metrics.record_llm_call(model, provider, latency)
    metrics.record_rag_retrieval("hybrid", random.uniform(0.2, 1.0))

    # OTel span
    with tracer.start_as_current_span(f"pr_review_{i}") as span:
        span.set_attribute("pr.number",    100 + i)
        span.set_attribute("agent.type",   agent)
        span.set_attribute("model.name",   model)
        span.set_attribute("issues.found", issues)

    print(f"  Review {i+1}: {agent}/{model} | {issues} issues | {latency:.1f}s")

metrics.update_canary("security-agent-v2", "stable", 75.0)
metrics.update_canary("security-agent-v2", "canary", 25.0)

print("\n" + "=" * 60)
print("OBSERVABILITY STACK RUNNING")
print("=" * 60)
print(f"  Prometheus metrics: http://localhost:8001/metrics")
print(f"  Prometheus UI:      http://localhost:9090")
print(f"  Grafana:            http://localhost:3000  (admin/codeguard)")
print(f"  Jaeger:             http://localhost:16686")
print(f"  Phoenix:            http://localhost:6006")
print(f"\nUseful Prometheus queries:")
print(f"  codeguard_reviews_total")
print(f"  rate(codeguard_reviews_total[1m])")
print(f"  histogram_quantile(0.99, codeguard_review_latency_seconds_bucket)")
print(f"  sum(codeguard_llm_cost_usd_total)")
print("=" * 60)
print("Keeping alive for 120s. Press Ctrl+C to stop.")

try:
    time.sleep(120)
except KeyboardInterrupt:
    print("\nStopped.")
