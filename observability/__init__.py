"""
CodeGuard Observability Stack

Components:
  OpenTelemetry  -> distributed tracing (standard protocol)
  OpenInference  -> LLM semantic conventions on top of OTel
  Phoenix        -> self-hosted LLM observability dashboard
  Prometheus     -> metrics collection + alerting
  Grafana        -> metrics dashboards
  Jaeger         -> distributed trace visualization
"""
from observability.prometheus_metrics import metrics, start_metrics_server
from observability.phoenix_tracer     import phoenix_tracer

__all__ = ["metrics", "start_metrics_server", "phoenix_tracer"]
