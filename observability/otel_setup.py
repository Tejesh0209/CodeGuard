"""
OpenTelemetry + OpenInference — Unified Observability Layer

WHY OPENTELEMETRY:
  OpenTelemetry (OTel) is the industry standard for distributed tracing.
  Every major observability tool (Jaeger, Grafana, Datadog, New Relic)
  speaks OTel protocol (OTLP).

  Write instrumentation ONCE -> send to ANY backend:
    OTel SDK -> OTLP exporter -> Jaeger  (traces)
                              -> Phoenix (LLM traces)
                              -> Grafana (via Tempo)
                              -> Datadog (production)

WHY OPENINFERENCE:
  OTel has no standard for LLM-specific data:
    What is "input" for an LLM? The prompt.
    What is "output"? The response.
    What are "attributes"? Token counts, model name, temperature.

  OpenInference defines semantic conventions for LLM observability:
    openinference.span.kind = "LLM" or "RETRIEVER" or "CHAIN"
    llm.input_messages       = prompt messages
    llm.output_messages      = response messages
    llm.token_count.prompt   = input tokens
    llm.token_count.completion = output tokens
    retrieval.documents      = chunks retrieved from Weaviate

  Together: OTel handles transport, OpenInference handles meaning.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── OTel core imports ─────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# ── OpenInference instrumentation ─────────────────────────────────
from openinference.instrumentation.openai    import OpenAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

# ── Phoenix OTel exporter ─────────────────────────────────────────
from phoenix.otel import register as phoenix_register

JAEGER_ENDPOINT  = os.getenv("JAEGER_ENDPOINT",  "http://localhost:4318/v1/traces")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
SERVICE_NAME     = "codeguard"


def setup_telemetry() -> trace.Tracer:
    """
    Initialize OpenTelemetry with dual export:
    1. Jaeger  -> distributed tracing UI (localhost:16686)
    2. Phoenix -> LLM-specific observability (localhost:6006)

    After calling this:
    - Every OpenAI call automatically traced (OpenInference)
    - Every LangChain chain automatically traced (OpenInference)
    - Spans visible in Jaeger and Phoenix dashboards
    """

    # Resource: identifies this service in traces
    resource = Resource.create({
        "service.name"    : SERVICE_NAME,
        "service.version" : "1.0.0",
        "deployment.env"  : os.getenv("ENV", "development"),
    })

    # TracerProvider: central registry for all tracers
    provider = TracerProvider(resource=resource)

    # Exporter 1: Jaeger (OTLP HTTP)
    try:
        jaeger_exporter = OTLPSpanExporter(endpoint=JAEGER_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        print(f"  [OTel] Jaeger exporter: {JAEGER_ENDPOINT}")
    except Exception as e:
        print(f"  [OTel] Jaeger not available: {e}")

    # Exporter 2: Phoenix (OTLP HTTP)
    try:
        phoenix_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(phoenix_exporter))
        print(f"  [OTel] Phoenix exporter: {PHOENIX_ENDPOINT}")
    except Exception as e:
        print(f"  [OTel] Phoenix not available: {e}")

    # Register as global provider
    trace.set_tracer_provider(provider)

    # Auto-instrument OpenAI + LangChain
    # From this point: every OpenAI/LangChain call creates a span
    OpenAIInstrumentor().instrument()
    LangChainInstrumentor().instrument()

    print(f"  [OTel] OpenAI + LangChain auto-instrumented")
    print(f"  [OTel] Jaeger UI:  http://localhost:16686")
    print(f"  [OTel] Phoenix UI: http://localhost:6006")

    return trace.get_tracer(SERVICE_NAME)


def setup_phoenix_only() -> trace.Tracer:
    """
    Simpler setup: Phoenix only (no Jaeger required).
    Phoenix is the recommended setup for LLM tracing.
    """
    try:
        phoenix_register(
            project_name    = "codeguard",
            endpoint        = PHOENIX_ENDPOINT,
            set_global_tracer_provider = True
        )
        OpenAIInstrumentor().instrument()
        LangChainInstrumentor().instrument()
        print(f"  [Phoenix] Registered. UI: http://localhost:6006")
    except Exception as e:
        print(f"  [Phoenix] Setup warning: {e}")

    return trace.get_tracer(SERVICE_NAME)


# ── Global tracer ─────────────────────────────────────────────────
tracer = trace.get_tracer(SERVICE_NAME)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(SERVICE_NAME)


if __name__ == "__main__":
    print("Setting up OpenTelemetry...")
    t = setup_telemetry()

    # Test: create a manual span
    with t.start_as_current_span("test_codeguard_span") as span:
        span.set_attribute("pr.number",   42)
        span.set_attribute("pr.repo",     "Tejesh0209/SentinelAI")
        span.set_attribute("agent.type",  "security")
        span.set_attribute("model.name",  "gpt-4o")
        span.set_attribute("issues.found", 9)
        print("  Test span created")
        print("  Check Jaeger: http://localhost:16686")
        print("  Check Phoenix: http://localhost:6006")
