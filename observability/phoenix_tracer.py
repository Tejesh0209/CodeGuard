"""
Arize Phoenix — Self-Hosted LLM Observability

WHY PHOENIX:
  LangSmith = SaaS (your traces go to Langchain's cloud)
  Phoenix   = Self-hosted (your traces stay on YOUR machine)

  For enterprises with data privacy requirements:
  - Code diffs contain proprietary code
  - Can't send to third-party cloud
  - Phoenix runs locally in Docker
  - Same features as LangSmith: traces, spans, evals

  Phoenix also ships with:
  - Built-in RAGAS-style evaluations
  - Embedding visualization (UMAP clusters)
  - Dataset management for fine-tuning
  - Prompt playground

  Dashboard: http://localhost:6006
"""
import os
from dotenv import load_dotenv
load_dotenv()


class PhoenixTracer:
    """
    Wraps Phoenix tracing for CodeGuard.
    Auto-instruments OpenAI and LangChain calls.
    """

    def __init__(self):
        self.endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
        self._setup()

    def _setup(self):
        try:
            import phoenix as px
            from phoenix.otel import register
            from openinference.instrumentation.openai    import OpenAIInstrumentor
            from openinference.instrumentation.langchain import LangChainInstrumentor

            # Register Phoenix as OTel endpoint
            register(
                project_name               = "codeguard",
                endpoint                   = f"{self.endpoint}/v1/traces",
                set_global_tracer_provider = True
            )

            # Auto-instrument all OpenAI + LangChain calls
            OpenAIInstrumentor().instrument()
            LangChainInstrumentor().instrument()

            self.active = True
            print(f"Phoenix tracer active")
            print(f"  Dashboard: {self.endpoint}")
            print(f"  Auto-instrumented: OpenAI + LangChain")

        except Exception as e:
            self.active = False
            print(f"Phoenix not available (is Docker container running?): {e}")
            print(f"  Start with: docker run -p 6006:6006 arizephoenix/phoenix:latest")

    def trace_review(self, pr_number: int, repo: str, agent: str):
        """
        Context manager for tracing a complete review.
        Creates a parent span with PR metadata.

        Usage:
            with phoenix_tracer.trace_review(42, "SentinelAI", "security"):
                result = await security_agent.review(diff)
        """
        from opentelemetry import trace
        tracer = trace.get_tracer("codeguard")

        return tracer.start_as_current_span(
            f"{agent}_review",
            attributes={
                "pr.number"  : pr_number,
                "pr.repo"    : repo,
                "agent.type" : agent,
                "codeguard.version": "1.0.0"
            }
        )

    def evaluate_traces(self, sample_size: int = 10):
        """
        Run Phoenix built-in evals on recent traces.
        Returns hallucination + relevance scores.
        """
        try:
            import phoenix as px
            from phoenix.evals import (
                HallucinationEvaluator,
                RelevanceEvaluator,
                run_evals
            )
            from phoenix.evals.models import OpenAIModel

            client = px.Client(endpoint=self.endpoint)
            spans  = client.get_spans_dataframe(project_name="codeguard")

            if spans is None or spans.empty:
                print("No spans found in Phoenix yet.")
                print("Run some PR reviews first, then call evaluate_traces()")
                return {}

            eval_model = OpenAIModel(
                model   = "gpt-4o-mini",
                api_key = os.getenv("OPENAI_API_KEY")
            )

            results = run_evals(
                dataframe  = spans.head(sample_size),
                evaluators = [
                    HallucinationEvaluator(eval_model),
                    RelevanceEvaluator(eval_model)
                ],
                provide_explanation = True
            )
            return results

        except Exception as e:
            print(f"Phoenix eval failed: {e}")
            return {}


# ── Global singleton ──────────────────────────────────────────────
phoenix_tracer = PhoenixTracer()


if __name__ == "__main__":
    print("Phoenix tracer initialized")
    print(f"Active: {phoenix_tracer.active}")
    print(f"Dashboard: http://localhost:6006")
