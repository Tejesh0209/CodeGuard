"""
LangSmith — Tracing & Observability for LLM calls

WHY LANGSMITH:
  Every time an agent calls GPT-4o or Fireworks, we want to know:
  - Exact prompt sent (input tokens)
  - Exact response received (output tokens)
  - Latency breakdown (time per step)
  - Cost per review
  - Which runs failed and why
  - How prompts change over time

  Without LangSmith: you're flying blind.
  With LangSmith: full X-ray vision into every LLM call.
"""
import os
from langsmith import Client
from langsmith.wrappers import wrap_openai
from langchain_core.tracers import LangChainTracer
from dotenv import load_dotenv

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "codeguard-eval")


class CodeGuardTracer:
    """
    Wraps CodeGuard review pipeline with LangSmith tracing.
    Every LLM call automatically logged to smith.langchain.com
    """

    def __init__(self):
        self.client  = Client(api_key=LANGSMITH_API_KEY)
        self.tracer  = LangChainTracer(project_name=LANGSMITH_PROJECT)
        self.project = LANGSMITH_PROJECT
        print(f"LangSmith tracer initialized")
        print(f"  Project: {LANGSMITH_PROJECT}")
        print(f"  Dashboard: https://smith.langchain.com/projects/{LANGSMITH_PROJECT}")

    def get_callbacks(self) -> list:
        """Returns callbacks to pass to LangChain chain.invoke()"""
        return [self.tracer]

    def log_review_run(
        self,
        pr_number  : int,
        repo       : str,
        inputs     : dict,
        outputs    : dict,
        severity   : str,
        model_used : str
    ):
        """
        Manually log a complete review run to LangSmith.
        Captures: inputs, outputs, severity, model, timing.
        """
        from langsmith import traceable

        run_name = f"PR#{pr_number} - {severity} - {model_used}"

        try:
            self.client.create_run(
                name       = run_name,
                run_type   = "chain",
                project_name = self.project,
                inputs     = {
                    "pr_number" : pr_number,
                    "repo"      : repo,
                    "model"     : model_used,
                    **inputs
                },
                outputs    = outputs,
                tags       = [severity, model_used, repo]
            )
            print(f"  [LangSmith] Run logged: {run_name}")
        except Exception as e:
            print(f"  [LangSmith] Logging failed (non-critical): {e}")

    def get_project_stats(self) -> dict:
        """Fetch aggregate stats from LangSmith for this project."""
        try:
            runs  = list(self.client.list_runs(project_name=self.project, limit=100))
            if not runs:
                return {"status": "no runs yet", "project": self.project}

            total    = len(runs)
            errors   = sum(1 for r in runs if r.error)
            latencies = [
                r.end_time.timestamp() - r.start_time.timestamp()
                for r in runs
                if r.end_time and r.start_time
            ]
            avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0

            return {
                "project"      : self.project,
                "total_runs"   : total,
                "error_count"  : errors,
                "error_rate"   : f"{round(errors/total*100, 1)}%",
                "avg_latency_s": avg_lat,
                "dashboard_url": f"https://smith.langchain.com"
            }
        except Exception as e:
            return {"status": f"fetch failed: {e}"}


# ── Global singleton ──────────────────────────────────────────────
tracer = CodeGuardTracer()


if __name__ == "__main__":
    print(f"\nLangSmith project stats:")
    import json
    print(json.dumps(tracer.get_project_stats(), indent=2))
