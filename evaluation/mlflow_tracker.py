"""
MLflow — Experiment Tracking & Model Registry

WHY MLFLOW:
  LangSmith tracks individual LLM calls (traces).
  MLflow tracks EXPERIMENTS — comparing runs across versions.

  MLflow answers:
  - Which prompt version finds the most security issues?
  - Does adding RAG context improve review quality?
  - Which model (gpt-4o vs deepseek) has better precision/recall?
  - What was the exact config of the best-performing run?

  MLflow stores:
  - Parameters: model_name, temperature, top_k, alpha
  - Metrics:    issues_found, latency, quality_score, cost
  - Artifacts:  prompt templates, review outputs, RAGAS scores
  - Tags:       severity, repo, agent_type

  Result: complete experiment history, reproducible runs,
  model version registry for production deployments.
"""
import os
import json
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT",   "codeguard-reviews")


class MLflowTracker:
    """
    Tracks CodeGuard review experiments in MLflow.

    Each PR review = one MLflow run.
    Parameters + metrics + artifacts all logged.
    Enables comparison across model versions.
    """

    def __init__(self):
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        self.mlflow    = mlflow
        self.active_run = None
        print(f"MLflow tracker initialized")
        print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"  Experiment:   {MLFLOW_EXPERIMENT}")
        print(f"  UI command:   mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    def start_review_run(
        self,
        pr_number  : int,
        repo       : str,
        model      : str,
        provider   : str,
        agent_type : str
    ):
        """Start tracking a new review run."""
        run_name = f"{agent_type}-PR#{pr_number}-{model}"
        self.active_run = self.mlflow.start_run(run_name=run_name)

        # Log parameters (inputs that define the run)
        self.mlflow.log_params({
            "pr_number"  : pr_number,
            "repo"       : repo,
            "model"      : model,
            "provider"   : provider,
            "agent_type" : agent_type,
            "timestamp"  : time.strftime("%Y-%m-%d %H:%M:%S")
        })

        return self.active_run

    def log_rag_params(self, alpha: float, top_k: int, rerank: bool):
        """Log RAG retrieval parameters."""
        self.mlflow.log_params({
            "rag_alpha"  : alpha,
            "rag_top_k"  : top_k,
            "rag_rerank" : rerank,
        })

    def log_review_metrics(
        self,
        issues_found  : int,
        critical_count: int,
        high_count    : int,
        latency_s     : float,
        quality_score : float,
        cost_usd      : float = 0.0,
        tokens_used   : int   = 0
    ):
        """Log review output metrics."""
        self.mlflow.log_metrics({
            "issues_found"  : issues_found,
            "critical_count": critical_count,
            "high_count"    : high_count,
            "latency_s"     : latency_s,
            "quality_score" : quality_score,
            "cost_usd"      : cost_usd,
            "tokens_used"   : tokens_used,
        })

    def log_ragas_scores(self, scores: Dict[str, float]):
        """Log RAGAS evaluation scores."""
        self.mlflow.log_metrics({
            f"ragas_{k}": v
            for k, v in scores.items()
        })

    def log_trulens_scores(self, scores: Dict[str, float]):
        """Log TruLens RAG Triad scores."""
        self.mlflow.log_metrics({
            f"trulens_{k}": v
            for k, v in scores.items()
        })

    def log_review_artifact(self, review_output: Dict, filename: str = "review.json"):
        """Save full review output as artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(review_output, f, indent=2)
            tmp_path = f.name
        self.mlflow.log_artifact(tmp_path, filename)
        os.unlink(tmp_path)

    def end_run(self, status: str = "FINISHED"):
        """End the current tracking run."""
        if self.active_run:
            self.mlflow.end_run(status=status)
            self.active_run = None

    def compare_models(self, experiment_name: str = None) -> Dict:
        """
        Compare all runs in the experiment.
        Returns model performance comparison.
        """
        exp_name = experiment_name or MLFLOW_EXPERIMENT
        experiment = self.mlflow.get_experiment_by_name(exp_name)
        if not experiment:
            return {"error": "No experiment found"}

        runs = self.mlflow.search_runs(
            experiment_ids = [experiment.experiment_id],
            order_by       = ["metrics.quality_score DESC"]
        )

        if runs.empty:
            return {"error": "No runs found"}

        # Group by model
        model_stats = {}
        for _, run in runs.iterrows():
            model = run.get("params.model", "unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "runs"          : 0,
                    "total_issues"  : 0,
                    "total_latency" : 0,
                    "total_quality" : 0,
                    "total_cost"    : 0,
                }
            stats = model_stats[model]
            stats["runs"]          += 1
            stats["total_issues"]  += run.get("metrics.issues_found", 0)
            stats["total_latency"] += run.get("metrics.latency_s", 0)
            stats["total_quality"] += run.get("metrics.quality_score", 0)
            stats["total_cost"]    += run.get("metrics.cost_usd", 0)

        comparison = {}
        for model, stats in model_stats.items():
            n = stats["runs"]
            comparison[model] = {
                "runs"             : n,
                "avg_issues_found" : round(stats["total_issues"] / n, 1),
                "avg_latency_s"    : round(stats["total_latency"] / n, 2),
                "avg_quality_score": round(stats["total_quality"] / n, 3),
                "total_cost_usd"   : round(stats["total_cost"], 4),
            }

        print("\nMLflow Model Comparison:")
        print(f"{'Model':20s} {'Runs':6s} {'Issues':8s} {'Latency':10s} {'Quality':10s} {'Cost':10s}")
        print("-" * 70)
        for model, stats in comparison.items():
            print(
                f"{model:20s} "
                f"{stats['runs']:6d} "
                f"{stats['avg_issues_found']:8.1f} "
                f"{stats['avg_latency_s']:10.2f}s "
                f"{stats['avg_quality_score']:10.3f} "
                f"${stats['total_cost_usd']:9.4f}"
            )

        return comparison


# ── Global singleton ──────────────────────────────────────────────
mlflow_tracker = MLflowTracker()


if __name__ == "__main__":
    tracker = MLflowTracker()

    # Simulate a full review tracking cycle
    print("\nSimulating 3 review runs...")

    for i, (model, provider, issues, latency, quality, cost) in enumerate([
        ("gpt-4o",      "openai",     9, 8.2,  9.1, 0.003),
        ("deepseek-v3", "fireworks",  8, 12.1, 8.7, 0.0002),
        ("gpt-4o",      "openai",     7, 7.8,  8.9, 0.0028),
    ]):
        tracker.start_review_run(
            pr_number  = 100 + i,
            repo       = "Tejesh0209/SentinelAI",
            model      = model,
            provider   = provider,
            agent_type = "security"
        )
        tracker.log_rag_params(alpha=0.5, top_k=5, rerank=True)
        tracker.log_review_metrics(
            issues_found   = issues,
            critical_count = issues - 2,
            high_count     = 2,
            latency_s      = latency,
            quality_score  = quality,
            cost_usd       = cost,
            tokens_used    = 4200
        )
        tracker.log_ragas_scores({
            "faithfulness"      : 0.85,
            "answer_relevancy"  : 0.91,
            "context_precision" : 0.78,
            "context_recall"    : 0.82
        })
        tracker.end_run()
        print(f"  Run {i+1}: {model} | issues={issues} | latency={latency}s | quality={quality}")

    # Compare models
    tracker.compare_models()

    print(f"\nView in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  Open: http://localhost:5000")
