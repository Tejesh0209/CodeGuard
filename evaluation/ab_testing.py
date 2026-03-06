import os
import time
import random
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelVariant:
    """
    A single variant in an A/B test.
    Example:
      variant_a = ModelVariant("control",    model="gpt-4o",      traffic_pct=80)
      variant_b = ModelVariant("challenger", model="deepseek-v3", traffic_pct=20)
    """
    name        : str
    model       : str
    provider    : str
    traffic_pct : float      # 0-100, must sum to 100 across variants
    # Metrics collected during experiment
    total_calls   : int   = 0
    total_latency : float = 0.0
    total_issues  : int   = 0
    total_errors  : int   = 0
    quality_scores: list  = field(default_factory=list)

    def avg_latency(self) -> float:
        return round(self.total_latency / self.total_calls, 2) if self.total_calls else 0.0

    def avg_quality(self) -> float:
        return round(sum(self.quality_scores) / len(self.quality_scores), 2) if self.quality_scores else 0.0

    def error_rate(self) -> float:
        return round(self.total_errors / self.total_calls * 100, 1) if self.total_calls else 0.0

    def record(self, latency: float, issues: int, quality: float, error: bool = False):
        self.total_calls   += 1
        self.total_latency += latency
        self.total_issues  += issues
        self.quality_scores.append(quality)
        if error:
            self.total_errors += 1


@dataclass
class ABExperiment:
    """
    Tracks a single A/B experiment comparing two model variants.

    Example use cases:
      - GPT-4o vs Fireworks for security reviews: which finds more issues?
      - Larger context window vs smaller: does quality improve?
      - Two different system prompts: which produces better structured output?

    Assignment is DETERMINISTIC based on PR number hash.
    Same PR always gets same variant (reproducibility).
    """
    name       : str
    description: str
    variants   : list[ModelVariant]
    start_time : float = field(default_factory=time.time)
    active     : bool  = True

    def assign_variant(self, pr_number: int, repo: str) -> ModelVariant:
        """
        Deterministic variant assignment.
        Same PR number + repo always gets same variant.
        This ensures consistent review experience per PR.
        """
        # Hash PR identifier to get consistent bucket
        key    = f"{repo}:{pr_number}"
        bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100

        # Assign based on cumulative traffic percentages
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.traffic_pct
            if bucket < cumulative:
                return variant

        return self.variants[-1]  # fallback to last variant

    def get_results(self) -> dict:
        """Statistical comparison between variants."""
        results = {
            "experiment"  : self.name,
            "description" : self.description,
            "active"      : self.active,
            "duration_hrs": round((time.time() - self.start_time) / 3600, 2),
            "variants"    : {}
        }

        for v in self.variants:
            results["variants"][v.name] = {
                "model"        : v.model,
                "provider"     : v.provider,
                "traffic_pct"  : v.traffic_pct,
                "total_calls"  : v.total_calls,
                "avg_latency_s": v.avg_latency(),
                "avg_quality"  : v.avg_quality(),
                "error_rate_pct": v.error_rate(),
                "total_issues_found": v.total_issues,
            }

        # Winner determination (simple: higher quality wins)
        if all(v.total_calls > 10 for v in self.variants):
            scores = {v.name: v.avg_quality() for v in self.variants}
            winner = max(scores, key=scores.get)
            results["winner"]     = winner
            results["confidence"] = "high" if self.variants[0].total_calls > 50 else "low"
        else:
            results["winner"]     = "insufficient_data"
            results["confidence"] = "none"

        return results


class ABTestingEngine:
    """
    Manages multiple A/B experiments simultaneously.

    WHY A/B TESTING FOR LLMs:
      - Different models have different strengths
      - Can't know in advance which is better for YOUR codebase
      - A/B testing gives empirical evidence, not assumptions
      - Example: Fireworks may find 90% of issues at 16x lower cost
        but A/B test might show it misses 40% of CRITICAL issues
        -> data drives the routing decision

    HOW IT WORKS IN CODEGUARD:
      1. Define experiment: 80% GPT-4o vs 20% Fireworks for security
      2. Every PR review: deterministically assigned to a variant
      3. Metrics collected: latency, issues found, quality score
      4. After N reviews: compare variants, pick winner
      5. Gradually shift traffic to winner (canary deploy)
    """

    def __init__(self):
        self.experiments: dict[str, ABExperiment] = {}
        self._setup_default_experiments()

    def _setup_default_experiments(self):
        """
        Default experiments for CodeGuard.
        These run automatically on every review.
        """

        # Experiment 1: GPT-4o vs Fireworks for security review quality
        self.experiments["security_model_comparison"] = ABExperiment(
            name        = "security_model_comparison",
            description = "Compare GPT-4o vs Fireworks DeepSeek for security review accuracy",
            variants    = [
                ModelVariant(
                    name        = "control",
                    model       = "gpt-4o",
                    provider    = "openai",
                    traffic_pct = 80    # 80% of security reviews use GPT-4o
                ),
                ModelVariant(
                    name        = "challenger",
                    model       = "deepseek-v3",
                    provider    = "fireworks",
                    traffic_pct = 20    # 20% use Fireworks to collect data
                ),
            ]
        )

        # Experiment 2: Fast vs thorough review for style
        self.experiments["style_speed_vs_quality"] = ABExperiment(
            name        = "style_speed_vs_quality",
            description = "Compare fast small model vs larger model for style review",
            variants    = [
                ModelVariant(
                    name        = "fast",
                    model       = "gpt-oss-20b",
                    provider    = "fireworks",
                    traffic_pct = 50
                ),
                ModelVariant(
                    name        = "thorough",
                    model       = "deepseek-v3",
                    provider    = "fireworks",
                    traffic_pct = 50
                ),
            ]
        )

    def get_variant(self, experiment_name: str, pr_number: int, repo: str) -> Optional[ModelVariant]:
        """Get the assigned variant for a PR in an experiment."""
        exp = self.experiments.get(experiment_name)
        if not exp or not exp.active:
            return None
        return exp.assign_variant(pr_number, repo)

    def record_result(
        self,
        experiment_name: str,
        variant_name   : str,
        latency        : float,
        issues_found   : int,
        quality_score  : float,
        error          : bool = False
    ):
        exp = self.experiments.get(experiment_name)
        if not exp:
            return
        for variant in exp.variants:
            if variant.name == variant_name:
                variant.record(latency, issues_found, quality_score, error)
                break

    def get_all_results(self) -> dict:
        return {
            name: exp.get_results()
            for name, exp in self.experiments.items()
        }

    def conclude_experiment(self, experiment_name: str):
        """Mark experiment as concluded. Winner becomes new default."""
        exp = self.experiments.get(experiment_name)
        if exp:
            exp.active = False
            results    = exp.get_results()
            print(f"Experiment '{experiment_name}' concluded.")
            print(f"  Winner: {results.get('winner', 'unknown')}")
            return results
        return {}


# ── Global singleton ──────────────────────────────────────────────
ab_engine = ABTestingEngine()
