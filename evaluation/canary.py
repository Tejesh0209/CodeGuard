import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class CanaryStage(Enum):
    """
    Standard canary deployment stages.
    Each stage increases traffic to the new model version.
    """
    INACTIVE  = "inactive"   # 0%  - new version not deployed
    CANARY    = "canary"     # 5%  - tiny slice, watch for errors
    EARLY     = "early"      # 25% - broader test, still safe
    HALF      = "half"       # 50% - equal split
    MAJORITY  = "majority"   # 75% - mostly new version
    FULL      = "full"       # 100%- complete rollout


STAGE_TRAFFIC = {
    CanaryStage.INACTIVE : 0,
    CanaryStage.CANARY   : 5,
    CanaryStage.EARLY    : 25,
    CanaryStage.HALF     : 50,
    CanaryStage.MAJORITY : 75,
    CanaryStage.FULL     : 100,
}

STAGE_ORDER = [
    CanaryStage.INACTIVE,
    CanaryStage.CANARY,
    CanaryStage.EARLY,
    CanaryStage.HALF,
    CanaryStage.MAJORITY,
    CanaryStage.FULL,
]


@dataclass
class CanaryMetrics:
    """Health metrics collected at each canary stage."""
    stage         : CanaryStage
    start_time    : float = field(default_factory=time.time)
    total_calls   : int   = 0
    total_errors  : int   = 0
    total_latency : float = 0.0
    issues_found  : int   = 0

    def error_rate(self) -> float:
        return (self.total_errors / self.total_calls * 100) if self.total_calls else 0.0

    def avg_latency(self) -> float:
        return (self.total_latency / self.total_calls) if self.total_calls else 0.0

    def is_healthy(self, max_error_rate: float = 5.0, max_latency: float = 30.0) -> bool:
        """
        Health check for advancing to next stage.
        max_error_rate: 5% errors allowed before rollback
        max_latency:    30s max average latency
        """
        if self.total_calls < 3:
            return True   # insufficient data, assume healthy
        return (
            self.error_rate() <= max_error_rate and
            self.avg_latency() <= max_latency
        )


@dataclass
class CanaryDeployment:
    """
    Manages a single canary deployment.

    WHY CANARY DEPLOYMENTS FOR LLMs:
      Switching from GPT-4o to a new model overnight is risky.
      The new model might:
        - Miss critical security vulnerabilities
        - Produce malformed JSON (breaking the parser)
        - Be 10x slower under real load
        - Behave differently on YOUR codebase vs benchmarks

      Canary: send 5% of real traffic to new model first.
      Watch error rate and latency for 1 hour.
      If healthy -> advance to 25%.
      If unhealthy -> instant rollback to stable version.

    STAGES:
      0% -> 5% (canary)   : 1 hour minimum
      5% -> 25% (early)   : 2 hours minimum
      25% -> 50% (half)   : 4 hours minimum
      50% -> 75% (majority): 4 hours minimum
      75% -> 100% (full)  : 8 hours minimum
      Total: ~19 hours for full rollout with safety checks
    """
    name           : str
    stable_model   : str    # e.g. "gpt-4o"
    canary_model   : str    # e.g. "fireworks/deepseek-v3"
    stable_provider: str
    canary_provider: str
    current_stage  : CanaryStage = CanaryStage.INACTIVE
    auto_advance   : bool        = True   # advance stages automatically if healthy
    stage_metrics  : dict        = field(default_factory=dict)
    rollback_reason: str         = ""
    created_at     : float       = field(default_factory=time.time)

    # Stage minimum durations (seconds) before advancing
    STAGE_MIN_DURATION = {
        CanaryStage.CANARY  : 3600,    # 1 hour
        CanaryStage.EARLY   : 7200,    # 2 hours
        CanaryStage.HALF    : 14400,   # 4 hours
        CanaryStage.MAJORITY: 14400,   # 4 hours
    }

    def start(self):
        """Begin the canary deployment at 5% traffic."""
        self.current_stage = CanaryStage.CANARY
        self.stage_metrics[CanaryStage.CANARY] = CanaryMetrics(stage=CanaryStage.CANARY)
        print(f"Canary '{self.name}' started: {self.stable_model} -> {self.canary_model}")
        print(f"  Stage: CANARY (5% traffic to new model)")

    def get_model_for_request(self, pr_number: int) -> tuple[str, str, str]:
        """
        Decide which model handles this request.
        Returns: (model, provider, variant_name)
        """
        canary_traffic = STAGE_TRAFFIC.get(self.current_stage, 0)
        if canary_traffic == 0:
            return self.stable_model, self.stable_provider, "stable"
        if canary_traffic == 100:
            return self.canary_model, self.canary_provider, "canary"

        # Use PR number modulo for deterministic assignment
        if (pr_number % 100) < canary_traffic:
            return self.canary_model, self.canary_provider, "canary"
        return self.stable_model, self.stable_provider, "stable"

    def record(self, variant: str, latency: float, error: bool = False, issues: int = 0):
        """Record metrics for current stage."""
        metrics = self.stage_metrics.get(self.current_stage)
        if not metrics:
            return
        metrics.total_calls   += 1
        metrics.total_latency += latency
        metrics.issues_found  += issues
        if error:
            metrics.total_errors += 1

        # Auto-advance check
        if self.auto_advance:
            self._check_advance()

    def _check_advance(self):
        """Check if conditions are met to advance to next stage."""
        metrics = self.stage_metrics.get(self.current_stage)
        if not metrics:
            return

        # Check minimum duration
        min_duration = self.STAGE_MIN_DURATION.get(self.current_stage, 0)
        elapsed      = time.time() - metrics.start_time
        if elapsed < min_duration:
            return

        # Check health metrics
        if not metrics.is_healthy():
            self.rollback(f"Unhealthy at stage {self.current_stage.value}: "
                         f"error_rate={metrics.error_rate():.1f}%")
            return

        # Advance to next stage
        self.advance()

    def advance(self) -> bool:
        """Manually advance to next stage."""
        idx = STAGE_ORDER.index(self.current_stage)
        if idx >= len(STAGE_ORDER) - 1:
            print(f"Canary '{self.name}' already at FULL stage")
            return False

        next_stage         = STAGE_ORDER[idx + 1]
        self.current_stage = next_stage
        self.stage_metrics[next_stage] = CanaryMetrics(stage=next_stage)

        traffic = STAGE_TRAFFIC[next_stage]
        print(f"Canary '{self.name}' advanced to {next_stage.value} ({traffic}% traffic)")
        return True

    def rollback(self, reason: str = ""):
        """Emergency rollback to stable model."""
        self.rollback_reason = reason
        self.current_stage   = CanaryStage.INACTIVE
        print(f"ROLLBACK: Canary '{self.name}' rolled back to {self.stable_model}")
        if reason:
            print(f"  Reason: {reason}")

    def get_status(self) -> dict:
        metrics = self.stage_metrics.get(self.current_stage)
        return {
            "name"           : self.name,
            "stable_model"   : self.stable_model,
            "canary_model"   : self.canary_model,
            "current_stage"  : self.current_stage.value,
            "canary_traffic" : STAGE_TRAFFIC.get(self.current_stage, 0),
            "stable_traffic" : 100 - STAGE_TRAFFIC.get(self.current_stage, 0),
            "rollback_reason": self.rollback_reason,
            "current_metrics": {
                "calls"       : metrics.total_calls if metrics else 0,
                "error_rate"  : round(metrics.error_rate(), 2) if metrics else 0,
                "avg_latency" : round(metrics.avg_latency(), 2) if metrics else 0,
                "healthy"     : metrics.is_healthy() if metrics else True,
            } if metrics else {}
        }


class CanaryManager:
    """
    Manages multiple canary deployments.
    In CodeGuard, each agent can have its own independent canary.
    """

    def __init__(self):
        self.deployments: dict[str, CanaryDeployment] = {}

    def create(
        self,
        name           : str,
        stable_model   : str,
        canary_model   : str,
        stable_provider: str = "openai",
        canary_provider: str = "fireworks",
        auto_advance   : bool = True
    ) -> CanaryDeployment:
        deployment = CanaryDeployment(
            name            = name,
            stable_model    = stable_model,
            canary_model    = canary_model,
            stable_provider = stable_provider,
            canary_provider = canary_provider,
            auto_advance    = auto_advance,
        )
        self.deployments[name] = deployment
        return deployment

    def get(self, name: str) -> Optional[CanaryDeployment]:
        return self.deployments.get(name)

    def get_all_status(self) -> dict:
        return {
            name: dep.get_status()
            for name, dep in self.deployments.items()
        }


# ── Global singleton ──────────────────────────────────────────────
canary_manager = CanaryManager()
