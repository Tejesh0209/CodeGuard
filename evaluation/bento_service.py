import os
import asyncio
from typing import Any
import bentoml
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# ── Request / Response schemas ────────────────────────────────────
class ReviewRequest(BaseModel):
    repo_name  : str
    pr_number  : int
    pr_title   : str
    pr_author  : str
    diff_chunks: list[dict]


class ReviewResponse(BaseModel):
    pr_number   : int
    severity    : str
    total_issues: int
    critical    : int
    high        : int
    approved    : bool
    model_used  : str
    latency_ms  : int


# ── BentoML Service Definition ────────────────────────────────────
@bentoml.service(
    name    = "codeguard-review",
    traffic = {"timeout": 300},
    resources = {"cpu": "2"},
)
class CodeGuardService:
    """
    BentoML service wrapping CodeGuard review pipeline.

    WHY BentoML:
      - Versioned deployments: v1, v2, v3
      - Built-in health checks, metrics, request logging
      - Easy A/B testing via traffic splitting
      - One command to serve: bentoml serve bento_service:CodeGuardService
      - Containerization: bentoml build -> Docker image ready
      - Model registry: track which model version is in production
    """

    def __init__(self):
        from tools.model_router import model_router
        self.model_router = model_router
        print(f"CodeGuardService initialized")
        print(f"  Model router: {self.model_router}")

    @bentoml.api(route="/review")
    async def review(self, request: ReviewRequest) -> ReviewResponse:
        """
        Main review endpoint.
        Called by webhook handler or directly via HTTP.
        """
        import time
        start = time.time()

        from review_agents.security_agent import SecurityAgent
        from review_agents.style_agent    import StyleAgent

        # Run security + style in parallel (fast demo)
        security_agent = SecurityAgent()
        style_agent    = StyleAgent()

        security_result, style_result = await asyncio.gather(
            security_agent.review(request.diff_chunks),
            style_agent.review(request.diff_chunks)
        )

        latency_ms = int((time.time() - start) * 1000)

        critical = security_result.get("critical_count", 0)
        total    = (
            security_result.get("vulnerability_count", 0) +
            style_result.get("issue_count", 0)
        )

        return ReviewResponse(
            pr_number    = request.pr_number,
            severity     = "CRITICAL" if critical > 0 else "HIGH" if total > 5 else "LOW",
            total_issues = total,
            critical     = critical,
            high         = style_result.get("issue_count", 0),
            approved     = critical == 0 and total < 5,
            model_used   = os.getenv("ACTIVE_MODEL", "gpt-4o"),
            latency_ms   = latency_ms
        )

    @bentoml.api(route="/health")
    def health(self) -> dict:
        return {
            "status"      : "healthy",
            "service"     : "codeguard-review",
            "model_router": str(self.model_router.get_status()),
        }

    @bentoml.api(route="/router/status")
    def router_status(self) -> dict:
        return self.model_router.get_status()
