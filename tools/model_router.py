import os
import time
from typing import Literal
from langchain_openai import ChatOpenAI
from tools.fireworks_client import FireworksClient
from tools.vllm_client import VLLMClient
from tools.load_balancer import LoadBalancer
from dotenv import load_dotenv

load_dotenv()

TaskType = Literal["security", "style", "performance", "architecture", "autofix", "notification"]


# ── Routing Rules ─────────────────────────────────────────────────
# Defines which provider to use for each task type
# COST CONTEXT:
#   GPT-4o:           $15/M output tokens
#   Fireworks Llama:  $0.9/M output tokens (16x cheaper)
#   vLLM:             $0 (self-hosted)

ROUTING_RULES = {
    "security"     : {"primary": "openai",     "fallback": "fireworks", "model": "gpt-4o"},
    "autofix"      : {"primary": "openai",     "fallback": "fireworks", "model": "gpt-4o"},
    "style"        : {"primary": "fireworks",  "fallback": "openai",    "model": "gpt-oss-20b"},
    "performance"  : {"primary": "fireworks",  "fallback": "openai",    "model": "gpt-oss-20b"},
    "architecture" : {"primary": "fireworks",  "fallback": "openai",    "model": "gpt-oss-20b"},
    "notification" : {"primary": "fireworks",  "fallback": "openai",    "model": "gpt-oss-20b"},
}
class ModelRouter:
    """
    Routes LLM requests to the right provider based on:
    1. Task type (security -> GPT-4o, style -> Fireworks)
    2. Provider health (circuit breaker, latency)
    3. Load balancing (weighted round-robin)
    4. Fallback chain (primary -> fallback -> vLLM)

    Usage:
        router = ModelRouter()
        llm    = router.get_llm(task="security")
        result = llm.invoke(messages)
    """

    def __init__(self):
        self.load_balancer = LoadBalancer()
        self._fireworks    = {}   # cache by model name
        self._vllm         = None
        self._openai       = {}   # cache by model name
        print("ModelRouter initialized")
        print("  security/autofix  -> GPT-4o (primary) -> Fireworks (fallback)")
        print("  style/perf/arch   -> Fireworks Llama-70B (primary) -> GPT-4o (fallback)")

    def get_llm(self, task: TaskType = "style", severity: str = "LOW"):
        """
        Returns the right LLM client for the given task and severity.
        CRITICAL severity always routes to GPT-4o regardless of task.
        """
        # Force GPT-4o for CRITICAL severity
        if severity == "CRITICAL" and task != "notification":
            return self._get_openai("gpt-4o")

        rule     = ROUTING_RULES.get(task, ROUTING_RULES["style"])
        primary  = rule["primary"]
        fallback = rule["fallback"]
        model    = rule["model"]

        # Try primary provider
        if self.load_balancer.providers[primary].is_available():
            print(f"   [Router] {task} -> {primary} ({model})")
            return self._get_client(primary, model)

        # Primary unavailable -> try fallback
        print(f"   [Router] {primary} unavailable, falling back to {fallback}")
        if self.load_balancer.providers[fallback].is_available():
            return self._get_client(fallback, model)

        # Both unavailable -> vLLM last resort
        print(f"   [Router] Both providers down, using vLLM (simulated)")
        return self._get_vllm()

    def invoke_with_tracking(
        self,
        task    : TaskType,
        messages: list,
        severity: str = "LOW"
    ) -> str:
        """
        Invoke LLM with automatic latency + error tracking.
        Tries primary, then fallback, then vLLM.
        """
        rule     = ROUTING_RULES.get(task, ROUTING_RULES["style"])
        primary  = rule["primary"] if severity != "CRITICAL" else "openai"
        fallback = rule["fallback"] if severity != "CRITICAL" else "fireworks"
        tried    = []

        for provider in [primary, fallback, "vllm"]:
            if provider in tried:
                continue
            tried.append(provider)

            try:
                start  = time.time()
                client = self._get_client(provider, rule["model"])
                result = client.invoke(messages) if hasattr(client, "invoke") else str(
                    client.chat.completions.create(
                        model    = rule["model"],
                        messages = messages
                    ).choices[0].message.content
                )
                latency = time.time() - start
                self.load_balancer.record_success(provider, latency)
                return result

            except Exception as e:
                print(f"   [Router] {provider} failed: {e}")
                self.load_balancer.record_failure(provider)
                continue

        raise RuntimeError("All LLM providers failed")

    def _get_client(self, provider: str, model: str):
        if provider == "openai":
            return self._get_openai("gpt-4o")
        elif provider == "fireworks":
            return self._get_fireworks(model)
        else:
            return self._get_vllm()

    def _get_openai(self, model: str = "gpt-4o"):
        if model not in self._openai:
            self._openai[model] = ChatOpenAI(model=model, temperature=0.1)
        return self._openai[model]

    def _get_fireworks(self, model: str = "deepseek-v3") -> ChatOpenAI:
        """
        Use LangChain ChatOpenAI pointed at Fireworks base URL.
        Fireworks is OpenAI-compatible -> works with LangChain | pipe operator.
        """
        model_id = {
            "deepseek-v3" : "accounts/fireworks/models/deepseek-v3p2",
            "mixtral-22b" : "accounts/fireworks/models/mixtral-8x22b-instruct",
            "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
            "gpt-oss-20b" : "accounts/fireworks/models/gpt-oss-20b",
        }.get(model, "accounts/fireworks/models/deepseek-v3p2")

        key = f"fireworks_{model}"
        if key not in self._openai:
            self._openai[key] = ChatOpenAI(
                model            = model_id,
                temperature      = 0.1,
                api_key          = os.getenv("FIREWORKS_API_KEY"),
                base_url         = "https://api.fireworks.ai/inference/v1",
                request_timeout  = 60,
                max_retries      = 1
            )
            print(f"   [Router] Fireworks LangChain client created: {model}")
        return self._openai[key]

    def _get_vllm(self):
        if self._vllm is None:
            self._vllm = VLLMClient()
        return self._vllm

    def get_status(self) -> dict:
        return {
            "load_balancer": self.load_balancer.get_status(),
            "routing_rules": ROUTING_RULES
        }


# ── Global singleton ──────────────────────────────────────────────
model_router = ModelRouter()


def get_llm_for_task(task: TaskType, severity: str = "LOW"):
    """Convenience function used by agents."""
    return model_router.get_llm(task=task, severity=severity)
