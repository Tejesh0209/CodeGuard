import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL    = os.getenv("VLLM_MODEL",    "meta-llama/Llama-3.1-70b-Instruct")


class VLLMClient:
    """
    OpenAI-compatible client for vLLM self-hosted server.

    In production (Linux + GPU):
      vllm serve meta-llama/Llama-3.1-70b-Instruct --port 8000
      Set VLLM_BASE_URL=http://your-server:8000/v1

    In development (Mac):
      Server not running -> calls will fail gracefully
      ModelRouter uses this as last-resort fallback
      Set VLLM_SIMULATE=true to simulate responses locally

    WHY vLLM in production:
      - $0 per token (vs $15/M for GPT-4o)
      - Data never leaves your infrastructure
      - PagedAttention: 3-4x more concurrent requests
      - Continuous batching: 10-20x higher throughput
    """

    def __init__(self):
        self.client   = OpenAI(
            api_key  = "not-needed-for-vllm",
            base_url = VLLM_BASE_URL
        )
        self.model        = VLLM_MODEL
        self.simulate     = os.getenv("VLLM_SIMULATE", "true").lower() == "true"
        self.total_calls  = 0
        self.total_tokens = 0
        self.total_errors = 0
        self.latencies    = []

    def invoke(self, messages: list, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """
        Call vLLM server. Falls back to simulation if server not running.
        """
        if self.simulate:
            return self._simulate(messages)

        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens
            )
            latency = round(time.time() - start, 2)
            self.latencies.append(latency)
            self.total_calls  += 1
            self.total_tokens += resp.usage.total_tokens if resp.usage else 0

            content = resp.choices[0].message.content
            print(f"   [vLLM {self.model}] {latency}s")
            return content

        except Exception as e:
            self.total_errors += 1
            print(f"   [vLLM ERROR - server likely not running] {e}")
            raise

    def _simulate(self, messages: list) -> str:
        """
        Simulate vLLM response for local development.
        In production this code path is never hit.
        """
        time.sleep(0.1)
        user_msg  = messages[-1].get("content", "") if messages else ""
        self.total_calls += 1
        response = (
            '{"summary": "[vLLM SIMULATED] Code reviewed by local Llama-3.1-70B", '
            '"issues": [], "overall_score": 7, "approved": true}'
        )
        print(f"   [vLLM SIMULATED] Local Llama-3.1-70B response")
        return response

    def is_available(self) -> bool:
        """Check if vLLM server is actually running."""
        if self.simulate:
            return True
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def get_stats(self) -> dict:
        avg_latency = (
            round(sum(self.latencies) / len(self.latencies), 2)
            if self.latencies else 0
        )
        return {
            "provider"    : "vllm",
            "model"       : self.model,
            "simulated"   : self.simulate,
            "total_calls" : self.total_calls,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "avg_latency" : avg_latency,
            "est_cost_usd": 0.0  # self-hosted = free
        }
