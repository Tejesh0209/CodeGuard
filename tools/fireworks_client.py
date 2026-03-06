import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FIREWORKS_API_KEY  = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Available models on Fireworks AI
FIREWORKS_MODELS = {
    "mixtral-22b" : "accounts/fireworks/models/mixtral-8x22b-instruct",
    "deepseek-v3" : "accounts/fireworks/models/deepseek-v3p2",
    "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
    "gpt-oss-20b" : "accounts/fireworks/models/gpt-oss-20b",
}

class FireworksClient:
    """
    OpenAI-compatible client for Fireworks AI.
    Fireworks runs open-source models (Llama, Mixtral, Qwen)
    at ~16x lower cost than GPT-4o with similar quality.
    """

    def __init__(self, model: str = "llama-70b"):
        self.model_key = model
        self.model_id  = FIREWORKS_MODELS.get(model, FIREWORKS_MODELS["mixtral-22b"])
        self.client    = OpenAI(
            api_key  = FIREWORKS_API_KEY,
            base_url = FIREWORKS_BASE_URL
        )
        self.total_tokens = 0
        self.total_calls  = 0
        self.total_errors = 0
        self.latencies    = []

    def invoke(self, messages: list, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """
        Call Fireworks AI. Same interface as LangChain .invoke()
        Returns content string directly.
        """
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model       = self.model_id,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens
            )
            latency = round(time.time() - start, 2)
            self.latencies.append(latency)
            self.total_calls  += 1
            self.total_tokens += resp.usage.total_tokens if resp.usage else 0

            content = resp.choices[0].message.content
            print(f"   [Fireworks {self.model_key}] {latency}s | "
                  f"{resp.usage.total_tokens if resp.usage else '?'} tokens")
            return content

        except Exception as e:
            self.total_errors += 1
            print(f"   [Fireworks ERROR] {e}")
            raise

    def get_stats(self) -> dict:
        avg_latency = (
            round(sum(self.latencies) / len(self.latencies), 2)
            if self.latencies else 0
        )
        return {
            "provider"    : "fireworks",
            "model"       : self.model_key,
            "total_calls" : self.total_calls,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "avg_latency" : avg_latency,
            "est_cost_usd": round(self.total_tokens * 0.0000009, 4)  # $0.9/M tokens
        }

def test_fireworks():
    client = FireworksClient(model="deepseek-v3")
    response = client.invoke([
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user",   "content": "In one sentence, what is SQL injection?"}
    ])
    print(f"\nFireworks response: {response}")
    print(f"Stats: {client.get_stats()}")

if __name__ == "__main__":
    test_fireworks()
