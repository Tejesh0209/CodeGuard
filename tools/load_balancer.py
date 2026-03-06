import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal


ProviderName = Literal["openai", "fireworks", "vllm"]


@dataclass
class ProviderStats:
    name          : str
    weight        : float = 1.0    # higher = more traffic
    latencies     : deque = field(default_factory=lambda: deque(maxlen=20))
    errors        : deque = field(default_factory=lambda: deque(maxlen=10))
    total_calls   : int   = 0
    total_errors  : int   = 0
    total_tokens  : int   = 0
    circuit_open  : bool  = False   # True = provider disabled
    circuit_until : float = 0.0     # timestamp when circuit closes again

    def avg_latency(self) -> float:
        return round(sum(self.latencies) / len(self.latencies), 2) if self.latencies else 0.0

    def error_rate(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)

    def record_success(self, latency: float, tokens: int = 0):
        self.latencies.append(latency)
        self.errors.append(0)
        self.total_calls  += 1
        self.total_tokens += tokens
        # Close circuit if it was open and provider recovered
        if self.circuit_open and time.time() > self.circuit_until:
            self.circuit_open = False
            print(f"   [LoadBalancer] Circuit CLOSED for {self.name} - recovered")

    def record_failure(self):
        self.errors.append(1)
        self.total_errors += 1
        # Open circuit breaker if error rate > 50% in last 10 calls
        if self.error_rate() > 0.5 and len(self.errors) >= 3:
            self.circuit_open  = True
            self.circuit_until = time.time() + 60  # disable for 60 seconds
            print(f"   [LoadBalancer] Circuit OPEN for {self.name} - too many errors")

    def is_available(self) -> bool:
        if self.circuit_open:
            if time.time() > self.circuit_until:
                self.circuit_open = False  # auto-recover after timeout
                return True
            return False
        return True


class LoadBalancer:
    """
    Weighted round-robin load balancer across LLM providers.

    Features:
    - Weighted routing: send more traffic to faster/cheaper providers
    - Circuit breaker: disable provider after repeated failures
    - Latency tracking: auto-adjust weights based on performance
    - Cost tracking: know how much each provider costs

    In CodeGuard:
      OpenAI   weight=3  -> 50% of traffic (high quality)
      Fireworks weight=3 -> 50% of traffic (cost efficient)
      vLLM     weight=0  -> 0% normally, only as fallback
    """

    def __init__(self):
        self.providers = {
            "openai"    : ProviderStats(name="openai",     weight=3.0),
            "fireworks" : ProviderStats(name="fireworks",  weight=3.0),
            "vllm"      : ProviderStats(name="vllm",       weight=0.0),
        }
        self._round_robin_index = 0

    def get_next_provider(self, exclude: list[str] = None) -> str | None:
        """
        Pick next provider using weighted round-robin.
        Skips providers with open circuit breakers.
        exclude = list of providers already tried (for fallback chain)
        """
        exclude = exclude or []
        available = [
            (name, stats) for name, stats in self.providers.items()
            if stats.is_available()
            and stats.weight > 0
            and name not in exclude
        ]

        if not available:
            # All providers with weight>0 failed, try vLLM as last resort
            if "vllm" not in exclude:
                return "vllm"
            return None

        # Weighted selection
        total_weight = sum(s.weight for _, s in available)
        if total_weight == 0:
            return available[0][0]

        # Build weighted list and pick by round-robin position
        weighted = []
        for name, stats in available:
            count = max(1, int(stats.weight))
            weighted.extend([name] * count)

        provider = weighted[self._round_robin_index % len(weighted)]
        self._round_robin_index += 1
        return provider

    def record_success(self, provider: str, latency: float, tokens: int = 0):
        if provider in self.providers:
            self.providers[provider].record_success(latency, tokens)
            self._adjust_weights()

    def record_failure(self, provider: str):
        if provider in self.providers:
            self.providers[provider].record_failure()

    def _adjust_weights(self):
        """
        Auto-adjust weights based on latency.
        Faster providers get more traffic.
        """
        latencies = {
            name: stats.avg_latency()
            for name, stats in self.providers.items()
            if stats.latencies and stats.is_available()
        }
        if len(latencies) < 2:
            return

        max_lat = max(latencies.values()) or 1
        for name, lat in latencies.items():
            if name == "vllm":
                continue  # vLLM weight stays 0 unless explicitly needed
            # Lower latency = higher weight
            new_weight = round(max(0.5, (max_lat - lat + 1) / max_lat * 3), 2)
            self.providers[name].weight = new_weight

    def get_status(self) -> dict:
        return {
            name: {
                "weight"       : stats.weight,
                "avg_latency"  : stats.avg_latency(),
                "error_rate"   : round(stats.error_rate() * 100, 1),
                "total_calls"  : stats.total_calls,
                "circuit_open" : stats.circuit_open,
                "available"    : stats.is_available()
            }
            for name, stats in self.providers.items()
        }
