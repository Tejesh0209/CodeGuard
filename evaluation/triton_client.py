"""
NVIDIA Triton Inference Server - Client Integration

WHY TRITON:
  Triton is NVIDIA's production inference server, optimized for GPUs.
  It's different from vLLM:

  vLLM:   Specialized for LLM autoregressive generation
          Best for: ChatGPT-style text generation
          Handles: KV cache, continuous batching for LLMs

  Triton: General model serving (supports LLMs, CV models, etc.)
          Best for: embedding models, rerankers, custom models
          Handles: dynamic batching, model ensembles, TensorRT optimization

  In CodeGuard production:
    Triton serves: SentenceTransformer (embedder) + CrossEncoder (reranker)
    vLLM serves:   Llama/DeepSeek for review generation
    Together:      Complete inference infrastructure

  On Mac: Triton server cannot run (needs Linux + NVIDIA GPU)
  We build: HTTP client + model config files (the hard engineering part)
"""
import os
import time
import json
import httpx
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

TRITON_URL   = os.getenv("TRITON_URL",   "http://localhost:8001")
TRITON_MODEL = os.getenv("TRITON_MODEL", "sentence-transformer")


class TritonEmbeddingClient:
    """
    Client for Triton-served SentenceTransformer embedding model.

    In production:
      Triton serves all-MiniLM-L6-v2 with:
        - Dynamic batching: batches requests automatically
        - TensorRT optimization: 3-5x faster than PyTorch
        - GPU utilization: keeps GPU busy with batched requests
        - Multiple model instances: scales with traffic

    In development (Mac):
      TRITON_SIMULATE=true -> calls local SentenceTransformer directly
    """

    def __init__(self):
        self.url      = TRITON_URL
        self.simulate = os.getenv("TRITON_SIMULATE", "true").lower() == "true"
        self._local_model = None

        if self.simulate:
            print("TritonEmbeddingClient: SIMULATION MODE (use TRITON_SIMULATE=false in production)")
        else:
            print(f"TritonEmbeddingClient: connecting to {self.url}")

    def _get_local_model(self):
        """Lazy-load local model for simulation."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._local_model

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        In prod: sends HTTP request to Triton server.
        In dev:  uses local SentenceTransformer directly.
        """
        if self.simulate:
            return self._get_local_model().encode(texts)

        return self._call_triton(texts)

    def _call_triton(self, texts: list[str]) -> np.ndarray:
        """
        Call Triton HTTP inference API.

        Triton REST API format:
        POST /v2/models/{model_name}/infer
        {
            "inputs": [{
                "name": "TEXT",
                "shape": [batch_size, 1],
                "datatype": "BYTES",
                "data": ["text1", "text2", ...]
            }]
        }
        """
        payload = {
            "inputs": [{
                "name"    : "TEXT",
                "shape"   : [len(texts), 1],
                "datatype": "BYTES",
                "data"    : texts
            }]
        }

        try:
            response = httpx.post(
                f"{self.url}/v2/models/{TRITON_MODEL}/infer",
                json    = payload,
                timeout = 30
            )
            response.raise_for_status()
            result = response.json()

            # Extract embeddings from Triton response
            embeddings = np.array(result["outputs"][0]["data"])
            return embeddings.reshape(len(texts), -1)

        except Exception as e:
            print(f"Triton call failed, falling back to local: {e}")
            return self._get_local_model().encode(texts)

    def is_available(self) -> bool:
        """Check if Triton server is running."""
        if self.simulate:
            return True
        try:
            resp = httpx.get(f"{self.url}/v2/health/ready", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class TritonRerankerClient:
    """
    Client for Triton-served CrossEncoder reranker.
    Same pattern as TritonEmbeddingClient but for reranking.
    """

    def __init__(self):
        self.url      = TRITON_URL
        self.simulate = os.getenv("TRITON_SIMULATE", "true").lower() == "true"
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import CrossEncoder
            self._local_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._local_model

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Predict relevance scores for (query, document) pairs.
        """
        if self.simulate:
            return self._get_local_model().predict(pairs).tolist()

        return self._call_triton(pairs)

    def _call_triton(self, pairs: list[tuple[str, str]]) -> list[float]:
        queries   = [p[0] for p in pairs]
        documents = [p[1] for p in pairs]

        payload = {
            "inputs": [
                {
                    "name"    : "QUERY",
                    "shape"   : [len(queries), 1],
                    "datatype": "BYTES",
                    "data"    : queries
                },
                {
                    "name"    : "DOCUMENT",
                    "shape"   : [len(documents), 1],
                    "datatype": "BYTES",
                    "data"    : documents
                }
            ]
        }

        try:
            response = httpx.post(
                f"{self.url}/v2/models/cross-encoder/infer",
                json    = payload,
                timeout = 30
            )
            response.raise_for_status()
            result = response.json()
            return result["outputs"][0]["data"]
        except Exception as e:
            print(f"Triton reranker failed, falling back to local: {e}")
            return self._get_local_model().predict(pairs).tolist()


def generate_triton_model_config():
    """
    Generate Triton model repository configs.
    These files tell Triton how to serve each model.

    In production:
      /models/
        sentence-transformer/
          config.pbtxt        <- model configuration
          1/
            model.onnx        <- exported ONNX model
        cross-encoder/
          config.pbtxt
          1/
            model.onnx
    """
    embedding_config = """
name: "sentence-transformer"
platform: "onnxruntime_onnx"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100
}
input [
  {
    name: "TEXT"
    data_type: TYPE_BYTES
    dims: [1]
  }
]
output [
  {
    name: "EMBEDDING"
    data_type: TYPE_FP32
    dims: [384]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
"""

    reranker_config = """
name: "cross-encoder"
platform: "onnxruntime_onnx"
max_batch_size: 16
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 200
}
input [
  {
    name: "QUERY"
    data_type: TYPE_BYTES
    dims: [1]
  },
  {
    name: "DOCUMENT"
    data_type: TYPE_BYTES
    dims: [1]
  }
]
output [
  {
    name: "SCORE"
    data_type: TYPE_FP32
    dims: [1]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
"""

    os.makedirs("serving/triton_configs/sentence-transformer/1", exist_ok=True)
    os.makedirs("serving/triton_configs/cross-encoder/1",        exist_ok=True)

    with open("serving/triton_configs/sentence-transformer/config.pbtxt", "w") as f:
        f.write(embedding_config)
    with open("serving/triton_configs/cross-encoder/config.pbtxt", "w") as f:
        f.write(reranker_config)

    print("Triton model configs generated:")
    print("  serving/triton_configs/sentence-transformer/config.pbtxt")
    print("  serving/triton_configs/cross-encoder/config.pbtxt")
    print("\nTo deploy on Linux+GPU:")
    print("  docker run --gpus all -v ./serving/triton_configs:/models \\")
    print("    nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print("    tritonserver --model-repository=/models")


if __name__ == "__main__":
    generate_triton_model_config()
    client = TritonEmbeddingClient()
    print(f"\nTriton available: {client.is_available()}")
    embeddings = client.encode(["SQL injection vulnerability", "parameterized queries"])
    print(f"Embeddings shape: {embeddings.shape}")
