import os
from typing import List, Dict
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery, HybridFusion
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rag.weaviate_schema import get_client

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder  = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading reranker model...")
        self.reranker  = CrossEncoder(RERANKER_MODEL)

        self.client    = get_client()
        self.collection = self.client.collections.get("CodeChunk")
        print(f"HybridRetriever ready")
        print(f"  Embedder : {EMBEDDING_MODEL}")
        print(f"  Reranker : {RERANKER_MODEL}")

    def retrieve(
        self,
        query     : str,
        repo      : str,
        top_k     : int = 5,
        alpha     : float = 0.5,    # 0=BM25 only, 1=dense only, 0.5=balanced
        rerank    : bool = True,
        fetch_k   : int = 20        # fetch more, rerank down to top_k
    ) -> List[Dict]:
        """
        Hybrid search: BM25 + Dense vectors, then rerank with CrossEncoder.

        alpha controls the balance:
          alpha=0.0 -> pure BM25 (keyword matching)
          alpha=0.5 -> balanced hybrid
          alpha=1.0 -> pure dense (semantic similarity)
        """
        # Generate query embedding for dense search
        query_vector = self.embedder.encode(query).tolist()

        # Hybrid search — Weaviate combines BM25 + vector internally
        response = self.collection.query.hybrid(
            query      = query,           # for BM25
            vector     = query_vector,    # for dense
            alpha      = alpha,           # fusion weight
            fusion_type = HybridFusion.RELATIVE_SCORE,  # normalize scores before fusion
            limit      = fetch_k,         # fetch more for reranking
            filters    = wvc.query.Filter.by_property("repo").equal(repo),
            return_metadata = MetadataQuery(score=True, explain_score=True)
        )

        if not response.objects:
            return []

        # Extract results
        candidates = []
        for obj in response.objects:
            candidates.append({
                "filepath"      : obj.properties.get("filepath", ""),
                "function_name" : obj.properties.get("function_name", ""),
                "chunk_text"    : obj.properties.get("chunk_text", ""),
                "language"      : obj.properties.get("language", ""),
                "hybrid_score"  : round(obj.metadata.score or 0, 4),
                "rerank_score"  : None
            })

        if not rerank or len(candidates) <= 1:
            return candidates[:top_k]

        # Reranking with CrossEncoder
        # CrossEncoder reads (query, document) pairs and scores relevance
        # Much more accurate than cosine similarity alone
        pairs  = [(query, c["chunk_text"][:512]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = round(float(score), 4)

        # Sort by rerank score (higher = more relevant)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    def retrieve_multi_query(
        self,
        queries: List[str],
        repo   : str,
        top_k  : int = 5
    ) -> List[Dict]:
        """
        Run multiple queries and merge+deduplicate results.
        Used when diff has multiple files/functions to check.
        """
        seen    = set()
        results = []

        for query in queries:
            hits = self.retrieve(query, repo, top_k=top_k, rerank=False)
            for hit in hits:
                key = f"{hit['filepath']}:{hit['function_name']}"
                if key not in seen:
                    seen.add(key)
                    results.append(hit)

        if not results:
            return []

        # Final rerank across all merged results
        pairs  = [(queries[0], r["chunk_text"][:512]) for r in results]
        scores = self.reranker.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = round(float(score), 4)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]

    def format_for_prompt(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No similar code found in team codebase."

        lines = ["### Similar code from team codebase (hybrid search):\n"]
        for i, chunk in enumerate(chunks, 1):
            score_info = (
                f"rerank={chunk['rerank_score']}"
                if chunk["rerank_score"] is not None
                else f"hybrid={chunk['hybrid_score']}"
            )
            lines.append(
                f"**Example {i}** — `{chunk['filepath']}` "
                f"| fn: `{chunk['function_name'] or 'N/A'}` "
                f"| score: {score_info}\n"
                f"```python\n{chunk['chunk_text'][:400]}\n```\n"
            )
        return "\n".join(lines)

    def close(self):
        self.client.close()



# ── Singleton ─────────────────────────────────────────────────────
# Shared across all 4 agents - loads models only ONCE
_hybrid_retriever_instance = None

def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRetriever()
    return _hybrid_retriever_instance


# ── Compatibility wrapper ─────────────────────────────────────────
# Keeps existing agent code working without changes

class CodeRetriever:
    """Drop-in replacement for old pgvector CodeRetriever."""

    def __init__(self):
        self._hybrid = get_hybrid_retriever()

    def retrieve(self, query: str, repo: str, top_k: int = 5) -> List[Dict]:
        results = self._hybrid.retrieve(query, repo, top_k=top_k)
        # Add 'similarity' key for backward compatibility
        for r in results:
            r["similarity"] = r.get("rerank_score") or r.get("hybrid_score") or 0
        return results

    def format_for_prompt(self, chunks: List[Dict]) -> str:
        return self._hybrid.format_for_prompt(chunks)


if __name__ == "__main__":
    retriever = HybridRetriever()

    print("\nTest 1: Pure hybrid search")
    results = retriever.retrieve(
        query = "SQL injection parameterized queries sqlite3",
        repo  = "Tejesh0209/SentinelAI",
        top_k = 3,
        alpha = 0.5
    )
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  {r['filepath']} -> {r['function_name']} | rerank={r['rerank_score']}")

    print("\nTest 2: BM25 dominant (alpha=0.1)")
    results = retriever.retrieve(
        query = "cursor execute SELECT FROM users",
        repo  = "Tejesh0209/SentinelAI",
        top_k = 3,
        alpha = 0.1
    )
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  {r['filepath']} -> {r['function_name']} | rerank={r['rerank_score']}")

    retriever.close()

class CodeRetriever:
    """Drop-in replacement for old pgvector CodeRetriever."""

    def __init__(self):
        self._hybrid = get_hybrid_retriever()

    def retrieve(self, query: str, repo: str, top_k: int = 5) -> List[Dict]:
        results = self._hybrid.retrieve(query, repo, top_k=top_k)
        for r in results:
            r["similarity"] = r.get("rerank_score") or r.get("hybrid_score") or 0
        return results

    def format_for_prompt(self, chunks: List[Dict]) -> str:
        return self._hybrid.format_for_prompt(chunks)

    def __del__(self):
        try:
            self._hybrid.close()
        except Exception:
            pass