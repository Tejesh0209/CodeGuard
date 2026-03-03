# Backward compatibility shim
# All code that imports from rag.retriever now gets hybrid search automatically
from rag.hybrid_retriever import CodeRetriever, HybridRetriever

__all__ = ["CodeRetriever", "HybridRetriever"]
