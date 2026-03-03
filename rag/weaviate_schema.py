import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, VectorDistances


def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host="localhost", port=8080)


def init_schema():
    client = get_client()

    # Delete existing collection if exists (clean slate)
    if client.collections.exists("CodeChunk"):
        client.collections.delete("CodeChunk")
        print("Deleted existing CodeChunk collection")

    # Create collection with BM25 + vector hybrid support
    client.collections.create(
        name        = "CodeChunk",
        description = "Code chunks from team repositories for RAG retrieval",

        # Vector index config — HNSW for fast dense search
        vector_index_config = Configure.VectorIndex.hnsw(
            distance_metric = VectorDistances.COSINE
        ),

        # BM25 index config — for keyword search
        inverted_index_config = Configure.inverted_index(
            bm25_b           = 0.75,   # document length normalization
            bm25_k1          = 1.2,    # term frequency saturation
            index_timestamps = True
        ),

        # Properties — what we store per chunk
        properties = [
            Property(name="repo",          data_type=DataType.TEXT),
            Property(name="filepath",      data_type=DataType.TEXT),
            Property(name="function_name", data_type=DataType.TEXT),
            Property(name="chunk_text",    data_type=DataType.TEXT),
            Property(name="language",      data_type=DataType.TEXT),
        ]
    )

    print("Weaviate schema created — CodeChunk collection ready")
    print("  BM25: b=0.75, k1=1.2")
    print("  Vector: HNSW cosine similarity")
    client.close()


if __name__ == "__main__":
    init_schema()
