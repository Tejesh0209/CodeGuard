import os
import ast
from pathlib import Path
from typing import List, Dict
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rag.weaviate_schema import get_client

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class WeaviateIngestionPipeline:
    def __init__(self):
        self.model  = SentenceTransformer(EMBEDDING_MODEL)
        self.client = get_client()
        print(f"Weaviate ingestion ready — model: {EMBEDDING_MODEL}")

    def chunk_python_file(self, filepath: str, content: str) -> List[Dict]:
        chunks = []
        try:
            tree  = ast.parse(content)
            lines = content.split("\n")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start      = node.lineno - 1
                    end        = node.end_lineno
                    chunk_text = "\n".join(lines[start:end])
                    if len(chunk_text.strip()) > 20:
                        chunks.append({
                            "filepath"      : filepath,
                            "function_name" : node.name,
                            "chunk_text"    : chunk_text,
                            "language"      : "python"
                        })
        except SyntaxError:
            chunks.append({
                "filepath"      : filepath,
                "function_name" : None,
                "chunk_text"    : content[:2000],
                "language"      : "python"
            })
        return chunks

    def chunk_generic_file(self, filepath: str, content: str) -> List[Dict]:
        lines  = content.split("\n")
        chunks = []
        step, overlap = 50, 10
        for i in range(0, len(lines), step - overlap):
            chunk_text = "\n".join(lines[i:i + step]).strip()
            if len(chunk_text) > 20:
                chunks.append({
                    "filepath"      : filepath,
                    "function_name" : None,
                    "chunk_text"    : chunk_text,
                    "language"      : Path(filepath).suffix.lstrip(".")
                })
        return chunks

    def ingest_repo(self, repo_path: str, repo_name: str) -> int:
        supported  = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".c"}
        all_chunks = []

        print(f"\nScanning repo: {repo_path}")

        for path in Path(repo_path).rglob("*"):
            if any(p in str(path) for p in [".git", "venv", "node_modules", "__pycache__"]):
                continue
            if path.is_file() and path.suffix in supported:
                try:
                    content  = path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(path.relative_to(repo_path))
                    chunks   = (
                        self.chunk_python_file(rel_path, content)
                        if path.suffix == ".py"
                        else self.chunk_generic_file(rel_path, content)
                    )
                    for chunk in chunks:
                        chunk["repo"] = repo_name
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

        print(f"   Found {len(all_chunks)} code chunks")
        print(f"   Generating embeddings...")

        texts      = [c["chunk_text"] for c in all_chunks]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)

        print(f"   Storing in Weaviate...")

        collection = self.client.collections.get("CodeChunk")

        # Delete existing chunks for this repo
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("repo").equal(repo_name)
        )

        # Batch insert — much faster than one-by-one
        with collection.batch.dynamic() as batch:
            for chunk, embedding in zip(all_chunks, embeddings):
                batch.add_object(
                    properties = {
                        "repo"          : chunk["repo"],
                        "filepath"      : chunk["filepath"],
                        "function_name" : chunk.get("function_name") or "",
                        "chunk_text"    : chunk["chunk_text"],
                        "language"      : chunk["language"]
                    },
                    vector = embedding.tolist()
                )

        print(f"Ingestion complete — {len(all_chunks)} chunks stored in Weaviate")
        self.client.close()
        return len(all_chunks)


if __name__ == "__main__":
    from rag.weaviate_schema import init_schema
    init_schema()

    pipeline = WeaviateIngestionPipeline()
    pipeline.ingest_repo(
        repo_path = "/Users/tejeshboppana/SentinelAI",
        repo_name = "Tejesh0209/SentinelAI"
    )
