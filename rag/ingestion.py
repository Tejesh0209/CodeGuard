import os
import ast
import psycopg2
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL    = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class CodeIngestionPipeline:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Embedding model loaded: {EMBEDDING_MODEL}")

    def get_conn(self):
        return psycopg2.connect(DATABASE_URL)

    def chunk_python_file(self, filepath: str, content: str) -> List[Dict]:
        chunks = []
        try:
            tree  = ast.parse(content)
            lines = content.split("\n")
            for node in ast.walk(tree):
                if isinstance(node, (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef
                )):
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
        lines   = content.split("\n")
        chunks  = []
        step    = 50
        overlap = 10
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

    def ingest_repo(self, repo_path: str, repo_name: str):
        supported  = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".c"}
        all_chunks = []

        print(f"\n📂 Scanning repo: {repo_path}")

        for path in Path(repo_path).rglob("*"):
            if any(p in str(path) for p in [
                ".git", "venv", "node_modules",
                "__pycache__", ".env"
            ]):
                continue
            if path.is_file() and path.suffix in supported:
                try:
                    content  = path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(path.relative_to(repo_path))
                    if path.suffix == ".py":
                        chunks = self.chunk_python_file(rel_path, content)
                    else:
                        chunks = self.chunk_generic_file(rel_path, content)
                    for chunk in chunks:
                        chunk["repo"] = repo_name
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

        print(f"   Found {len(all_chunks)} code chunks")
        print(f"   Generating embeddings...")

        texts      = [c["chunk_text"] for c in all_chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )

        print(f"   Storing in pgvector...")

        conn = self.get_conn()
        cur  = conn.cursor()

        cur.execute(
            "DELETE FROM code_chunks WHERE repo = %s",
            (repo_name,)
        )

        for chunk, embedding in zip(all_chunks, embeddings):
            # Format as fixed-point — avoids scientific notation
            emb_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"

            cur.execute("""
                INSERT INTO code_chunks
                    (repo, filepath, function_name, chunk_text, embedding, language)
                VALUES
                    (%s, %s, %s, %s, %s::vector, %s)
            """, (
                repo_name,
                chunk["filepath"],
                chunk["function_name"],
                chunk["chunk_text"],
                emb_str,
                chunk["language"]
            ))

        conn.commit()
        cur.close()
        conn.close()

        print(f"Ingestion complete — {len(all_chunks)} chunks stored\n")
        return len(all_chunks)


if __name__ == "__main__":
    pipeline = CodeIngestionPipeline()
    pipeline.ingest_repo(
        repo_path="/Users/tejeshboppana/SentinelAI",
        repo_name="Tejesh0209/SentinelAI"
    )