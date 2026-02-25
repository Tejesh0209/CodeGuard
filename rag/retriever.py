import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL    = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class CodeRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(self, query: str, repo: str, top_k: int = 5) -> list[dict]:
        # Encode query to embedding
        raw = self.model.encode(query).tolist()

        # Fixed-point format — avoids scientific notation that breaks pgvector
        emb_str = "[" + ",".join(f"{v:.8f}" for v in raw) + "]"

        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()

        cur.execute(f"""
            SELECT
                filepath,
                function_name,
                chunk_text,
                language,
                1 - (embedding <=> '{emb_str}'::vector) AS similarity
            FROM code_chunks
            WHERE repo = %s
            ORDER BY embedding <=> '{emb_str}'::vector
            LIMIT %s
        """, (repo, top_k))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "filepath"      : row[0],
                "function_name" : row[1],
                "chunk_text"    : row[2],
                "language"      : row[3],
                "similarity"    : round(float(row[4]), 3)
            }
            for row in rows
        ]

    def format_for_prompt(self, chunks: list[dict]) -> str:
        if not chunks:
            return "No similar code found in team codebase."

        formatted = ["### Similar code from your team's codebase:\n"]
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"""
**Example {i}** — `{chunk['filepath']}`
Function: `{chunk['function_name'] or 'N/A'}`
Similarity: {chunk['similarity']}
```python
{chunk['chunk_text'][:500]}
```
""")
        return "\n".join(formatted)


if __name__ == "__main__":
    retriever = CodeRetriever()
    results   = retriever.retrieve(
        query="def calc(x,y,z): return x*y+z",
        repo="Tejesh0209/SentinelAI",
        top_k=3
    )
    print(f"\nFound {len(results)} similar chunks:")
    for r in results:
        print(f"  📄 {r['filepath']} → {r['function_name']} (similarity: {r['similarity']})")