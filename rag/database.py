import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = get_conn()
    cur  = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS code_chunks (
            id            SERIAL PRIMARY KEY,
            repo          TEXT NOT NULL,
            filepath      TEXT NOT NULL,
            function_name TEXT,
            chunk_text    TEXT NOT NULL,
            embedding     vector(384),
            language      TEXT,
            created_at    TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized — pgvector ready")


if __name__ == "__main__":
    init_db()