import os
import psycopg2
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from app.schemas.users import UserIn, UserOut

DATABASE_URL = os.getenv("DATABASE_URL")
TOP_K_SHORT = 5
TOP_K_LONG = 5

app = FastAPI()

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    cache_folder="/models/cache"
)

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def embed_query(text: str):
    q = f"query: {text}"
    emb = model.encode(
        [q],
        normalize_embeddings=True,
    )[0]
    return emb.tolist()

def search_short(query, top_k):
    emb = embed_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM documents_short
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (emb, emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def search_long(query, top_k):
    emb = embed_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM documents_long
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (emb, emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

@app.post("/api/question", response_model=UserOut)
def search(q: UserIn):

    short_rows = search_short(q.question, 5)
    long_rows = search_long(q.question, 5)

    short = [r[0] for r in short_rows]
    long = [r[0] for r in long_rows]

    return UserOut(
        short=short,
        long=long,
    )