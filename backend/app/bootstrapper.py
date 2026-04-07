import os
import glob
import logging

import psycopg2
import numpy as np

from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("bootstrapper")


DATABASE_URL = os.getenv("DATABASE_URL")
TXT_PATH = os.getenv("PDF_PATH", "/app/sources")
REBUILD = os.getenv("REBUILD_EMBEDDINGS", "false").lower() == "true"


def get_conn():

    log.info("Connecting to DB")

    conn = psycopg2.connect(DATABASE_URL)

    return conn


def clear_tables():

    log.info("Clearing tables")

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE documents_short")
    cur.execute("TRUNCATE TABLE documents_long")

    conn.commit()

    cur.close()
    conn.close()


def has_data():

    try:

        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT count(*) FROM documents_short")

        count = cur.fetchone()[0]

        cur.close()
        conn.close()

        log.info(f"documents_short rows = {count}")

        return count > 0

    except Exception as e:

        log.error(e)

        return False


def read_txt(path):

    log.info(f"Reading {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    log.info(f"Length = {len(text)}")

    return text


def chunk_tokens(text, model, chunk_size, overlap):

    tokenizer = model.tokenizer

    tokens = tokenizer.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk = tokens[start:end]

        chunk_text = tokenizer.decode(chunk)

        chunks.append(chunk_text)

        start += chunk_size - overlap

    log.info(
        f"Chunks created: {len(chunks)} | size={chunk_size}"
    )

    return chunks


def embed(texts, model, prefix):

    log.info(f"Embedding {len(texts)}")

    texts = [f"{prefix}: {t}" for t in texts]

    emb = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    log.info("Embedding done")

    return emb


def save_short(chunks, embeddings):

    log.info("Saving SHORT")

    conn = get_conn()
    cur = conn.cursor()

    for i, (text, emb) in enumerate(
        zip(chunks, embeddings)
    ):

        cur.execute(
            """
            INSERT INTO documents_short
            (content, embedding)
            VALUES (%s, %s)
            """,
            (
                text,
                emb.tolist(),
            ),
        )

        if i % 100 == 0:
            log.info(f"short inserted {i}")

    conn.commit()

    cur.close()
    conn.close()


def save_long(chunks, embeddings):

    log.info("Saving LONG")

    conn = get_conn()
    cur = conn.cursor()

    for i, (text, emb) in enumerate(
        zip(chunks, embeddings)
    ):

        cur.execute(
            """
            INSERT INTO documents_long
            (content, embedding)
            VALUES (%s, %s)
            """,
            (
                text,
                emb.tolist(),
            ),
        )

        if i % 100 == 0:
            log.info(f"long inserted {i}")

    conn.commit()

    cur.close()
    conn.close()


def main():

    log.info("BOOTSTRAPPER START")

    if has_data():

        if not REBUILD:
            log.info("Data exists -> skip")
            return

        log.info("REBUILD=true")

        clear_tables()

    txt_files = glob.glob(f"{TXT_PATH}/*.txt")

    if not txt_files:

        log.error("No TXT found")

        return

    log.info(f"TXT found: {len(txt_files)}")

    log.info("Loading model")

    model = SentenceTransformer(
        "intfloat/multilingual-e5-base"
    )

    log.info("Model loaded")

    short_all = []
    long_all = []

    for file in txt_files:

        text = read_txt(file)

        short_chunks = chunk_tokens(
            text,
            model,
            chunk_size=15,
            overlap=3,
        )

        long_chunks = chunk_tokens(
            text,
            model,
            chunk_size=100,
            overlap=20,
        )

        short_all.extend(short_chunks)
        long_all.extend(long_chunks)

    log.info(f"Total short = {len(short_all)}")
    log.info(f"Total long = {len(long_all)}")

    short_emb = embed(
        short_all,
        model,
        prefix="passage",
    )

    long_emb = embed(
        long_all,
        model,
        prefix="passage",
    )

    save_short(short_all, short_emb)

    save_long(long_all, long_emb)

    log.info("BOOTSTRAPPER DONE")


if __name__ == "__main__":
    main()