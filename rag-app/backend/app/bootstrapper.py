import os
import glob
import logging

import psycopg2
import numpy as np

from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("bootstrapper")


DATABASE_URL = os.getenv("DATABASE_URL")
PDF_PATH = os.getenv("PDF_PATH", "/app/sources")
REBUILD = os.getenv("REBUILD_EMBEDDINGS", "false").lower() == "true"

log.info(f"DATABASE_URL={DATABASE_URL}")
log.info(f"PDF_PATH={PDF_PATH}")
log.info(f"REBUILD_EMBEDDINGS={REBUILD}")


log.info("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

log.info("Model loaded")


def get_conn():

    log.info("Connecting to DB...")

    conn = psycopg2.connect(DATABASE_URL)

    log.info("DB connected")

    return conn


def has_data():

    try:

        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT count(*) FROM documents")

        count = cur.fetchone()[0]

        cur.close()
        conn.close()

        log.info(f"Documents in DB: {count}")

        return count > 0

    except Exception as e:

        log.error(f"has_data error: {e}")

        return False


def clear_table():

    log.info("Clearing table documents...")

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE documents")

    conn.commit()

    cur.close()
    conn.close()

    log.info("Table cleared")



def pdf_to_markdown(pdf_path: str) -> str:

    log.info(f"Converting PDF -> markdown: {pdf_path}")

    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    md = result.document.export_to_markdown()

    log.info(f"Markdown length: {len(md)}")

    return md



def chunk_markdown(text, chunk_size=500, overlap=100):

    log.info("Chunking markdown...")

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    log.info(f"Chunks created: {len(chunks)}")

    return chunks



def embed(texts):

    log.info(f"Embedding {len(texts)} chunks...")

    emb = model.encode(texts)

    log.info("Embedding done")

    return emb


def save_chunks(chunks, embeddings):

    log.info("Saving to DB...")

    conn = get_conn()
    cur = conn.cursor()

    for i, (text, emb) in enumerate(zip(chunks, embeddings)):

        cur.execute(
            """
            INSERT INTO documents (content, embedding)
            VALUES (%s, %s)
            """,
            (text, emb.tolist()),
        )

        if i % 50 == 0:
            log.info(f"Inserted {i}")

    conn.commit()

    cur.close()
    conn.close()

    log.info("Saved to DB")


def main():

    log.info("BOOTSTRAPPER START")

    if has_data():

        if not REBUILD:
            log.info("Embeddings exist → skip")
            return

        log.info("REBUILD=true → clearing table")
        clear_table()

    log.info("Searching PDF files...")

    pdf_files = glob.glob(f"{PDF_PATH}/*.pdf")

    if not pdf_files:
        log.error("No PDF found!")
        return

    log.info(f"Found {len(pdf_files)} pdf")

    all_chunks = []

    for pdf in pdf_files:

        log.info(f"Processing {pdf}")

        md = pdf_to_markdown(pdf)

        chunks = chunk_markdown(md)

        all_chunks.extend(chunks)

    log.info(f"Total chunks: {len(all_chunks)}")

    embeddings = embed(all_chunks)

    save_chunks(all_chunks, embeddings)

    log.info("BOOTSTRAPPER DONE")


if __name__ == "__main__":
    main()