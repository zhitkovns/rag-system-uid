import os
import sys
import glob
import logging
import traceback
import psycopg2
from app.chunking import chunk_text
from sentence_transformers import SentenceTransformer
from app.question_generator import generate_and_store_questions, question_exists, clear_questions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bootstrapper")

DATABASE_URL = os.getenv("DATABASE_URL")
DEFAULT_TXT_PATH = os.getenv("PDF_PATH", "/app/sources")
REBUILD = os.getenv("REBUILD_EMBEDDINGS", "false").lower() == "true"

def get_conn():
    log.info("Connecting to DB")
    return psycopg2.connect(DATABASE_URL)

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

# def chunk_tokens(text, model, chunk_size, overlap):
#     tokenizer = model.tokenizer
#     tokens = tokenizer.encode(text)
#     chunks = []
#     start = 0
#     while start < len(tokens):
#         end = start + chunk_size
#         chunk = tokens[start:end]
#         chunk_text = tokenizer.decode(chunk)
#         chunks.append(chunk_text)
#         start += chunk_size - overlap
#     log.info(f"Chunks created: {len(chunks)} | size={chunk_size}")
#     return chunks

def embed(texts, model, prefix):
    log.info(f"Embedding {len(texts)}")
    texts = [f"{prefix}: {t}" for t in texts]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    log.info("Embedding done")
    return emb

def save_short(chunks, embeddings):
    log.info("Saving SHORT")
    conn = get_conn()
    cur = conn.cursor()
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            "INSERT INTO documents_short (content, embedding) VALUES (%s, %s)",
            (text, emb.tolist()),
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
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            "INSERT INTO documents_long (content, embedding) VALUES (%s, %s)",
            (text, emb.tolist()),
        )
        if i % 100 == 0:
            log.info(f"long inserted {i}")
    conn.commit()
    cur.close()
    conn.close()

def main():
    log.info("BOOTSTRAPPER START")
    try:
        # Определяем путь к файлам
        txt_path = DEFAULT_TXT_PATH
        txt_files = glob.glob(f"{txt_path}/*.txt")
        if not txt_files:
            log.warning(f"В {txt_path} нет .txt файлов, пробуем встроенный путь /app/builtin_sources")
            builtin_path = "/app/builtin_sources"
            txt_files = glob.glob(f"{builtin_path}/*.txt")
            if txt_files:
                txt_path = builtin_path
                log.info(f"Используем встроенный источник: {builtin_path}")
            else:
                log.error("Нет файлов .txt ни в одном источнике")
                return

        log.info("Loading model")
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        log.info("Model loaded")

        rebuild_docs = False
        if has_data():
            if not REBUILD:
                log.info("Данные документов существуют, пропускаем перестроение")
            else:
                log.info("REBUILD=true -> очищаем таблицы документов")
                clear_tables()
                rebuild_docs = True
        else:
            rebuild_docs = True

        if rebuild_docs:
            txt_files = glob.glob(f"{txt_path}/*.txt")
            if not txt_files:
                log.error("Нет .txt файлов для построения документов")
                return
            log.info(f"TXT found: {len(txt_files)}")
            short_all = []
            long_all = []
            for file in txt_files:
                text = read_txt(file)
                short_chunks = chunk_text(
                        text,
                        model,
                        max_tokens=90,
                        overlap_sentences=1,
                    )

                long_chunks = chunk_text(
                        text,
                        model,
                        max_tokens=240,
                        overlap_sentences=2,
                    )
                short_all.extend(short_chunks)
                long_all.extend(long_chunks)
            log.info(f"Total short = {len(short_all)}")
            log.info(f"Total long = {len(long_all)}")
            short_emb = embed(short_all, model, prefix="passage")
            long_emb = embed(long_all, model, prefix="passage")
            save_short(short_all, short_emb)
            save_long(long_all, long_emb)

        log.info("Проверяем наличие вопросов в БД")
        conn_check = get_conn()
        try:
            questions_exist = question_exists(conn_check)
        finally:
            conn_check.close()
        log.info(f"Вопросы существуют: {questions_exist}, REBUILD={REBUILD}")

        if REBUILD or not questions_exist:
            log.info("Начинаем генерацию вопросов")
            if REBUILD and questions_exist:
                log.info("Очищаем старые вопросы")
                conn_clear = get_conn()
                try:
                    clear_questions(conn_clear)
                finally:
                    conn_clear.close()
            generate_and_store_questions(txt_path, model, DATABASE_URL)
        else:
            log.info("Вопросы уже есть, пропускаем генерацию")

        log.info("BOOTSTRAPPER DONE")
    except Exception as e:
        log.error(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()