import os
import json
import random
import psycopg2
import re
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
from sentence_transformers import CrossEncoder

from app.schemas.users import UserIn, UserOut

DATABASE_URL = os.getenv("DATABASE_URL")
TOP_K_SHORT = 3
TOP_K_LONG = 3
FETCH_K = 25
MIN_VECTOR_SIMILARITY = 0.72
MIN_LEXICAL_OVERLAP = 0.15


TOKEN_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)

STOP_WORDS = {
    "как", "что", "это", "или", "для", "при", "над", "под", "где",
    "если", "чем", "они", "она", "оно", "его", "еще", "уже",
    "такое", "такой", "такая", "такие", "является", "являются",
}

QUERY_EXPANSIONS = {
    # Частый пользовательский typo из твоего примера:
    "предаются": "представляются задаются способы представления графов матрица смежности матрица инцидентности списки смежности",

    # Пользователь пишет «связанные графы», а в теории обычно «связные графы»
    "связанные графы": "связные графы связный граф связность путь между вершинами",
    "связанный граф": "связный граф связность путь между вершинами",
    "связанные": "связные связность",
}
DEFINITION_MARKERS = (
    "называется",
    "называются",
    "определяется",
    "определяются",
    "это",
)

reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

def is_definition_query(query: str) -> bool:
    q = query.lower().replace("ё", "е").strip()
    return (
        len(q.split()) <= 5
        or "что такое" in q
        or q.endswith("это?")
        or q.endswith("это")
    )


def definition_bonus(query: str, text: str) -> float:
    if not is_definition_query(query):
        return 0.0

    low = text.lower().replace("ё", "е")
    bonus = 0.0

    for marker in DEFINITION_MARKERS:
        if marker in low:
            bonus += 0.08

    return min(bonus, 0.20)

def expand_query(text: str) -> str:
    q = text.strip()
    low = q.lower().replace("ё", "е")

    additions = []

    for trigger, expansion in QUERY_EXPANSIONS.items():
        if trigger in low:
            additions.append(expansion)

    if additions:
        return q + " " + " ".join(additions)

    return q


def lexical_tokens(text: str) -> set[str]:
    words = TOKEN_RE.findall(text.lower().replace("ё", "е"))
    return {
        w[:6]
        for w in words
        if w not in STOP_WORDS
    }


def lexical_bonus(query: str, text: str) -> float:
    q_tokens = lexical_tokens(query)
    if not q_tokens:
        return 0.0

    t_tokens = lexical_tokens(text)
    if not t_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / len(q_tokens)


def dedupe_rows(rows, limit: int):
    result = []
    seen = set()

    for _, content, distance in rows:
        key = re.sub(r"\W+", " ", content.lower().replace("ё", "е"))[:220]

        if key in seen:
            continue

        seen.add(key)
        result.append(content)

        if len(result) >= limit:
            break

    return result


def rerank(query: str, rows, limit: int):
    scored = []

    for row in rows:
        content = row[0]
        distance = float(row[1])
        db_lexical = float(row[2] or 0.0) if len(row) > 2 else 0.0

        vector_similarity = 1.0 - distance
        lexical = lexical_bonus(query, content)

        # Отсекаем совсем нерелевантные результаты.
        
        if vector_similarity < MIN_VECTOR_SIMILARITY and lexical < MIN_LEXICAL_OVERLAP:
            continue

        score = (
            distance
            - 0.12 * lexical
            - 0.20 * db_lexical
            - definition_bonus(query, content)
        )
        scored.append((score, content, distance))

    scored.sort(key=lambda x: x[0])
    return dedupe_rows(scored, limit)

app = FastAPI()

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    cache_folder="/models/cache"
)


def cross_rerank(query: str, chunks: list[str], limit: int):
    if not chunks:
        return []

    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return [chunk for chunk, _ in ranked[:limit]]


def get_conn():
    return psycopg2.connect(DATABASE_URL)

def embed_query(text: str):
    q = f"query: {text}"
    emb = model.encode(
        [q],
        normalize_embeddings=True,
    )[0]
    return emb.tolist()

def search_table(table: str, query: str, fetch_k: int):
    if table not in {"documents_short", "documents_long"}:
        raise ValueError("Invalid table")

    expanded_query = expand_query(query)
    emb = embed_query(expanded_query)

    conn = get_conn()
    cur = conn.cursor()

    # Если IVFFlat пока остался, увеличиваем probes.
    # Если индекса нет — не мешает.
    cur.execute("SET LOCAL ivfflat.probes = 50")

    cur.execute(
        f"""
        SELECT content,
               embedding <=> %s::vector AS distance,
               ts_rank_cd(search_vector, plainto_tsquery('russian', %s)) AS lexical_rank
        FROM {table}
        ORDER BY
               (embedding <=> %s::vector)
               - 0.20 * ts_rank_cd(search_vector, plainto_tsquery('russian', %s))
        LIMIT %s
        """,
        (emb, query, emb, query, fetch_k),
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows, expanded_query


@app.post("/api/question", response_model=UserOut)
def search(q: UserIn):
    short_rows, _ = search_table(
        "documents_short",
        q.question,
        FETCH_K,
    )

    long_rows, _ = search_table(
        "documents_long",
        q.question,
        FETCH_K,
    )

    # Сначала дешёвый фильтр: threshold + lexical + definition boost.
    short_candidates = rerank(q.question, short_rows, FETCH_K)
    long_candidates = rerank(q.question, long_rows, FETCH_K)

    # Потом дорогой, но более точный CrossEncoder-rerank.
    short = cross_rerank(q.question, short_candidates, TOP_K_SHORT)
    long = cross_rerank(q.question, long_candidates, TOP_K_LONG)

    return UserOut(short=short, long=long)

# def search_short(query, top_k):
#     emb = embed_query(query)
#     conn = get_conn()
#     cur = conn.cursor()
#     cur.execute(
#         """
#         SELECT content,
#                embedding <=> %s::vector AS distance
#         FROM documents_short
#         ORDER BY embedding <=> %s::vector
#         LIMIT %s
#         """,
#         (emb, emb, top_k),
#     )
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     return rows

# def search_long(query, top_k):
#     emb = embed_query(query)
#     conn = get_conn()
#     cur = conn.cursor()
#     cur.execute(
#         """
#         SELECT content,
#                embedding <=> %s::vector AS distance
#         FROM documents_long
#         ORDER BY embedding <=> %s::vector
#         LIMIT %s
#         """,
#         (emb, emb, top_k),
#     )
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     return rows

# @app.post("/api/question", response_model=UserOut)
# def search(q: UserIn):
#     short_rows = search_short(q.question, TOP_K_SHORT)
#     long_rows = search_long(q.question, TOP_K_LONG)
#     short = [r[0] for r in short_rows]
#     long = [r[0] for r in long_rows]
#     return UserOut(short=short, long=long)

# ---------- Модели данных для тренажёра ----------
class TrainerQuestionResponse(BaseModel):
    id: int
    question: str

class TrainerCheckRequest(BaseModel):
    question_id: int
    answer: str

class TrainerCheckResponse(BaseModel):
    status: str          # "Верно", "Неверно", "Верно частично"
    explanation: str

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

def _to_list(emb):
    """Преобразует эмбеддинг из разных форматов в список float"""
    if hasattr(emb, 'tolist'):
        return emb.tolist()
    if isinstance(emb, str):
        return json.loads(emb)
    return list(emb)

@app.get("/api/trainer/question", response_model=TrainerQuestionResponse)
def get_random_question():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM questions")
    total = cur.fetchone()[0]
    if total == 0:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Нет вопросов. Запустите bootstrapper с REBUILD_EMBEDDINGS=true")
    random_id = random.randint(1, total)
    cur.execute("SELECT id, question_text FROM questions WHERE id = %s", (random_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Вопрос не найден")
    return TrainerQuestionResponse(id=row[0], question=row[1])

@app.post("/api/trainer/check", response_model=TrainerCheckResponse)
def check_answer(req: TrainerCheckRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT answer_text, embedding FROM questions WHERE id = %s", (req.question_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Вопрос не найден")
    answer_text, embedding_ref = row
    # Вычисляем эмбеддинг ответа пользователя
    user_emb = embed_query(req.answer)
    # Приводим эталонный эмбеддинг к списку float
    ref_emb = _to_list(embedding_ref)
    similarity = cosine_similarity(user_emb, ref_emb)

    # Настроенные пороги для реалистичной оценки
    if similarity >= 0.9:
        status = "Верно"
        explanation = ""
    elif similarity >= 0.8:
        status = "Верно частично"
        explanation = answer_text
    else:
        status = "Неверно"
        explanation = answer_text

    return TrainerCheckResponse(status=status, explanation=explanation)