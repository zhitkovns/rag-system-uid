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
from app.llm import (
    generate_dual_answer,
    rephrase_trainer_feedback,
    rephrase_trainer_question,
)

DATABASE_URL = os.getenv("DATABASE_URL")
TOP_K_SHORT = 1
TOP_K_LONG = 2
FETCH_K = 40
MIN_VECTOR_SIMILARITY = 0.72
MIN_LEXICAL_OVERLAP = 0.15
MIN_CROSS_SCORE = -1.5   # Порог cross-encoder: ниже — считаем нерелевантным


TOKEN_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)
SECTION_TITLE_RE = re.compile(r"^(\d+(?:\.\d+)+\.?\s+[^.]{3,120})\.")
LEADING_SECTION_TITLE_RE = re.compile(
    r"^\s*\d+(?:\.\d+)+\.?\s+[^.]{3,140}\.\s*"
)

STOP_WORDS = {
    "как", "что", "это", "или", "для", "при", "над", "под", "где",
    "если", "чем", "они", "она", "оно", "его", "еще", "уже",
    "такое", "такой", "такая", "такие", "является", "являются",
}

RUSSIAN_VOWELS = set('аеёиоуыэюя')
LATIN_VOWELS   = set('aeiou')

def validate_query(text: str) -> tuple[bool, str]:
    """
    Проверяет, является ли запрос осмысленным.
    Возвращает (валидный, причина_отказа).
    """
    text = text.strip()

    # Слишком короткий
    if len(text) < 3:
        return False, "Запрос слишком короткий"

    # Нет ни одной буквы (только цифры/знаки)
    if not re.search(r'[а-яёА-ЯЁa-zA-Z]', text):
        return False, "Запрос не содержит слов"

    # Доля букв слишком мала — много спецсимволов/цифр
    alpha = sum(c.isalpha() for c in text)
    if alpha / len(text) < 0.40:
        return False, "Слишком много нетекстовых символов"

    # Повторяющийся один символ (аааааа, ffffff)
    if re.search(r'(.)\1{4,}', text):
        return False, "Случайный набор символов"

    # Длинные «слова» без гласных → клавиатурный мусор
    words = re.findall(r'[а-яёa-z]+', text.lower())
    long_words = [w for w in words if len(w) > 4]
    if long_words:
        no_vowel = [
            w for w in long_words
            if not any(c in RUSSIAN_VOWELS | LATIN_VOWELS for c in w)
        ]
        if len(no_vowel) / len(long_words) > 0.6:
            return False, "Случайный набор символов"

    return True, ""


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
    stop_words = {
        "это", "как", "что", "или", "для", "при", "над", "под", "так",
        "вот", "где", "она", "они", "оно", "его", "ее", "ещё", "уже",
        "чем", "тем", "без", "про", "когда", "если", "такой", "такая",
        "такие", "является", "который", "которая", "которые", "может",
        "быть", "есть", "все", "всех", "этот", "эта", "эти", "того",
        "того", "того", "суть", "значит", "образом"
    }

    tokens = re.findall(r"[а-яёa-z0-9]+", text.lower())

    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in stop_words
    }


def lexical_bonus(query: str, text: str) -> float:
    q_tokens = lexical_tokens(query)
    if not q_tokens:
        return 0.0

    t_tokens = lexical_tokens(text)
    if not t_tokens:
        return 0.0

    return len(q_tokens & t_tokens) / len(q_tokens)


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def phrase_bonus(query: str, text: str) -> float:
    query_norm = normalize_for_match(query).rstrip("?!.")

    if len(query_norm) < 5:
        return 0.0

    return 0.25 if query_norm in normalize_for_match(text) else 0.0


def section_title(text: str) -> str:
    match = SECTION_TITLE_RE.match(text)
    if not match:
        return ""

    return match.group(1)


def title_bonus(query: str, text: str) -> float:
    q_tokens = lexical_tokens(query)
    if not q_tokens:
        return 0.0

    title_tokens = lexical_tokens(section_title(text))
    if not title_tokens:
        return 0.0

    overlap = len(q_tokens & title_tokens) / len(q_tokens)
    if q_tokens <= title_tokens:
        return 0.18

    return 0.12 * overlap


def section_key(text: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)+\.?\s+.*?\(\d+/\d+\))", text)
    if match:
        return match.group(1).lower().replace("ё", "е")

    match = re.match(r"^(\d+(?:\.\d+)+\.?\s+[^.]{3,120})", text)
    if match:
        return match.group(1).lower().replace("ё", "е")

    return re.sub(r"\W+", " ", text.lower().replace("ё", "е"))[:120]


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = lexical_tokens(a)
    b_tokens = lexical_tokens(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def dedupe_rows(
    rows,
    limit: int,
    max_per_section: int = 1,
    similarity_threshold: float = 0.62,
):
    result = []
    seen_exact = set()
    section_counts = {}

    for _, content, distance in rows:
        exact_key = re.sub(r"\W+", " ", content.lower().replace("ё", "е"))[:260]
        sec_key = section_key(content)

        if exact_key in seen_exact:
            continue

        if section_counts.get(sec_key, 0) >= max_per_section:
            continue

        if any(jaccard_similarity(content, existing) > similarity_threshold for existing in result):
            continue

        seen_exact.add(exact_key)
        section_counts[sec_key] = section_counts.get(sec_key, 0) + 1
        result.append(content)

        if len(result) >= limit:
            break

    return result

def rerank(
    query: str,
    rows,
    limit: int,
    max_per_section: int = 1,
    similarity_threshold: float = 0.62,
):
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
            - title_bonus(query, content)
            - phrase_bonus(query, content)
        )
        scored.append((score, content, distance))

    scored.sort(key=lambda x: x[0])
    return dedupe_rows(
        scored,
        limit,
        max_per_section=max_per_section,
        similarity_threshold=similarity_threshold,
    )

app = FastAPI()

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    cache_folder="/models/cache"
)


def cross_rerank(query: str, chunks: list[str], limit: int) -> list[str]:
    if not chunks:
        return []

    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    # Отсекаем результаты ниже порога релевантности
    filtered = [
        chunk for chunk, score in ranked
        if float(score) >= MIN_CROSS_SCORE
    ]

    return filtered[:limit]


def unique_chunks(
    chunks: list[str],
    limit: int,
    similarity_threshold: float = 0.55,
) -> list[str]:
    result = []

    for chunk in chunks:
        if any(jaccard_similarity(chunk, existing) > similarity_threshold for existing in result):
            continue

        result.append(chunk)
        if len(result) >= limit:
            break

    return result


def best_answer_start(query: str, text: str) -> int:
    q_tokens = lexical_tokens(query)
    if len(q_tokens) < 2:
        return 0

    query_words = TOKEN_RE.findall(query)
    if len(query_words) >= 2:
        pattern = r"(?<!\w)" + r"\s+".join(
            re.escape(word) for word in query_words
        ) + r"(?!\w)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start()

    starts = [0]
    starts.extend(
        match.start(1)
        for match in re.finditer(r"(?:^|[.!?:;]\s+)([А-ЯЁA-Z][^.!?:;]{3,120})", text)
    )

    query_norm = normalize_for_match(query).rstrip("?!.")
    best_start = 0
    best_score = 0.0

    for start in sorted(set(starts)):
        preview = text[start:start + 240]
        preview_tokens = lexical_tokens(preview)
        if not preview_tokens:
            continue

        overlap = len(q_tokens & preview_tokens) / len(q_tokens)
        phrase = 1.0 if query_norm and query_norm in normalize_for_match(preview) else 0.0
        score = overlap + phrase

        if score > best_score:
            best_score = score
            best_start = start

    return best_start if best_score >= 1.0 else 0


def strip_answer_headings(query: str, text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text[best_answer_start(query, text):].strip()
    text = LEADING_SECTION_TITLE_RE.sub("", text).strip()

    query_prefix = normalize_for_match(query).rstrip("?!.")
    text_norm = normalize_for_match(text)
    query_clean = query.strip().rstrip("?!.")

    if query_prefix and text_norm.startswith(query_prefix):
        text = text[len(query_clean):].lstrip(" .:-–—")

    return text


def prepare_answer_chunks(query: str, chunks: list[str], limit: int) -> list[str]:
    cleaned = [
        chunk
        for chunk in (strip_answer_headings(query, item) for item in chunks)
        if chunk
    ]

    return unique_chunks(cleaned, limit, similarity_threshold=0.50)


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
    is_valid, _ = validate_query(q.question)
    if not is_valid:
        return UserOut(
                answer=None,
                answer_short=None,
                answer_long=None,
                short=[],
                long=[],
                llm_used=False,
            )

    short_rows, _ = search_table("documents_short", q.question, FETCH_K)
    long_rows, _ = search_table("documents_long", q.question, FETCH_K)

    short_candidates = rerank(
        q.question,
        short_rows,
        FETCH_K,
        max_per_section=1,
        similarity_threshold=0.62,
    )

    long_candidates = rerank(
        q.question,
        long_rows,
        FETCH_K,
        max_per_section=2,
        similarity_threshold=0.50,
    )

    short_ranked = cross_rerank(q.question, short_candidates, TOP_K_SHORT * 4)
    long_ranked = cross_rerank(q.question, long_candidates, TOP_K_LONG * 4)

    short = prepare_answer_chunks(q.question, short_ranked, TOP_K_SHORT)
    long = prepare_answer_chunks(q.question, long_ranked, TOP_K_LONG)

    context_chunks = unique_chunks(
        short + long,
        limit=TOP_K_SHORT + TOP_K_LONG,
        similarity_threshold=0.50,
    )

    answer_short, answer_long = generate_dual_answer(
    question=q.question,
    chunks=context_chunks,
    use_llm=q.use_llm,
    )

    llm_used = bool(answer_short or answer_long)

    return UserOut(
    answer=answer_short,
    answer_short=answer_short,
    answer_long=answer_long,
    short=short,
    long=long,
    llm_used=llm_used,
    )
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
    llm_used: bool = False


class TrainerCheckRequest(BaseModel):
    question_id: int
    answer: str
    use_llm: bool = False


class TrainerCheckResponse(BaseModel):
    status: str          # "Верно", "Неверно", "Верно частично"
    explanation: str
    llm_used: bool = False

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
def get_random_question(use_llm: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, question_text FROM questions ORDER BY RANDOM() LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Нет вопросов. Запустите bootstrapper.")

    question_text = row[1]

    final_question, llm_used = rephrase_trainer_question(
        question=question_text,
        use_llm=use_llm,
    )

    return TrainerQuestionResponse(
        id=row[0],
        question=final_question,
        llm_used=llm_used,
    )

@app.post("/api/trainer/check", response_model=TrainerCheckResponse)
def check_answer(req: TrainerCheckRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT question_text, answer_text, embedding FROM questions WHERE id = %s",
        (req.question_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Вопрос не найден")

    question_text, answer_text, embedding_ref = row

    user_emb = embed_query(req.answer)
    ref_emb = _to_list(embedding_ref)
    similarity = cosine_similarity(user_emb, ref_emb)

    answer_tokens = lexical_tokens(req.answer)
    reference_tokens = lexical_tokens(answer_text)

    common_tokens = answer_tokens & reference_tokens

    if reference_tokens:
        overlap = len(common_tokens) / len(reference_tokens)
    else:
        overlap = 0.0

    # Защита от бессмысленных ответов.
    MIN_MEANINGFUL_TOKENS = 2
    MIN_COMMON_TOKENS_FOR_PARTIAL = 1
    MIN_COMMON_TOKENS_FOR_CORRECT = 2

    is_too_short = len(answer_tokens) < MIN_MEANINGFUL_TOKENS
    has_no_domain_overlap = len(common_tokens) == 0

    # Итоговый score используем только если есть пересечение по терминам.
    # Иначе случайная embedding-близость не должна давать "частично верно".
    if common_tokens:
        final_score = 0.7 * similarity + 0.3 * overlap
    else:
        final_score = 0.0

    print(
        "[TRAINER CHECK]",
        f"similarity={similarity:.3f}",
        f"overlap={overlap:.3f}",
        f"common_tokens={sorted(common_tokens)}",
        f"final_score={final_score:.3f}",
    )

    if is_too_short or has_no_domain_overlap:
        status = "Неверно"
        explanation = answer_text
    elif (
        similarity >= 0.84 and len(common_tokens) >= MIN_COMMON_TOKENS_FOR_CORRECT
    ) or final_score >= 0.68:
        status = "Верно"
        explanation = "Ответ верный."
    elif (
        similarity >= 0.70 and len(common_tokens) >= MIN_COMMON_TOKENS_FOR_PARTIAL
    ) or final_score >= 0.50:
        status = "Верно частично"
        explanation = answer_text
    else:
        status = "Неверно"
        explanation = answer_text

    final_explanation, llm_used = rephrase_trainer_feedback(
        question=question_text,
        user_answer=req.answer,
        correct_answer=answer_text,
        status=status,
        explanation=explanation,
        use_llm=req.use_llm,
    )

    return TrainerCheckResponse(
        status=status,
        explanation=final_explanation,
        llm_used=llm_used,
    )