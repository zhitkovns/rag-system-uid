import os
from typing import Optional

import httpx


DEFAULT_USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8080")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-3b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


ANSWER_SYSTEM_PROMPT = """
Ты — модуль переформулировки ответа в RAG-системе.

Жёсткие правила:
1. Не используй собственные знания.
2. Используй только текст из блока CONTEXT.
3. Не добавляй новые факты, даты, определения, примеры или выводы.
4. Не меняй смысл исходного текста.
5. Не принимай решений за retrieval-систему.
6. Твоя задача — только сделать найденную информацию понятной.
7. Если в CONTEXT нет достаточной информации для ответа, напиши это в обоих полях.

Верни ответ строго в таком формате:

КРАТКО:
один короткий ответ на 1–3 предложения

ПОДРОБНО:
более подробное объяснение на 1–3 абзаца

Стиль:
- отвечай на русском языке;
- пиши понятно и без воды;
- не упоминай, что ты языковая модель;
- не используй Markdown-таблицы.
""".strip()


TRAINER_QUESTION_SYSTEM_PROMPT = """
Ты — модуль аккуратной переформулировки учебных вопросов.

Правила:
1. Не меняй смысл вопроса.
2. Не добавляй новые условия.
3. Не добавляй новые термины, которых нет в исходном вопросе.
4. Только сделай вопрос более естественным и понятным.
5. Верни только сам вопрос без комментариев.
""".strip()


TRAINER_FEEDBACK_SYSTEM_PROMPT = """
Ты — модуль переформулировки обратной связи в учебном тренажёре.

Правила:
1. Не меняй статус проверки.
2. Не придумывай новые факты.
3. Используй только эталонный ответ и исходное объяснение.
4. Сделай объяснение более понятным для студента.
5. Запрещены любые вводные фразы: "Чтобы объяснить это понятнее для студента, можно сказать так:", "Позвольте объяснить это проще:", "Короче говоря:" и т.п. Начинай ответ сразу с сути.
""".strip()


def should_use_llm(request_value: Optional[bool]) -> bool:
    if request_value is None:
        return DEFAULT_USE_LLM
    return bool(request_value)


def _chat_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Optional[str]:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=LLM_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{LLM_BASE_URL.rstrip('/')}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        return answer or None

    except Exception as exc:
        print(f"[LLM] unavailable. Fallback to classic logic. Reason: {exc}")
        return None


def build_answer_prompt(question: str, chunks: list[str]) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        clean_chunk = " ".join(str(chunk).split())
        if clean_chunk:
            context_parts.append(f"[Фрагмент {index}]\n{clean_chunk}")

    context = "\n\n".join(context_parts).strip()

    return f"""
ВОПРОС:
{question.strip()}

CONTEXT:
{context}

ЗАДАЧА:
Сформулируй связный ответ на вопрос, используя только CONTEXT.
Не добавляй ничего от себя.
Если информации недостаточно, явно скажи, что в найденных фрагментах нет достаточной информации.
""".strip()


def generate_answer(question: str, chunks: list[str], use_llm: Optional[bool]) -> Optional[str]:
    if not should_use_llm(use_llm):
        return None

    useful_chunks = [chunk for chunk in chunks if chunk and str(chunk).strip()]
    if not useful_chunks:
        return None

    return _chat_completion(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=build_answer_prompt(question, useful_chunks),
        max_tokens=LLM_MAX_TOKENS,
    )


def rephrase_trainer_question(question: str, use_llm: Optional[bool]) -> tuple[str, bool]:
    if not should_use_llm(use_llm):
        return question, False

    prompt = f"""
ИСХОДНЫЙ ВОПРОС:
{question.strip()}

ЗАДАЧА:
Переформулируй вопрос более понятно, но строго сохрани смысл.
Верни только вопрос.
""".strip()

    result = _chat_completion(
        system_prompt=TRAINER_QUESTION_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=128,
    )

    if not result:
        return question, False

    return result, True

def parse_dual_answer(raw_answer: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not raw_answer:
        return None, None

    text = raw_answer.strip()

    short_markers = ["КРАТКО:", "Кратко:", "КРАТКИЙ ОТВЕТ:", "Краткий ответ:"]
    long_markers = ["ПОДРОБНО:", "Подробно:", "ПОДРОБНЫЙ ОТВЕТ:", "Подробный ответ:"]

    short_pos = -1
    short_marker = ""
    for marker in short_markers:
        pos = text.find(marker)
        if pos != -1:
            short_pos = pos
            short_marker = marker
            break

    long_pos = -1
    long_marker = ""
    for marker in long_markers:
        pos = text.find(marker)
        if pos != -1:
            long_pos = pos
            long_marker = marker
            break

    if short_pos != -1 and long_pos != -1 and short_pos < long_pos:
        short_text = text[short_pos + len(short_marker):long_pos].strip()
        long_text = text[long_pos + len(long_marker):].strip()
        return short_text or None, long_text or None

    # fallback: если модель нарушила формат
    return text, None


def generate_dual_answer(
    question: str,
    chunks: list[str],
    use_llm: Optional[bool],
) -> tuple[Optional[str], Optional[str]]:
    raw = generate_answer(
        question=question,
        chunks=chunks,
        use_llm=use_llm,
    )
    return parse_dual_answer(raw)

def rephrase_trainer_feedback(
    question: str,
    user_answer: str,
    correct_answer: str,
    status: str,
    explanation: str,
    use_llm: Optional[bool],
) -> tuple[str, bool]:
    if not should_use_llm(use_llm):
        return explanation, False

    prompt = f"""
ВОПРОС:
{question.strip()}

ОТВЕТ СТУДЕНТА:
{user_answer.strip()}

СТАТУС ПРОВЕРКИ:
{status}

ЭТАЛОННЫЙ ОТВЕТ:
{correct_answer.strip()}

ИСХОДНОЕ ОБЪЯСНЕНИЕ:
{explanation.strip()}

ЗАДАЧА:
Сделай объяснение понятнее для студента.
Не меняй статус проверки.
Не добавляй фактов, которых нет в эталонном ответе.
КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать вводные фразы:
- "Чтобы объяснить это понятнее для студента, можно сказать так:"
- "Позвольте объяснить это проще:"
- "Короче говоря:"
- любые другие фразы, не несущие информации.
Начинай ответ сразу с объяснения без предисловий.
""".strip()

    result = _chat_completion(
        system_prompt=TRAINER_FEEDBACK_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=256,
    )

    if not result:
        return explanation, False

    # Очистка от нежелательных фраз (на случай, если модель их всё же добавит)
    forbidden = [
        "Чтобы объяснить это понятнее для студента, можно сказать так:",
        "Позвольте объяснить это проще:",
        "Короче говоря:",
    ]
    for phrase in forbidden:
        result = result.replace(phrase, "")

    result = result.strip()
    return result, True