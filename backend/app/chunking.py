import re
from typing import List, Tuple


HEADING_RE = re.compile(
    r"(?m)^\s*(\d+(?:\.\d+)+\.?\s+.{3,140})\s*$"
)

SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9])"
)


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")

    # Склейка переносов внутри слов: "пред-\nставление" -> "представление"
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Мягкие переносы строк внутри абзацев
    text = re.sub(r"(?<![.!?:;])\n(?!\s*\d+(?:\.\d+)+)", " ", text)

    # Нормализация пробелов
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_sections(text: str) -> List[Tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        return [("", text)]

    sections: List[Tuple[str, str]] = []

    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if body:
            sections.append((title, body))

    return sections or [("", text)]


def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = SENTENCE_SPLIT_RE.split(text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) >= 10
    ]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def with_title(title: str, body: str) -> str:
    body = body.strip()
    if not title:
        return body

    if body.startswith(title):
        return body

    return f"{title}. {body}"


def split_oversized_sentence(sentence: str, tokenizer, max_tokens: int) -> List[str]:
    words = sentence.split()
    chunks: List[str] = []
    buf: List[str] = []

    for word in words:
        candidate = " ".join(buf + [word])

        if buf and token_len(tokenizer, candidate) > max_tokens:
            chunks.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def chunk_text(
    text: str,
    model,
    max_tokens: int,
    overlap_sentences: int = 1,
) -> List[str]:
    tokenizer = model.tokenizer
    text = normalize_text(text)

    chunks: List[str] = []

    for title, body in split_sections(text):
        sentences = split_sentences(body)
        buf: List[str] = []

        for sent in sentences:
            titled_sent = with_title(title, sent)

            # Если одно предложение слишком длинное — режем его по словам,
            # но не через tokenizer.decode(), чтобы не получить <unk> и обрубки.
            if token_len(tokenizer, titled_sent) > max_tokens:
                if buf:
                    chunks.append(with_title(title, " ".join(buf)))
                    buf = []

                parts = split_oversized_sentence(sent, tokenizer, max_tokens)
                chunks.extend(with_title(title, part) for part in parts)
                continue

            candidate = " ".join(buf + [sent])
            titled_candidate = with_title(title, candidate)

            if buf and token_len(tokenizer, titled_candidate) > max_tokens:
                chunks.append(with_title(title, " ".join(buf)))
                buf = buf[-overlap_sentences:] if overlap_sentences else []

            buf.append(sent)

        if buf:
            chunks.append(with_title(title, " ".join(buf)))

    # Дедупликация и финальная чистка
    result: List[str] = []
    seen = set()

    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        key = chunk.lower()

        if len(chunk) < 40:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(chunk)

    return result