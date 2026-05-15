import re
import os
import glob
import logging
import psycopg2
from sentence_transformers import SentenceTransformer

log = logging.getLogger("question_generator")

# Расширенный набор шаблонов
PATTERNS = [
    # X называется Y
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+называется\s+([^.!?]+[.!?])', re.MULTILINE),
    # X — Y (тире)
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+[-–—]\s+([^.!?]+[.!?])', re.MULTILINE),
    # X – это Y (тире + "это")
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+[-–—]\s*это\s+([^.!?]+[.!?])', re.MULTILINE),
    # Под X понимается Y
    re.compile(r'Под\s+([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+понимается\s+([^.!?]+[.!?])', re.MULTILINE),
    # X называют Y
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+называют\s+([^.!?]+[.!?])', re.MULTILINE),
    # X определяется (как) Y
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+определя(?:ется|ют)\s+(?:как\s+)?([^.!?]+[.!?])', re.MULTILINE),
    # X есть Y
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+есть\s+([^.!?]+[.!?])', re.MULTILINE),
    # Под X понимают Y
    re.compile(r'Под\s+([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s+понимают\s+([^.!?]+[.!?])', re.MULTILINE),
    # X: Y (двоеточие)
    re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё\s\-]{1,60}?)\s*:\s*([^.!?]+[.!?])', re.MULTILINE),
]

# Запрещённые термины – только одиночные слова-паразиты
FORBIDDEN_TERMS = {
    'если', 'поскольку', 'так как', 'пусть', 'тогда', 'далее',
    'следовательно', 'действительно', 'заметим', 'рассмотрим',
    'вход', 'выход', 'алгоритм', 'метод', 'функция', 'дерево', 'граф',
    'этот', 'эта', 'это', 'эти', 'такой', 'такая', 'такое', 'такие',
    'весь', 'вся', 'всё', 'все', 'один', 'одна', 'одно', 'одни',
    'любой', 'любая', 'любое', 'любые',
}

def clean_term(term: str) -> str:
    """Очищает термин от номеров, лишних символов."""
    term = term.strip()
    # Убираем номера в начале
    term = re.sub(r'^\d+(\.\d+)*[\.\)]?\s*', '', term)
    # Убираем кавычки
    term = term.strip('"\'')
    # Убираем знаки препинания в конце
    term = re.sub(r'[;:,.!?]$', '', term)
    # Ограничиваем длину – до 6 слов
    words = term.split()
    if len(words) > 6:
        term = ' '.join(words[:6])
    return term.strip()

def clean_definition(definition: str) -> str:
    """Очищает определение и делает первую букву заглавной."""
    definition = definition.strip()
    # Убираем номера
    definition = re.sub(r'^\d+(\.\d+)*[\.\)]?\s*', '', definition)
    # Убираем лишние пробелы
    definition = re.sub(r'\s+', ' ', definition)
    # Убираем начальные символы
    definition = re.sub(r'^[-–—:;]\s*', '', definition)
    if definition and definition[0].islower():
        definition = definition[0].upper() + definition[1:]
    return definition.strip()

def is_valid_term(term: str) -> bool:
    """Проверяет, что термин осмысленный."""
    t = term.lower().strip()
    if len(t) < 3 or len(t) > 80:
        return False
    # Не должен быть только из запрещённых слов
    if t in FORBIDDEN_TERMS:
        return False
    # Хотя бы одна буква (русская или латинская)
    if not re.search(r'[а-яёa-z]', t):
        return False
    return True

def extract_definitions(text: str):
    definitions = []
    for pattern in PATTERNS:
        for match in pattern.finditer(text):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            term = clean_term(term)
            definition = clean_definition(definition)

            if not is_valid_term(term):
                continue
            if len(definition) < 5 or len(definition) > 600:
                continue

            definitions.append((term, definition))

    # Удаляем дубликаты по термину (без учёта регистра)
    seen = set()
    unique = []
    for term, defn in definitions:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique.append((term, defn))
    return unique

def generate_question(term: str) -> str:
    """Формирует вопрос."""
    if term.isupper():
        return f"Что такое {term}?"
    else:
        return f"Что такое {term.lower()}?"

def clear_questions(conn):
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE questions")
    conn.commit()
    cur.close()

def question_exists(conn) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM questions")
    count = cur.fetchone()[0]
    cur.close()
    return count > 0

def generate_and_store_questions(txt_path: str, model, db_url: str):
    log.info("Начинаем генерацию вопросов")
    conn = psycopg2.connect(db_url)
    txt_files = glob.glob(os.path.join(txt_path, "*.txt"))
    if not txt_files:
        log.warning("Нет .txt файлов для генерации вопросов")
        conn.close()
        return

    all_definitions = []
    for file in txt_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        defs = extract_definitions(text)
        all_definitions.extend(defs)
        log.info(f"{file}: найдено определений {len(defs)}")

    log.info(f"Всего уникальных определений: {len(all_definitions)}")

    if not all_definitions:
        log.warning("Не найдено ни одного определения")
        conn.close()
        return

    questions_data = []
    for term, answer in all_definitions:
        question_text = generate_question(term)
        emb = model.encode([f"passage: {answer}"], normalize_embeddings=True)[0].tolist()
        questions_data.append((question_text, answer, emb))

    cur = conn.cursor()
    for q_text, a_text, emb in questions_data:
        cur.execute(
            "INSERT INTO questions (question_text, answer_text, embedding) VALUES (%s, %s, %s)",
            (q_text, a_text, emb)
        )
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Сохранено {len(questions_data)} вопросов")

# import re
# import os
# import glob
# import logging
# import psycopg2
# from sentence_transformers import SentenceTransformer
#
# log = logging.getLogger("question_generator")
#
# # Паттерны для поиска определений – расширенный набор
# PATTERNS = [
#     # X называется Y (X – 1-5 слов, может начинаться с маленькой буквы)
#     re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+называется\s+([^.!?]+[.!?])',
#                re.MULTILINE),
#     # X — Y (тире) – теперь допускаем строчную в термине
#     re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+[-–—]\s+([^.!?]+[.!?])',
#                re.MULTILINE),
#     # X – это Y
#     re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+[-–—]?\s*это\s+([^.!?]+[.!?])',
#                re.MULTILINE),
#     # Под X понимается Y
#     re.compile(r'Под\s+([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+понимается\s+([^.!?]+[.!?])',
#                re.MULTILINE),
#     # X называют Y
#     re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+называют\s+([^.!?]+[.!?])',
#                re.MULTILINE),
#     # X определяется (как) Y
#     re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+определя(?:ется|ют)\s+(?:как\s+)?([^.!?]+[.!?])',
#                re.MULTILINE),
#     # X есть Y
#     re.compile(r'(?<!\w)([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+есть\s+([^.!?]+[.!?])',
#                re.MULTILINE),
#     # Под X понимают Y
#     re.compile(r'Под\s+([А-ЯЁа-яё][А-ЯЁа-яё]+(?:\s+[А-ЯЁа-яё]+){0,4})\s+понимают\s+([^.!?]+[.!?])',
#                re.MULTILINE),
# ]
#
# # Слова, которые не могут быть терминами
# FORBIDDEN_TERMS = {
#     'если', 'поскольку', 'так как', 'имеем', 'пусть', 'тогда', 'далее',
#     'следовательно', 'от противного', 'действительно', 'заметим', 'рассмотрим',
#     'вход', 'выход', 'алгоритм', 'при этом', 'для', 'который', 'также',
#     'выдвигая', 'построение', 'создание', 'функций', 'дерево', 'граф',
#     'этот', 'эта', 'это', 'эти', 'такой', 'такая', 'такое', 'такие',
#     'весь', 'вся', 'всё', 'все', 'один', 'одна', 'одно', 'одни',
#     'адельсон', 'вельский', 'ландис', 'метод', 'уровень', 'добавляемое'
# }
#
#
# def clean_term(term: str) -> str:
#     """Очищает термин от номеров, лишних символов."""
#     term = term.strip()
#     # Убираем номера в начале
#     term = re.sub(r'^\d+(\.\d+)*[\.\)]?\s*', '', term)
#     # Убираем кавычки
#     term = term.strip('"\'')
#     # Убираем знаки препинания в конце
#     term = re.sub(r'[;:,.!?]$', '', term)
#     # Ограничиваем длину – теперь до 5 слов
#     words = term.split()
#     if len(words) > 5:
#         term = ' '.join(words[:5])
#     return term
#
#
# def clean_definition(definition: str) -> str:
#     """Очищает определение и делает первую букву заглавной."""
#     definition = definition.strip()
#     # Убираем номера
#     definition = re.sub(r'^\d+(\.\d+)*[\.\)]?\s*', '', definition)
#     # Убираем лишние пробелы
#     definition = re.sub(r'\s+', ' ', definition)
#     # Убираем начальные символы
#     definition = re.sub(r'^[-–—:;]\s*', '', definition)
#     # (черный список начал больше не применяем)
#     # Делаем первую букву заглавной
#     if definition and definition[0].islower():
#         definition = definition[0].upper() + definition[1:]
#     return definition
#
#
# def extract_definitions(text: str):
#     definitions = []
#     for pattern in PATTERNS:
#         for match in pattern.finditer(text):
#             term = match.group(1).strip()
#             definition = match.group(2).strip()
#             term = clean_term(term)
#             definition = clean_definition(definition)
#             # Фильтры
#             if len(term) < 3 or len(term) > 60:            # было 30
#                 continue
#             if term.lower() in FORBIDDEN_TERMS:
#                 continue
#             if any(term.lower().startswith(bad) for bad in FORBIDDEN_TERMS):
#                 continue
#             if not re.search(r'[А-ЯЁа-яё]', term):
#                 continue
#             if len(definition) < 10 or len(definition) > 500:   # было 20 и 400
#                 continue
#             # Убран фильтр на математические символы – он отсекал слишком много определений
#             definitions.append((term, definition))
#
#     # Удаляем дубликаты по термину (без учёта регистра)
#     seen = set()
#     unique = []
#     for term, defn in definitions:
#         key = term.lower()
#         if key not in seen:
#             seen.add(key)
#             unique.append((term, defn))
#     return unique
#
#
# def generate_question(term: str) -> str:
#     """Формирует вопрос с маленькой буквы (кроме аббревиатур)."""
#     if term.isupper():
#         return f"Что такое {term}?"
#     else:
#         return f"Что такое {term.lower()}?"
#
#
# def clear_questions(conn):
#     cur = conn.cursor()
#     cur.execute("TRUNCATE TABLE questions")
#     conn.commit()
#     cur.close()
#
#
# def question_exists(conn) -> bool:
#     cur = conn.cursor()
#     cur.execute("SELECT count(*) FROM questions")
#     count = cur.fetchone()[0]
#     cur.close()
#     return count > 0
#
#
# def generate_and_store_questions(txt_path: str, model, db_url: str):
#     log.info("Начинаем генерацию вопросов")
#     conn = psycopg2.connect(db_url)
#     txt_files = glob.glob(os.path.join(txt_path, "*.txt"))
#     if not txt_files:
#         log.warning("Нет .txt файлов для генерации вопросов")
#         conn.close()
#         return
#
#     all_definitions = []
#     for file in txt_files:
#         with open(file, "r", encoding="utf-8") as f:
#             text = f.read()
#         defs = extract_definitions(text)
#         all_definitions.extend(defs)
#         log.info(f"{file}: найдено определений {len(defs)}")
#
#     log.info(f"Всего уникальных определений: {len(all_definitions)}")
#
#     if not all_definitions:
#         log.warning("Не найдено ни одного определения")
#         conn.close()
#         return
#
#     questions_data = []
#     for term, answer in all_definitions:
#         question_text = generate_question(term)
#         emb = model.encode([f"passage: {answer}"], normalize_embeddings=True)[0].tolist()
#         questions_data.append((question_text, answer, emb))
#
#     cur = conn.cursor()
#     for q_text, a_text, emb in questions_data:
#         cur.execute(
#             "INSERT INTO questions (question_text, answer_text, embedding) VALUES (%s, %s, %s)",
#             (q_text, a_text, emb)
#         )
#     conn.commit()
#     cur.close()
#     conn.close()
#
#     # for i in range(100000):
#     #     print(questions_data)
#
#     log.info(f"Сохранено {len(questions_data)} вопросов")
