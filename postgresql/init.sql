CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents_short (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);

-- CREATE INDEX IF NOT EXISTS documents_short_embedding_idx
-- ON documents_short                            Временно отключаем индекс для коротких документов, так как их может быть много и они могут часто обновляться
-- USING ivfflat (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS documents_long (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);

-- CREATE INDEX IF NOT EXISTS documents_long_embedding_idx
-- ON documents_long         Временно отключаем индекс для длинных документов, так как их может быть много и они могут часто обновляться
-- USING ivfflat (embedding vector_cosine_ops);

ALTER TABLE documents_short
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED;

ALTER TABLE documents_long
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED;

CREATE INDEX IF NOT EXISTS documents_short_search_idx
ON documents_short USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS documents_long_search_idx
ON documents_long USING GIN (search_vector);


-- Таблица для тренажера (вопросы-определения)
CREATE TABLE IF NOT EXISTS questions (
    id SERIAL PRIMARY KEY,
    question_text TEXT NOT NULL,      -- текст вопроса (например, "Что такое X?")
    answer_text TEXT NOT NULL,        -- эталонный ответ (предложение из учебника)
    embedding vector(768)             -- эмбеддинг эталона (passage:)
);

-- Индекс для быстрого поиска по эмбеддингу (по желанию, для масштабирования)
-- CREATE INDEX IF NOT EXISTS questions_embedding_idx
-- ON questions
-- USING ivfflat (embedding vector_cosine_ops);
