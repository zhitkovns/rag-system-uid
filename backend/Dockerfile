FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/
ENV OMP_NUM_THREADS=4


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    poppler-utils \
    curl \
    wget \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


# CPU-only torch like in docling docs
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


COPY app ./app

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]