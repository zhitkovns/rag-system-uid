import os
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_REPO = os.getenv("LLM_MODEL_REPO", "bartowski/Qwen2.5-3B-Instruct-GGUF")
MODEL_FILE = os.getenv("LLM_MODEL_FILE", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
MODELS_DIR = Path(os.getenv("LLM_MODELS_DIR", "/models"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)

target_path = MODELS_DIR / MODEL_FILE

if target_path.exists() and target_path.stat().st_size > 0:
    print(f"[LLM downloader] Model already exists: {target_path}")
else:
    print(f"[LLM downloader] Downloading {MODEL_REPO}/{MODEL_FILE}")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"[LLM downloader] Downloaded to: {downloaded_path}")

if not target_path.exists():
    raise RuntimeError(f"Model was not found after download: {target_path}")

print(f"[LLM downloader] Ready: {target_path}")