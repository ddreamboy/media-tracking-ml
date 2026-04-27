import time
import warnings
from typing import Optional

import numpy as np


def _embed_local_hf(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("No GPU detected, running embeddings on CPU - will be slow")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def _embed_openai_compatible(
    texts: list[str],
    model_name: str,
    api_key: str,
    base_url: Optional[str],
    batch_size: int = 512,
    max_retries: int = 3,
) -> np.ndarray:
    import time

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model_name, input=batch)
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Embedding API failed after {max_retries} retries: {e}"
                    )
                wait = 2**attempt
                warnings.warn(
                    f"Embedding API error (attempt {attempt + 1}): {e}. Retrying in {wait}s"
                )
                time.sleep(wait)

    return np.array(all_embeddings, dtype=np.float32)


def generate_embeddings(
    texts: list[str],
    provider: str,
    model_name: str,
    batch_size: int = 512,
    api_key: str = "",
    api_base_url: str = "",
) -> tuple[np.ndarray, dict]:
    t0 = time.time()

    if provider == "local_hf":
        embeddings = _embed_local_hf(texts, model_name, batch_size)
    elif provider in ("openai", "gemini", "qwen"):
        embeddings = _embed_openai_compatible(
            texts, model_name, api_key, api_base_url or None, batch_size
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    elapsed = time.time() - t0
    meta = {
        "provider": provider,
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_docs": len(texts),
        "inference_time_seconds": round(elapsed, 2),
        "throughput_docs_per_sec": round(len(texts) / elapsed, 1),
    }
    return embeddings, meta
