import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _prepare_texts(texts: list[str]) -> list[str]:
    return [t if t and t.strip() else " " for t in texts]


def _embed_local_hf(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_seq_length: int | None = None,
    torch_dtype: str = "",
    dimensions: int = 0,
) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("No GPU detected, running embeddings on CPU - will be slow")

    model_kwargs = {}
    if torch_dtype:
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

    st_kwargs = {}
    if dimensions:
        st_kwargs["truncate_dim"] = dimensions

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs or None,
        **st_kwargs,
    )

    if max_seq_length:
        model.max_seq_length = max_seq_length

    if getattr(model, "default_prompt_name", None):
        warnings.warn(
            f"Model has default_prompt_name='{model.default_prompt_name}', disabling it "
            "to stay consistent with the remote API"
        )
        model.default_prompt_name = None

    print(
        f"Local embedder: {model_name} device={device} dtype={torch_dtype or 'default'} "
        f"max_seq_length={model.max_seq_length} batch_size={batch_size}"
    )

    embeddings = model.encode(
        _prepare_texts(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def _embed_openai_compatible(
    texts: list[str],
    model_name: str,
    api_key: str,
    base_url: str | None,
    batch_size: int = 512,
    max_retries: int = 3,
    dimensions: int = 0,
    max_workers: int = 1,
) -> np.ndarray:
    import time

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    safe_texts = _prepare_texts(texts)
    batches = [safe_texts[i : i + batch_size] for i in range(0, len(safe_texts), batch_size)]

    def embed_batch(batch: list[str]) -> list[list[float]]:
        extra = {"dimensions": dimensions} if dimensions else {}
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model_name, input=batch, **extra)
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Embedding API failed after {max_retries} retries: {e}")
                wait = 2**attempt
                warnings.warn(f"Embedding API error (attempt {attempt + 1}): {e}. Retrying in {wait}s")
                time.sleep(wait)
        return []

    if max_workers > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(embed_batch, batches))
    else:
        results = [embed_batch(b) for b in batches]

    all_embeddings = [emb for batch_result in results for emb in batch_result]
    return np.array(all_embeddings, dtype=np.float32)


def generate_embeddings(
    texts: list[str],
    provider: str,
    model_name: str,
    batch_size: int = 512,
    api_key: str = "",
    api_base_url: str = "",
    max_seq_length: int | None = None,
    torch_dtype: str = "",
    dimensions: int = 0,
    max_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    t0 = time.time()

    if provider == "local_hf":
        embeddings = _embed_local_hf(
            texts,
            model_name,
            batch_size,
            max_seq_length=max_seq_length,
            torch_dtype=torch_dtype,
            dimensions=dimensions,
        )
    elif provider in ("openai", "gemini", "qwen"):
        embeddings = _embed_openai_compatible(
            texts,
            model_name,
            api_key,
            api_base_url or None,
            batch_size,
            dimensions=dimensions,
            max_workers=max_workers,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    elapsed = time.time() - t0
    meta = {
        "provider": provider,
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_docs": len(texts),
        "max_seq_length": max_seq_length,
        "requested_dimensions": dimensions or None,
        "inference_time_seconds": round(elapsed, 2),
        "throughput_docs_per_sec": round(len(texts) / elapsed, 1),
    }
    return embeddings, meta
