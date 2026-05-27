"""Dense embeddings via a Hugging Face TEI `/embed` endpoint."""
from __future__ import annotations

import numpy as np

from app.config import settings
from app.services.providers._http import post_json


def _require_url() -> str:
    if not settings.HF_EMBEDDING_ENDPOINT_URL:
        raise RuntimeError("HF_EMBEDDING_ENDPOINT_URL is not set")
    return settings.HF_EMBEDDING_ENDPOINT_URL


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

    url = _require_url()
    vectors: list[list[float]] = []
    batch = settings.EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        # TEI embeddings route: {"inputs": [...]} -> [[...], ...]
        data = post_json(url, {"inputs": chunk})
        vectors.extend(data)

    return np.asarray(vectors, dtype=np.float32)


def warmup() -> None:
    # Wake a scale-to-zero endpoint and verify connectivity.
    embed_texts(["warmup"])
