"""Dense embeddings dispatcher.
Selects the backend from settings.MODEL_BACKEND:
  - "local"       -> SentenceTransformer on GPU/CPU (torch imported lazily)
  - "huggingface" -> HF TEI endpoint (no torch import at all)
"""
from __future__ import annotations

import numpy as np

from app.config import settings

_model = None  # SentenceTransformer (local mode only)
_device: str | None = None


def get_device() -> str:
    global _device
    if _device is None:
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{settings.APP_NAME}] device: {_device}")
    return _device


def get_embedding_model():
    """Local SentenceTransformer (lazy). Not used in huggingface mode."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=get_device())
    return _model


def _local_embed(texts: list[str]) -> np.ndarray:
    return get_embedding_model().encode(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    if settings.use_hf:
        from app.services.providers import hf_embeddings

        return hf_embeddings.embed_texts(texts)
    return _local_embed(texts)


def embedding_dimension() -> int:
    if settings.use_hf:
        return settings.EMBEDDING_DIM
    return get_embedding_model().get_embedding_dimension()


def warmup() -> int:
    """Load/ping the embedding backend; returns the vector dimension."""
    if settings.use_hf:
        from app.services.providers import hf_embeddings

        hf_embeddings.warmup()
        print(
            f"[{settings.APP_NAME}] embeddings via HF endpoint "
            f"(dim={settings.EMBEDDING_DIM})"
        )
        return settings.EMBEDDING_DIM

    dim = embedding_dimension()
    print(f"[{settings.APP_NAME}] embedding model loaded (dim={dim})")
    return dim
