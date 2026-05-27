"""Cross-encoder rerank dispatcher (local CrossEncoder or HF TEI endpoint)."""
from __future__ import annotations

from app.config import settings
from app.services.embeddings import get_device

_model = None  # CrossEncoder (local mode only)


def get_reranker():
    """Local CrossEncoder (lazy). Not used in huggingface mode."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(settings.RERANKER_MODEL, device=get_device())
    return _model


def _local_scores(query: str, texts: list[str]) -> list[float]:
    pairs = [[query, t] for t in texts]
    return [float(s) for s in get_reranker().predict(pairs, show_progress_bar=False)]


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Each candidate must have a "text" field. Returns sorted copy with rerank_score."""
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    if settings.use_hf:
        from app.services.providers import hf_reranker

        scores = hf_reranker.rerank_scores(query, texts)
    else:
        scores = _local_scores(query, texts)

    enriched = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["rerank_score"] = float(s)
        enriched.append(item)
    enriched.sort(key=lambda c: c["rerank_score"], reverse=True)
    return enriched


def warmup() -> None:
    if settings.use_hf:
        from app.services.providers import hf_reranker

        hf_reranker.warmup()
        print(f"[{settings.APP_NAME}] reranker via HF endpoint")
    else:
        get_reranker()
        print(f"[{settings.APP_NAME}] reranker loaded: {settings.RERANKER_MODEL}")
