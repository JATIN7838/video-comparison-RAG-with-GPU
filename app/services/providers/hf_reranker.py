"""Cross-encoder reranking via a Hugging Face TEI `/rerank` endpoint."""
from __future__ import annotations

from app.config import settings
from app.services.providers._http import post_json


def _require_url() -> str:
    if not settings.HF_RERANKER_ENDPOINT_URL:
        raise RuntimeError("HF_RERANKER_ENDPOINT_URL is not set")
    return settings.HF_RERANKER_ENDPOINT_URL


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Return one score per input text, aligned to input order."""
    if not texts:
        return []

    url = _require_url()
    # TEI rerank route: {"query": ..., "texts": [...], "raw_scores": true}
    #   -> [{"index": i, "score": s}, ...]  (raw logits, like CrossEncoder.predict)
    data = post_json(url, {"query": query, "texts": texts, "raw_scores": True})

    scores = [0.0] * len(texts)
    for item in data:
        scores[item["index"]] = float(item["score"])
    return scores


def warmup() -> None:
    rerank_scores("warmup", ["warmup"])
