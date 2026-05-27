"""Shared HTTP helper for Hugging Face Inference Endpoints (TEI).

Uses `requests` (already a dependency) since the FastAPI routes are sync. Adds
auth headers and retry/backoff for cold starts — scale-to-zero endpoints return
503 while a replica spins up.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from app.config import settings

_RETRY_STATUS = {429, 502, 503, 504}


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.HF_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"
    return headers


def post_json(url: str, payload: dict[str, Any]) -> Any:
    last_exc: Exception | None = None
    for attempt in range(settings.HF_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=_headers(), timeout=settings.HF_TIMEOUT
            )
            if resp.status_code in _RETRY_STATUS:
                raise requests.HTTPError(f"{resp.status_code} {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, requests.HTTPError) as e:
            last_exc = e
            if attempt < settings.HF_MAX_RETRIES:
                backoff = min(2.0 * (2**attempt), 30.0)
                print(
                    f"[hf] {url} attempt {attempt + 1} failed ({e}); "
                    f"retrying in {backoff:.0f}s"
                )
                time.sleep(backoff)
    raise RuntimeError(f"HF endpoint call failed after retries: {url}: {last_exc}")
