from __future__ import annotations

import hashlib
import re
from typing import Any


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[，。；：、,.!?:;\"'“”‘’（）()【】\[\]{}<>]+")


def normalize_for_fingerprint(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").strip().lower()
    if not s:
        return ""
    s = _WS_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def stable_fingerprint(text: str) -> str:
    norm = normalize_for_fingerprint(text)
    if not norm:
        return ""
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]


def dedup_group_id(*, content: str, chunk_type: str | None, metadata: dict[str, Any] | None = None) -> str:
    meta = metadata or {}
    ct = str(chunk_type or meta.get("chunk_type") or "").strip().lower()
    if ct == "chart":
        cid = str(meta.get("chart_id") or "").strip()
        if cid:
            return f"chart:{cid}"
    fp = stable_fingerprint(content)
    if not fp:
        return ""
    return f"{ct}:{fp}" if ct else fp
