from __future__ import annotations

import re
from typing import Iterable


_ANCHOR_RE = re.compile(r"\[Chart:\s*([^\]]+)\]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _split_paragraphs(md: str) -> list[str]:
    text = (md or "").replace("\r\n", "\n")
    parts = re.split(r"\n{2,}", text)
    return [p.strip("\n") for p in parts if p.strip("\n").strip()]


def _anchors_in_text(md: str) -> list[str]:
    out: list[str] = []
    for m in _ANCHOR_RE.finditer(md or ""):
        cid = _norm(m.group(1))
        if cid:
            out.append(cid)
    return out
 
 
def strip_chart_anchors(md: str) -> str:
    s = (md or "").replace("\r\n", "\n")
    if not s.strip():
        return s
    s = _ANCHOR_RE.sub("", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip() + "\n"
 
 
def place_chart_anchors(
    md: str,
    *,
    chart_ids: Iterable[str],
    hint_by_chart_id: dict[str, str] | None = None,
    min_paragraph_index: int = 1,
) -> str:
    text = strip_chart_anchors(md or "")
    ids = [str(x or "").strip() for x in (chart_ids or []) if str(x or "").strip()]
    if not ids:
        return text
    paras = _split_paragraphs(text)
    if not paras:
        return (text.rstrip() + "\n\n" + "\n\n".join([f"[Chart: {cid}]" for cid in ids]) + "\n").strip() + "\n"
 
    hints = hint_by_chart_id or {}
    updated = list(paras)
 
    def _is_heading(p: str) -> bool:
        return bool(re.match(r"^\s*#{1,6}\s+\S+", (p or "").strip()))
 
    def _score(cid: str, p: str) -> float:
        hint = _norm(hints.get(cid, ""))
        body = _norm(p)
        if not body:
            return 0.0
        score = 0.0
        if hint:
            for tok in re.split(r"[\s，。；：:、,()（）\[\]“”\"'【】]+", hint):
                t = tok.strip()
                if len(t) >= 2 and t in body:
                    score += 1.0
            if hint and hint in body:
                score += 1.5
        if len(body) >= 120:
            score += 0.2
        return score
 
    for cid in ids:
        best_idx = 0
        best_score = -1.0
        for i, p in enumerate(updated):
            if _is_heading(p):
                continue
            sc = _score(cid, p)
            if sc > best_score:
                best_score = sc
                best_idx = i
        if best_idx < int(min_paragraph_index or 0) and best_score < 1.0:
            candidates = [i for i, p in enumerate(updated) if not _is_heading(p)]
            if candidates:
                later = [i for i in candidates if i >= int(min_paragraph_index or 0)]
                best_idx = later[0] if later else candidates[-1]
        updated[best_idx] = updated[best_idx].rstrip() + f"\n\n[Chart: {cid}]"
 
    return "\n\n".join(updated).strip() + "\n"


def ensure_chart_anchors(
    md: str,
    *,
    required_chart_ids: Iterable[str],
    hint_by_chart_id: dict[str, str] | None = None,
) -> str:
    text = md or ""
    required = [str(x or "").strip() for x in (required_chart_ids or []) if str(x or "").strip()]
    if not required:
        return text
    existing = set(_anchors_in_text(text))
    missing = [cid for cid in required if cid not in existing]
    if not missing:
        return text

    paras = _split_paragraphs(text)
    if not paras:
        return (text.rstrip() + "\n\n" + "\n".join([f"[Chart: {cid}]" for cid in missing]) + "\n").strip() + "\n"

    hints = hint_by_chart_id or {}
    updated = list(paras)
    for cid in missing:
        hint = _norm(hints.get(cid, ""))
        target_idx = 0
        if hint:
            best = 0.0
            for i, p in enumerate(updated):
                p_norm = _norm(p)
                if not p_norm:
                    continue
                score = 0.0
                for tok in re.split(r"[\s，。；：:、,()（）\[\]“”\"'【】]+", hint):
                    t = tok.strip()
                    if len(t) >= 2 and t in p_norm:
                        score += 1.0
                if score > best:
                    best = score
                    target_idx = i
        updated[target_idx] = updated[target_idx].rstrip() + f"\n[Chart: {cid}]"

    return "\n\n".join(updated).strip() + "\n"


def reduce_tail_pileup(
    md: str,
    *,
    required_chart_ids: Iterable[str],
    hint_by_chart_id: dict[str, str] | None = None,
    tail_window_lines: int = 20,
) -> str:
    text = (md or "").replace("\r\n", "\n")
    required = [str(x or "").strip() for x in (required_chart_ids or []) if str(x or "").strip()]
    if not required:
        return text
    lines = text.split("\n")
    tail = "\n".join(lines[-max(1, int(tail_window_lines or 0)) :])
    tail_ids = set(_anchors_in_text(tail))
    if not tail_ids:
        return ensure_chart_anchors(text, required_chart_ids=required, hint_by_chart_id=hint_by_chart_id)
    if not all(cid in tail_ids or cid in _anchors_in_text(text) for cid in required):
        return ensure_chart_anchors(text, required_chart_ids=required, hint_by_chart_id=hint_by_chart_id)
    if len(tail_ids) < max(2, len(required) // 2):
        return text

    kept_lines: list[str] = []
    for ln in lines:
        if _ANCHOR_RE.search(ln):
            continue
        kept_lines.append(ln)
    stripped = "\n".join(kept_lines).strip()
    return ensure_chart_anchors(stripped, required_chart_ids=required, hint_by_chart_id=hint_by_chart_id)
