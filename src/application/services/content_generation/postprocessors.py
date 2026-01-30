from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _norm_text(s: str) -> str:
    s2 = (s or "").replace("\r\n", "\n").strip()
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def _split_blocks(md: str) -> list[str]:
    src = (md or "").replace("\r\n", "\n").strip()
    if not src:
        return []
    parts = re.split(r"\n{2,}", src)
    return [p.strip("\n") for p in parts if p.strip("\n").strip()]


def _join_blocks(blocks: list[str]) -> str:
    out = "\n\n".join([b.rstrip() for b in blocks if (b or "").strip()])
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _looks_like_heading(block: str) -> bool:
    s = (block or "").strip()
    if not s:
        return True
    if re.match(r"^#{1,6}\s+\S+", s):
        return True
    return False


def _looks_like_pure_outline_item(block: str) -> bool:
    lines = (block or "").replace("\r\n", "\n").split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return True
    if len(non_empty) == 1:
        s = non_empty[0].lstrip(" ")
        return s.startswith("- ") or re.match(r"^\d+\.\s+\S+", s) is not None
    return False


@dataclass
class DedupStats:
    removed_within_section: int = 0
    removed_across_sections: int = 0


class SemanticDeduplicator:
    def __init__(
        self,
        *,
        embedding_model: Any,
        similarity_threshold: float = 0.92,
        min_block_chars: int = 80,
        max_compare: int = 400,
    ):
        self._embedding_model = embedding_model
        self._threshold = float(similarity_threshold)
        self._min_chars = int(min_block_chars)
        self._max_compare = int(max_compare)

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        m = self._embedding_model

        def _do() -> list[list[float]]:
            if hasattr(m, "embed_documents"):
                return [list(x) for x in m.embed_documents(texts)]
            if hasattr(m, "get_text_embedding_batch"):
                return [list(x) for x in m.get_text_embedding_batch(texts)]
            if hasattr(m, "get_text_embedding"):
                return [list(m.get_text_embedding(t)) for t in texts]
            if hasattr(m, "embed_query"):
                return [list(m.embed_query(t)) for t in texts]
            raise RuntimeError("embedding_model missing embed method")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do)

    async def dedup_markdown(
        self,
        md: str,
        *,
        global_kept_embeddings: list[list[float]] | None = None,
    ) -> tuple[str, DedupStats, list[list[float]]]:
        blocks = _split_blocks(md)
        if not blocks:
            return (md or "").strip(), DedupStats(), global_kept_embeddings or []

        global_kept_embeddings = global_kept_embeddings or []
        kept_blocks: list[str] = []
        kept_norms: list[str] = []
        kept_embeds: list[list[float]] = []
        stats = DedupStats()

        cand_texts: list[str] = []
        cand_idx: list[int] = []
        for i, b in enumerate(blocks):
            bn = _norm_text(b)
            if len(bn) < self._min_chars:
                continue
            if _looks_like_heading(bn) or _looks_like_pure_outline_item(bn):
                continue
            cand_texts.append(bn)
            cand_idx.append(i)

        cand_embeds: dict[int, list[float]] = {}
        if cand_texts:
            embs = await self._embed_texts(cand_texts)
            for i, emb in zip(cand_idx, embs, strict=False):
                cand_embeds[i] = emb

        for i, b in enumerate(blocks):
            bn = _norm_text(b)
            if not bn:
                continue
            if bn in kept_norms:
                stats.removed_within_section += 1
                continue

            emb = cand_embeds.get(i)
            is_dup_within = False
            if emb is not None and kept_embeds:
                start = max(0, len(kept_embeds) - self._max_compare)
                for prev in kept_embeds[start:]:
                    if _cosine_similarity(prev, emb) >= self._threshold:
                        is_dup_within = True
                        break
            if is_dup_within:
                stats.removed_within_section += 1
                continue

            is_dup_global = False
            if emb is not None and global_kept_embeddings:
                start = max(0, len(global_kept_embeddings) - self._max_compare)
                for prev in global_kept_embeddings[start:]:
                    if _cosine_similarity(prev, emb) >= self._threshold:
                        is_dup_global = True
                        break
            if is_dup_global:
                stats.removed_across_sections += 1
                continue

            kept_blocks.append(b.rstrip())
            kept_norms.append(bn)
            if emb is not None:
                kept_embeds.append(emb)
                global_kept_embeddings.append(emb)

        cleaned = _join_blocks(kept_blocks)
        return cleaned, stats, global_kept_embeddings


def strip_empty_outline_items(md: str, *, skeleton: str) -> tuple[str, int]:
    src = (md or "").replace("\r\n", "\n")
    sk = (skeleton or "").replace("\r\n", "\n")
    if not src.strip() or not sk.strip():
        return src.strip(), 0

    skeleton_items: set[str] = set()
    for line in sk.split("\n"):
        s = line.lstrip(" ")
        if s.startswith("- "):
            item = s[2:].strip()
            if item:
                skeleton_items.add(item)
    if not skeleton_items:
        return src.strip(), 0

    lines = src.split("\n")
    out: list[str] = []
    removed = 0
    i = 0
    n = len(lines)

    def _is_list_item(ln: str) -> tuple[bool, int, str]:
        lstrip = ln.lstrip(" ")
        if not lstrip.startswith("- "):
            return False, 0, ""
        indent = len(ln) - len(lstrip)
        item = lstrip[2:].strip()
        return True, indent, item

    while i < n:
        line = lines[i]
        ok, indent, item = _is_list_item(line)
        if not ok or item not in skeleton_items:
            out.append(line.rstrip())
            i += 1
            continue

        j = i + 1
        while j < n and not lines[j].strip():
            j += 1

        if j >= n:
            removed += 1
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            continue

        next_line = lines[j]
        next_indent = len(next_line) - len(next_line.lstrip(" "))
        if next_indent <= indent:
            removed += 1
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            continue

        out.append(line.rstrip())
        i += 1

    cleaned = "\n".join(out).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed

