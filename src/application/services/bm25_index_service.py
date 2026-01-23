"""BM25 预建索引（T095 优化）。

目标：
- 在“建库阶段”预建并持久化 BM25 索引，避免查询时全量拉取并重建
- 不引入额外三方依赖（仅用标准库实现 BM25）
- 支持中文：默认采用“字/词混合”轻量分词（CJK 单字 + ASCII 单词）
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.application.schemas.ingest import KBChunk


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _is_cjk(ch: str) -> bool:
    # 常见 CJK 统一表意文字范围（足够覆盖中文检索场景）
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x3000 <= code <= 0x303F  # CJK Symbols and Punctuation
    )


def bm25_tokenize(text: str) -> list[str]:
    """轻量 tokenizer：ASCII 单词 + CJK 单字。

    说明：
    - 不依赖 jieba 等外部库，适配 Windows/离线环境
    - 对中文而言，字符级别 BM25 召回通常比“按空格分词”更靠谱
    """
    if not text:
        return []

    s = text.strip().lower()
    tokens: list[str] = []

    buf: list[str] = []
    for ch in s:
        if _is_cjk(ch):
            if buf:
                w = "".join(buf).strip()
                if w:
                    # 进一步按 word 规则拆分，避免混入符号
                    tokens.extend(_WORD_RE.findall(w))
                buf = []
            # CJK：单字 token（过滤空白）
            if ch.strip():
                tokens.append(ch)
            continue

        # ASCII：积累，遇到分隔符再落盘
        if ch.isalnum() or ch == "_":
            buf.append(ch)
        else:
            if buf:
                w = "".join(buf).strip()
                if w:
                    tokens.extend(_WORD_RE.findall(w))
                buf = []

    if buf:
        w = "".join(buf).strip()
        if w:
            tokens.extend(_WORD_RE.findall(w))

    return [t for t in tokens if t]


@dataclass
class BM25Doc:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    source_file_id: str
    doc_len: int
    tf: dict[str, int]


@dataclass
class BM25Index:
    """可 JSON 序列化的 BM25 索引。"""

    version: str
    tokenizer: str
    k1: float
    b: float
    doc_count: int
    avgdl: float
    # df / idf：token -> value
    df: dict[str, int]
    idf: dict[str, float]
    # docs：逐文档 tf 与元信息
    docs: list[BM25Doc]

    @staticmethod
    def build_from_chunks(
        chunks: list[KBChunk],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: str = "cjk_char+ascii_word:v1",
    ) -> "BM25Index":
        docs: list[BM25Doc] = []
        df: dict[str, int] = {}

        for ch in chunks:
            content = ch.content or ""
            tokens = bm25_tokenize(content)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            # df：每文档唯一 token
            for t in set(tf.keys()):
                df[t] = df.get(t, 0) + 1

            docs.append(
                BM25Doc(
                    chunk_id=ch.chunk_id,
                    content=content,
                    metadata=ch.metadata or {},
                    source_file_id=str(ch.source_file_id or ""),
                    doc_len=len(tokens),
                    tf=tf,
                )
            )

        n = len(docs)
        avgdl = (sum(d.doc_len for d in docs) / n) if n > 0 else 0.0

        # idf：BM25 常用公式（加 1 防止负值/零）
        idf: dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

        return BM25Index(
            version="bm25_index:v1",
            tokenizer=tokenizer,
            k1=float(k1),
            b=float(b),
            doc_count=n,
            avgdl=float(avgdl),
            df=df,
            idf=idf,
            docs=docs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tokenizer": self.tokenizer,
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "df": self.df,
            "idf": self.idf,
            "docs": [
                {
                    "chunk_id": d.chunk_id,
                    "content": d.content,
                    "metadata": d.metadata,
                    "source_file_id": d.source_file_id,
                    "doc_len": d.doc_len,
                    "tf": d.tf,
                }
                for d in self.docs
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> "BM25Index":
        data = json.loads(path.read_text(encoding="utf-8"))
        docs: list[BM25Doc] = []
        for d in data.get("docs", []) or []:
            if not isinstance(d, dict):
                continue
            docs.append(
                BM25Doc(
                    chunk_id=str(d.get("chunk_id", "")),
                    content=str(d.get("content", "")),
                    metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
                    source_file_id=str(d.get("source_file_id", "")),
                    doc_len=int(d.get("doc_len", 0) or 0),
                    tf=d.get("tf") if isinstance(d.get("tf"), dict) else {},
                )
            )
        return BM25Index(
            version=str(data.get("version", "bm25_index:v1")),
            tokenizer=str(data.get("tokenizer", "cjk_char+ascii_word:v1")),
            k1=float(data.get("k1", 1.2)),
            b=float(data.get("b", 0.75)),
            doc_count=int(data.get("doc_count", len(docs)) or len(docs)),
            avgdl=float(data.get("avgdl", 0.0)),
            df=data.get("df") if isinstance(data.get("df"), dict) else {},
            idf=data.get("idf") if isinstance(data.get("idf"), dict) else {},
            docs=docs,
        )

    def _match_filter(self, meta: dict[str, Any], flt: dict[str, Any] | None) -> bool:
        if not flt:
            return True
        # 仅支持简单等值过滤（与 Chroma 的 where 语义不完全等价，但足够用于常用场景）
        for k, v in flt.items():
            if meta.get(k) != v:
                return False
        return True

    def search(
        self,
        *,
        query: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or top_k <= 0 or not self.docs:
            return []

        q_tokens = bm25_tokenize(query)
        if not q_tokens:
            return []

        # 仅用唯一 token（BM25 query tf 可忽略）
        q_terms = list(dict.fromkeys(q_tokens))

        scores: list[tuple[float, int]] = []
        n = max(self.doc_count, 1)
        avgdl = self.avgdl or 1.0
        k1 = self.k1
        b = self.b

        for i, d in enumerate(self.docs):
            if filter_metadata and not self._match_filter(d.metadata, filter_metadata):
                continue
            dl = d.doc_len or 0
            denom_const = k1 * (1.0 - b + b * (dl / avgdl))
            score = 0.0
            for term in q_terms:
                tf = d.tf.get(term)
                if not tf:
                    continue
                # idf 缺失时按 df=0 估算（更保守）
                idf = self.idf.get(term)
                if idf is None:
                    df = int(self.df.get(term, 0) or 0)
                    idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                score += float(idf) * (tf * (k1 + 1.0)) / (tf + denom_const)
            if score > 0:
                scores.append((score, i))

        if not scores:
            return []

        scores.sort(key=lambda x: x[0], reverse=True)
        picked = scores[:top_k]

        out: list[dict[str, Any]] = []
        for s, idx in picked:
            d = self.docs[idx]
            # 归一化：用于展示/调试（融合阶段使用 RRF 不依赖该分数）
            norm = float(s) / (float(s) + 1.0)
            out.append(
                {
                    "chunk_id": d.chunk_id,
                    "content": d.content,
                    "metadata": d.metadata,
                    "source_file_id": d.source_file_id,
                    "bm25_score": float(s),
                    "bm25_score_norm": float(norm),
                }
            )
        return out

