"""内容生成文本工具（T042）。

目标：
- 统一处理模型输出中可能出现的 think/analysis 痕迹
- 提供编码兼容的文本读取
"""

from __future__ import annotations

import re
from pathlib import Path

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

CHART_SNIPPET_RE = re.compile(r"(?m)^\[图表\]\s*(.+?)\s*$")

_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_THINK_FENCE_RE = re.compile(r"(?is)```(?:thinking|thought|analysis)[\s\S]*?```")
_REASONING_HEAD_RE = re.compile(
    r"(?im)^(?:#+\s*)?(?:思考|推理|Chain-of-Thought|CoT|Thoughts?)\s*[:：].*$"
)


def read_text_best_effort(path: Path) -> str:
    """尽量读取文本（兼容 Windows 常见编码）。"""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def strip_model_think(text: str) -> str:
    """移除模型输出中的“思考/推理”痕迹，保证最终产物为纯正文。"""
    if not text:
        return ""
    s = str(text)
    s = _THINK_BLOCK_RE.sub("", s)
    s = _THINK_FENCE_RE.sub("", s)
    s = _REASONING_HEAD_RE.sub("", s)

    head = s.lstrip()
    if head.startswith("思考") or head.lower().startswith("thinking") or head.lower().startswith("analysis"):
        parts = re.split(r"\n\s*\n", s, maxsplit=1)
        if len(parts) == 2:
            s = parts[1]
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def strip_leading_title_heading(text: str, *, title: str) -> str:
    """移除模型重复输出的章节标题（例如开头的 '# 第一章...'），避免与外层 <h2> 重复。"""
    if not text or not title:
        return text or ""
    ttl = str(title).strip()
    if not ttl:
        return text

    def _norm(s: str) -> str:
        return re.sub(r"[\s:：]+", "", (s or "").strip())

    lines = str(text).replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return ""

    m = re.match(r"^#{1,6}\s+(.+?)\s*$", lines[i].strip())
    if not m:
        return text
    head = (m.group(1) or "").strip()
    if _norm(head) != _norm(ttl):
        return text

    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:]).strip()


def norm_compact(s: str) -> str:
    """用于弱匹配的归一化：去除多余空白，转小写。"""
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

