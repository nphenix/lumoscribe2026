"""白皮书大纲解析与骨架生成（T096）。"""

from __future__ import annotations

import re

from src.application.services.content_generation.types import OutlineItem


def parse_outline_items_from_section_content(section_content: str) -> list[OutlineItem]:
    """从章节 section 内容中解析大纲条目。

    约定：白皮书大纲子章节通常使用 Markdown 列表，例如：
    - `- 1.1 标题`（无序列表）
    - `1. 1.1 标题`（有序列表）
    也支持 Markdown 标题，例如：
    - `### 1.1 标题`
    - `#### 1.1.1 标题`
    嵌套层级使用缩进（常见 2/4 空格均可），例如：
    - `    - 2.2.1 标题`
    """
    items: list[OutlineItem] = []
    lines = (section_content or "").replace("\r\n", "\n").split("\n")
    for line in lines:
        # 兼容无序列表(-/*)与有序列表(1./1))
        m = re.match(r"^(\s*)(?:[-*]\s+|\d+[.)]\s+)(.*)$", line)
        depth: int | None = None
        raw: str | None = None
        indent = ""
        if m:
            indent = m.group(1) or ""
            raw = (m.group(2) or "").strip()
            if raw:
                depth = 1 + max(0, len(indent) // 4)
        else:
            mh = re.match(r"^(\s*)(#{3,6})\s+(.+)$", line)
            if mh:
                indent = mh.group(1) or ""
                hashes = mh.group(2) or "###"
                raw = (mh.group(3) or "").strip()
                if raw:
                    heading_depth = max(1, len(hashes) - 2)
                    depth = heading_depth + max(0, len(indent) // 4)
            else:
                mn = re.match(r"^(\s*)(\d+(?:\.\d+)+)\s+(.+)$", line)
                if mn:
                    indent = mn.group(1) or ""
                    number0 = (mn.group(2) or "").strip()
                    title0 = (mn.group(3) or "").strip()
                    if number0 and title0:
                        raw = f"{number0} {title0}".strip()
                        number_depth = max(1, number0.count("."))
                        depth = number_depth + max(0, len(indent) // 4)

        if not raw or depth is None:
            continue

        m2 = re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", raw)
        if m2:
            number = m2.group(1).strip()
            title = (m2.group(2) or "").strip()
        else:
            # 兼容：- 2.2 标题（只有一层点也可）
            m3 = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", raw)
            if m3:
                number = m3.group(1).strip() or None
                title = (m3.group(2) or "").strip()
            else:
                number = None
                title = raw

        if title:
            items.append(OutlineItem(raw=raw, number=number, title=title, depth=depth))

    return items


def build_outline_skeleton_markdown(items: list[OutlineItem]) -> str:
    """将 outline items 生成 Markdown 骨架（用于约束 LLM 输出结构）。"""
    if not items:
        return ""
    out: list[str] = []
    for it in items:
        # 严格保持白皮书大纲的列表格式：缩进层级 + "- 1.1 标题"
        indent = "    " * max(0, it.depth - 1)
        out.append(f"{indent}- {it.display}".rstrip())
        # 给正文留出空间：正文应缩进到 list item 内（额外 4 个空格）
        out.append(f"{indent}    ")
    return "\n".join(out).rstrip()

