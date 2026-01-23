"""Content Generation 模块提示词定义。"""

SCOPE_CONTENT_GENERATION_SECTION = "content_generation:generate_section"
SCOPE_OUTLINE_POLISH = "content_generation:polish_outline"

PROMPTS = {
    SCOPE_CONTENT_GENERATION_SECTION: {
        "description": "按模板章节生成内容（基于检索上下文）",
        "format": "text",
        "content": """你是一个专业的白皮书撰写助手。你将基于“章节模板骨架 + RAG 检索上下文”生成该章节的 Markdown 正文。

## 核心约束（必须严格遵守）
1. **必须严格保持模板骨架的结构与顺序**：不得改动、不得删减、不得重排任何标题行（含编号/标点/冒号/引号）。
2. **必须逐条填充**：模板骨架中每一个 `- 1.1 ...`（含缩进层级的子条目）下都必须生成对应内容，不能留空。
3. **优先使用上下文信息**：与上下文一致的事实/数字/政策名称可以引用；若上下文未覆盖则用更一般、保守、可验证的表述补全，避免捏造具体条款/数据。
4. **禁止输出思考过程**：不要输出任何“思考/推理/analysis/thought”等内容，不要输出 `<think>` 标签，不要输出分隔线后的内部推理。
5. **输出格式**：只输出 Markdown 正文，不要输出解释、不要输出前缀（如“以下为…”），不要输出 JSON。

## 列表格式要求（重要）
- 模板骨架使用 Markdown 列表（含缩进层级）。你必须保持每一行 `- ...` 原样不变。
- 每个条目的正文必须写在该条目下面，并且**缩进 4 个空格**以保持属于该列表项。

## 文档标题（可选参考）
{document_title}

## 章节标题
{title}

## 章节模板骨架（标题必须原样保留）
{template_content}

## 大纲条目（便于你检查是否逐条覆盖）
{outline_items}

## RAG 上下文（优先引用）
{context}

## 输出要求
请在保持“章节模板骨架”标题完全不变的前提下，在每个标题下生成正文段落；直接输出最终 Markdown。""",
    },
    SCOPE_OUTLINE_POLISH: {
        "description": "用户自定义大纲的 LLM 润色",
        "format": "text",
        "content": """你是一个专业的文档结构优化助手。请对用户提供的大纲进行润色和优化。

## 任务要求
1. 保持原有大纲的结构和层次关系
2. 优化标题的表述，使其更加专业、清晰、准确
3. 确保标题之间的逻辑关系合理
4. 保持原有的大纲格式（Markdown 格式）

## 原始大纲
{outline}

## 润色要求
请直接输出润色后的大纲，保持 Markdown 格式，不要包含任何解释或前缀。""",
    },
}

