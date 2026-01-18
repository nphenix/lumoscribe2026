"""Content Generation 模块提示词定义。"""

SCOPE_CONTENT_GENERATION_SECTION = "content_generation:generate_section"
SCOPE_OUTLINE_POLISH = "content_generation:polish_outline"

PROMPTS = {
    SCOPE_CONTENT_GENERATION_SECTION: {
        "description": "按模板章节生成内容（基于检索上下文）",
        "format": "text",
        "content": """你是一个专业的内容生成助手。根据提供的上下文信息，生成符合要求的章节内容。

## 任务要求
1. 根据上下文信息，生成高质量的章节内容
2. 内容要准确、专业、有深度
3. 保持与模板风格一致
4. 如果上下文信息不足以回答问题，请明确说明

## 章节标题
{title}

## 章节模板
{template_content}

## 上下文信息
{context}

## 生成要求
请生成完整的章节内容，直接输出最终结果，不要包含任何解释或前缀。""",
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

