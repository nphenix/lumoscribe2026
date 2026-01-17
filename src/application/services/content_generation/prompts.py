"""Content Generation 模块提示词定义。"""

SCOPE_CONTENT_GENERATION_SECTION = "content_generation:generate_section"

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
    }
}

