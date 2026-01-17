"""Document Cleaning 模块提示词定义。"""

# Scope 常量
SCOPE_DOC_CLEANING = "doc_cleaning:clean_text"
SCOPE_CHART_EXTRACTION = "chart_extraction:extract_json"

# 提示词种子
PROMPTS = {
    SCOPE_DOC_CLEANING: {
        "description": "文档清洗标准提示词",
        "format": "text",
        "content": """{instructions}

请对以下文档进行清洗，保留有意义的正文内容：

{text}

请直接返回清洗后的文本，不要添加任何解释或注释。""",
    },
    SCOPE_CHART_EXTRACTION: {
        "description": "图表转JSON提取",
        "format": "text",
        "content": """请分析以下{description}图片，提取其中的数据并以 JSON 格式返回。

要求：
1. 提取所有数据点和数值
2. 保持数据的层级结构
3. 如果是表格，请保持行列结构
4. 如果有标题或标签，请一并提取

请直接返回 JSON 对象，不要添加任何解释。JSON 格式如下：
{{
    "title": "图表标题",
    "type": "{chart_type}",
    "data": [
        {{
            "label": "数据标签",
            "value": 数值,
            "description": "描述信息"
        }}
    ],
    "labels": ["x轴标签1", "x轴标签2", ...],
    "series": [
        {{
            "name": "系列名称",
            "data": [数值1, 数值2, ...]
        }}
    ],
    "metadata": {{
        "source": "数据来源",
        "unit": "单位",
        "note": "备注"
    }}
}}

请直接返回 JSON 对象：""",
    },
}
