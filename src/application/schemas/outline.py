"""大纲 Pydantic 模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutlinePolishRequest(BaseModel):
    """大纲润色请求。"""

    outline: str = Field(..., min_length=1, description="大纲内容，Markdown格式")


class OutlinePolishResponse(BaseModel):
    """大纲润色响应。"""

    polished_outline: str = Field(..., description="润色后的大纲")


class OutlineSaveRequest(BaseModel):
    """大纲保存请求。"""

    outline: str = Field(..., min_length=1, description="大纲内容")
    filename: str = Field(..., min_length=1, max_length=255, description="文件名，不含扩展名")


class OutlineSaveResponse(BaseModel):
    """大纲保存响应。"""

    file_path: str = Field(..., description="保存后的文件路径")
