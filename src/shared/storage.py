"""本地文件存储模块。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal


class StorageCategory(str, Enum):
    """存储类别枚举。"""

    # 源文件
    SOURCES = "sources"
    SOURCES_ARCHIVE = "sources-archive"

    # 模板
    TEMPLATES_CUSTOM = "templates/custom"
    TEMPLATES_SYSTEM = "templates/system"

    # 目标文件
    TARGETS = "targets"

    # 中间态产物
    INTERMEDIATES_MINERU_RAW = "intermediates/mineru_raw"
    INTERMEDIATES_CLEANED_DOC = "intermediates/cleaned_doc"
    INTERMEDIATES_CHART_JSON = "intermediates/chart_json"
    INTERMEDIATES_KB_CHUNKS = "intermediates/kb_chunks"


class LocalFileStorage:
    """本地文件存储类。"""

    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relative_path: str) -> Path:
        # 统一使用 posix 风格的相对路径，避免路径穿越
        safe = Path(relative_path)
        if safe.is_absolute() or ".." in safe.parts:
            raise ValueError("invalid relative path")
        path = (self._root / safe).resolve()
        if self._root.resolve() not in path.parents and path != self._root.resolve():
            raise ValueError("path traversal detected")
        return path

    def write_bytes(self, relative_path: str, data: bytes) -> Path:
        path = self.resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


# 存储根目录配置
DATA_ROOT = Path("data")


def ensure_directory_structure(base_path: Path = DATA_ROOT) -> dict[str, Path]:
    """确保所有存储目录存在。

    Args:
        base_path: 存储根目录路径（默认: data/）

    Returns:
        目录路径字典，key 为类别标识
    """
    directories = {}

    # 源文件目录
    directories["sources"] = base_path / "sources"
    directories["sources-archive"] = base_path / "sources-archive"

    # 模板目录
    directories["templates/custom"] = base_path / "templates" / "custom"
    directories["templates/system"] = base_path / "templates" / "system"

    # 目标文件目录
    directories["targets"] = base_path / "targets"

    # 中间态产物目录
    directories["intermediates/mineru_raw"] = base_path / "intermediates" / "mineru_raw"
    directories["intermediates/cleaned_doc"] = base_path / "intermediates" / "cleaned_doc"
    directories["intermediates/chart_json"] = base_path / "intermediates" / "chart_json"
    directories["intermediates/kb_chunks"] = base_path / "intermediates" / "kb_chunks"

    # 创建所有目录
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def get_workspace_path(
    workspace_name: str,
    category: StorageCategory | str,
    base_path: Path = DATA_ROOT,
) -> Path:
    """获取特定工作空间的存储路径。

    Args:
        workspace_name: 工作空间名称（支持中文）
        category: 存储类别
        base_path: 存储根目录路径

    Returns:
        完整的工作空间目录路径

    Example:
        >>> get_workspace_path("我的项目", StorageCategory.SOURCES)
        PosixPath('data/sources/我的项目')
    """
    # 处理字符串类型的 category（兼容旧接口）
    if isinstance(category, str):
        category = StorageCategory(category)

    # 构建路径：base_path / category.value / workspace_name
    category_path = category.value
    full_path = base_path / category_path / workspace_name

    # 确保目录存在
    full_path.mkdir(parents=True, exist_ok=True)

    return full_path


def get_storage_path(
    workspace_name: str,
    category: StorageCategory | str,
    filename: str,
    base_path: Path = DATA_ROOT,
) -> Path:
    """获取文件的完整存储路径。

    Args:
        workspace_name: 工作空间名称（支持中文）
        category: 存储类别
        filename: 文件名
        base_path: 存储根目录路径

    Returns:
        完整的文件路径

    Example:
        >>> get_storage_path("项目A", StorageCategory.SOURCES, "document.pdf")
        PosixPath('data/sources/项目A/document.pdf')
    """
    workspace_path = get_workspace_path(workspace_name, category, base_path)
    file_path = workspace_path / filename

    # 确保父目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


# 便捷函数：获取各类型路径
def get_sources_path(workspace_name: str, filename: str = "") -> Path:
    """获取源文件路径。"""
    return get_storage_path(workspace_name, StorageCategory.SOURCES, filename)


def get_sources_archive_path(workspace_name: str, filename: str = "") -> Path:
    """获取源文件归档路径。"""
    return get_storage_path(workspace_name, StorageCategory.SOURCES_ARCHIVE, filename)


def get_templates_custom_path(workspace_name: str, filename: str = "") -> Path:
    """获取自定义模板路径。"""
    return get_storage_path(workspace_name, StorageCategory.TEMPLATES_CUSTOM, filename)


def get_templates_system_path(workspace_name: str, filename: str = "") -> Path:
    """获取系统模板路径。"""
    return get_storage_path(workspace_name, StorageCategory.TEMPLATES_SYSTEM, filename)


def get_targets_path(workspace_name: str, filename: str = "") -> Path:
    """获取目标文件路径。"""
    return get_storage_path(workspace_name, StorageCategory.TARGETS, filename)


def get_intermediates_path(
    workspace_name: str,
    subcategory: Literal["mineru_raw", "cleaned_doc", "chart_json", "kb_chunks"],
    filename: str = "",
) -> Path:
    """获取中间态产物路径。

    Args:
        workspace_name: 工作空间名称
        subcategory: 中间态子类别
        filename: 文件名
    """
    category_map = {
        "mineru_raw": StorageCategory.INTERMEDIATES_MINERU_RAW,
        "cleaned_doc": StorageCategory.INTERMEDIATES_CLEANED_DOC,
        "chart_json": StorageCategory.INTERMEDIATES_CHART_JSON,
        "kb_chunks": StorageCategory.INTERMEDIATES_KB_CHUNKS,
    }
    category = category_map.get(subcategory, StorageCategory.INTERMEDIATES_MINERU_RAW)
    return get_storage_path(workspace_name, category, filename)
