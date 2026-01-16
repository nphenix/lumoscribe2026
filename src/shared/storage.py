from __future__ import annotations

from pathlib import Path


class LocalFileStorage:
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
