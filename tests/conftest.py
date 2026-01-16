from __future__ import annotations

from pathlib import Path


def pytest_configure(config) -> None:
    """将 pytest 的 basetemp 优先指向 F:\\temp（如可用）。

    说明：
    - `tmp_path`/`tmp_path_factory` 会使用 basetemp 作为临时目录根；
    - 如果用户已显式传入 `--basetemp`，则不覆盖；
    - 如果 F 盘不存在/不可写，则回退为 pytest 默认行为（系统临时目录）。
    """

    # 若用户已显式设置（例如命令行 --basetemp 或 PYTEST_ADDOPTS），则不覆盖
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        return

    preferred_root = Path(r"F:\temp")
    preferred = preferred_root / "pytest"

    try:
        preferred.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 回退到 pytest 默认 basetemp（通常是系统临时目录）
        return

    config.option.basetemp = str(preferred)
