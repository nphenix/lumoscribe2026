from __future__ import annotations

import os
import uvicorn

from src.shared.config import get_settings

from src.interfaces.api.app import create_app


app = create_app()


def main() -> None:
    settings = get_settings()
    # 长任务（如全量建库）建议关闭 reload，避免文件变更触发重启导致任务中断
    reload = os.getenv("LUMO_API_RELOAD", "0").strip().lower() in {"1", "true", "yes"}
    uvicorn.run(
        "src.interfaces.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
