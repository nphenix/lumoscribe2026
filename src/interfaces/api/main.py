from __future__ import annotations

import uvicorn

from src.shared.config import get_settings

from src.interfaces.api.app import create_app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "src.interfaces.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
