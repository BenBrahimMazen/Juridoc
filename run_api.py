"""Launch the FastAPI service with uvicorn (dev server)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

import config  # noqa: E402


def main() -> None:
    uvicorn.run(
        "src.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
