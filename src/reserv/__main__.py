"""Entrypoint: `python -m reserv` (or uvicorn reserv.server:app)."""
from __future__ import annotations

import logging

import uvicorn

from .config import load_settings
from .server import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
