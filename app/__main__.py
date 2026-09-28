"""Entry point: `python -m app` launches the API server (with the UI mounted at /)."""
from __future__ import annotations

import uvicorn

from app.core.config import settings


def main():
    uvicorn.run("app.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)


if __name__ == "__main__":
    main()
