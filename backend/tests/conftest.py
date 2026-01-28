# Ensure backend/.env is loaded for tests that rely on DATABASE_URL.

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in test env
    load_dotenv = None


def pytest_configure() -> None:
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
