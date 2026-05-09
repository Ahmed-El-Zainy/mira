"""utils/config.py – centralised settings via pydantic-settings.

dotenv loading strategy (two layers, both needed):
  1. load_dotenv() here  – populates os.environ *before* pydantic-settings
     reads it, so tests and scripts work regardless of cwd.
  2. env_file in model_config – pydantic-settings secondary fallback for
     any var that load_dotenv missed (e.g. running via `uvicorn` directly).

override=False ensures real environment variables (Docker, CI secrets)
always win over whatever is in the .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it works regardless of cwd.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM – provider selection
    llm_provider: str = "openai"                        # "openai" or "ollama"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"                          # used when provider=openai
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "lfm2.5-thinking:1.2b"               # used when provider=ollama
    max_tool_calls_per_job: int = 10

    # News
    news_api_key: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # App
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Monitoring
    default_monitoring_cadence_hours: int = 24
    price_deviation_threshold: float = 2.0
    volume_spike_threshold: float = 2.0
    min_new_articles: int = 5


@lru_cache()
def get_settings() -> Settings:
    return Settings()
