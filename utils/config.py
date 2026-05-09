"""utils/config.py – centralised settings via pydantic-settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it works regardless of cwd.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM – provider selection
    llm_provider: str = "hf"                        # "hf", "openai", or "ollama"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"                          # used when provider=openai
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "lfm2.5-thinking:1.2b"               # used when provider=ollama
    
    # HuggingFace
    huggingface_token: str = ""
    hf_token: str = "" # Alternative key name often used
    llm_hf_model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    hf_model_id: str = "yiyanghkust/finbert-tone" # for sentiment tool
    hf_base_url: str = "https://router.huggingface.co/v1"
    
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
