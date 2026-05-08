"""utils/config.py – centralised settings via pydantic-settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
