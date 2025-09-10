import logging
import os
from functools import lru_cache
from typing import Any, List, Optional

from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


class Settings(BaseSettings):

    OPENAI_API_KEY: Optional[str] = None
 
    GOOGLE_API_KEY: Optional[str] = None

    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_LEGAL_CASES_COLLECTION_NAME: Optional[str] = "bd_legal_cases"
    QDRANT_LAW_REFERENCE_COLLECTION_NAME: Optional[str] = "bd_law_reference"

    # Custom model settings (for Modal deployment)
    CUSTOM_MODEL_URL: Optional[str] = None
    CUSTOM_MODEL_API_KEY: Optional[str] = "custom-api-key"



    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = None



    class Config:
        env_file = ".env"
        env_ignore_empty = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    env_name = os.environ.get("ENVIRONMENT", "development")
    logging.info(f"Loading settings for environment: {env_name}")
    load_dotenv(override=True)
    return Settings()


settings = get_settings()