from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from typing import Optional

# Load .env file from the project root (override to prioritise local dev env vars)
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    override=True,
)

class Settings(BaseSettings):
    """
    Application settings.
    """
    # LLM Provider Settings
    LLM_PROVIDER: str = "deepseek"
    DEEPSEEK_API_KEY: Optional[str] = Field(
        default=None, validation_alias="DEEPSEEK_API_KEY", repr=False
    )
    DEEPSEEK_BASE_URL: Optional[str] = "https://api.deepseek.com/v1"
    OPENAI_API_KEY: Optional[str] = Field(
        default=None, validation_alias="OPENAI_API_KEY", repr=False
    )
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None, validation_alias="OPENROUTER_API_KEY", repr=False
    )

    # Search Provider Settings
    SERPER_API_KEY: Optional[str] = Field(
        default=None, validation_alias="SERPER_API_KEY", repr=False
    )
    PUBMED_EMAIL: Optional[str] = Field(
        default="your.email@example.com", validation_alias="PUBMED_EMAIL"
    )

    # Cache Settings
    CACHE_TYPE: str = "redis"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6380  # Mapped to 6380 in docker-compose
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = Field(
        default=None, validation_alias="REDIS_PASSWORD", repr=False
    )

    # Vector Database Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Retry Mechanism
    MAX_RETRIES: int = Field(3, validation_alias="MAX_RETRIES")
    RETRY_DELAY: int = Field(2, validation_alias="RETRY_DELAY")  # seconds
    search_max_retries: int = Field(3, validation_alias="SEARCH_MAX_RETRIES")
    search_retry_delay: float = Field(1.0, validation_alias="SEARCH_RETRY_DELAY")
    
    # Logging
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------
    # Pydantic v2 configuration
    # ------------------------------------------------------------------
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

settings = Settings()
