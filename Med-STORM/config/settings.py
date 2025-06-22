"""Application configuration settings."""
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, HttpUrl, field_validator, ConfigDict, model_validator
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    APP_NAME: str = "Med-STORM"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Cache settings
    CACHE_DIR: Path = Path(".llm_cache")
    CACHE_ENABLED: bool = True
    CACHE_TTL_HOURS: int = 168  # 1 week
    CACHE_MAX_SIZE_MB: int = 2048  # 2GB max cache size
    CACHE_COMPRESSION: bool = True
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 60  # seconds total request timeout
    CONNECTION_TIMEOUT: int = 10  # seconds for initial connection
    SOCKET_TIMEOUT: int = 30  # seconds for socket operations
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 0.5
    TIMEOUT_MULTIPLIER: float = 1.5  # Multiplier for timeouts on retries
    
    # Timeout profiles (base timeouts in seconds)
    TIMEOUT_PROFILES: Dict[str, Dict[str, float]] = {
        "fast": {"total": 30, "connect": 5, "sock": 15},
        "standard": {"total": 60, "connect": 10, "sock": 30},
        "slow": {"total": 120, "connect": 15, "sock": 60}
    }
    
    # LLM API Keys
    OPENAI_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    GOOGLE_API_KEY: Optional[SecretStr] = None
    DEEPSEEK_API_KEY: Optional[SecretStr] = None
    
    # Knowledge Source APIs
    PUBMED_API_KEY: Optional[str] = None
    PUBMED_EMAIL: Optional[str] = None
    SERPER_API_KEY: Optional[SecretStr] = None
    
    # OpenRouter API Key (Primary LLM)
    OPENROUTER_API_KEY: Optional[SecretStr] = None
    HTTP_REFERER: str = "https://med-storm.ai"
    SITE_NAME: str = "Med-STORM"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "corpus_bariatric_surgery_for_type_2_diabetes"
    
    # Aliases for compatibility
    deepseek_api_key: Optional[SecretStr] = None
    serper_api_key: Optional[SecretStr] = None
    pubmed_api_key: Optional[str] = None

    # Model configurations
    DEFAULT_LLM: str = "deepseek"  # Options: openai, anthropic, google, deepseek
    DEFAULT_MODEL: str = "deepseek-chat"
    MAX_TOKENS: int = 4000

    # Streaming and processing settings
    STREAMING_ENABLED: bool = True
    STREAMING_CHUNK_SIZE: int = 1024  # bytes
    STREAMING_TIMEOUT: int = 300  # seconds
    STREAMING_BUFFER_SIZE: int = 8192  # 8KB buffer for streaming
    STREAMING_MAX_CONCURRENT: int = 5  # Max concurrent streaming operations

    # Processing batch sizes
    BATCH_SIZE_QUESTIONS: int = 5  # Number of questions to process in parallel
    BATCH_SIZE_SOURCES: int = 20  # Number of sources to process in parallel

    # Generation parameters
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.95
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    MAX_TOKENS_RESPONSE: int = 4000  # Max tokens for generated responses

    # Search settings with enhanced options
    MAX_SEARCH_RESULTS: int = 20
    SEARCH_TIMEOUT: int = 30  # seconds
    SEARCH_MAX_RETRIES: int = 3
    SEARCH_CACHE_TTL_HOURS: int = 24  # Cache search results for 24 hours
    # Default minimum publication year for evidence retrieval
    MIN_PUBLICATION_YEAR: int = 2018
    SORT_BY_DATE: bool = True  # Sort search results by date by default

    # Deduplication settings
    DEDUPLICATION_ENABLED: bool = True
    DEDUPLICATION_METHOD: str = "hybrid"  # Options: exact, simhash, tfidf, hybrid
    SIMILARITY_THRESHOLD: float = 0.85  # Threshold for considering documents similar

    # Personalization settings
    PERSONALIZATION_ENABLED: bool = True
    MAX_PATIENT_FACTORS: int = 20  # Maximum number of patient factors to consider

    # Table generation settings
    MAX_TABLE_ROWS: int = 50  # Maximum rows in comparison tables
    MAX_TABLE_COLUMNS: int = 8  # Maximum columns in comparison tables

    # Executive summary settings
    SUMMARY_TARGET_LENGTH: int = 1000  # Target length in words
    SUMMARY_INCLUDE_TABLES: bool = True  # Include tables in summaries
    SUMMARY_CONFIDENCE_THRESHOLD: float = 0.7  # Minimum confidence to include in summary

    # Trusted domains for search results with tier classification
    TRUSTED_DOMAINS: Dict[str, List[str]] = {
        "tier_1": [
            "who.int",
            "cdc.gov",
            "nih.gov",
            "nejm.org",
            "jamanetwork.com",
            "thelancet.com",
            "bmj.com",
            "jci.org",
            "nature.com",
            "sciencedirect.com",
            "ahajournals.org",
            "acpjournals.org",
            "cochranelibrary.com",
        ],
        "tier_2": [
            "mayoclinic.org",
            "uptodate.com",
            "nice.org.uk",
            "aafp.org",
            "acog.org",
            "diabetes.org",
            "heart.org",
            "cancer.org",
        ],
        "tier_3": [
            "webmd.com",
            "medscape.com",
            "medlineplus.gov",
            "healthline.com",
        ],
    }

    # Evidence confidence levels configuration - Updated with tiered approach
    EVIDENCE_LEVELS: Dict[str, Dict[str, any]] = {
        "tier_1": {
            "sources": ["pubmed", "clinicaltrials.gov"],
            "confidence": 0.95,
            "description": "Peer-reviewed clinical trials, systematic reviews, and meta-analyses",
        },
        "tier_2": {
            "sources": [
                "scholar",
                "guidelines",
                "who.int",
                "cdc.gov",
                "nice.org.uk",
            ],
            "confidence": 0.80,
            "description": "Guidelines from professional organizations, government health agencies, and high-quality observational studies",
        },
        "tier_3": {
            "sources": [
                "web",
                "preprints",
                "mayoclinic.org",
                "medlineplus.gov",
                "news",
            ],
            "confidence": 0.65,
            "description": "Other sources including preprints, conference abstracts, and general medical information",
        },
    }

    # ------------------------------------------------------------------
    # Validators (Pydantic v2)
    # ------------------------------------------------------------------

    @field_validator("MIN_PUBLICATION_YEAR", mode="before")
    def validate_min_publication_year(cls, v):  # noqa: N805
        """Ensure the minimum publication year is an integer even if provided as a string."""
        try:
            return int(str(v).split()[0])  # Cast and ignore inline comments
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MIN_PUBLICATION_YEAR must be an integer year, e.g., 2018"
            ) from exc

    @model_validator(mode="after")
    def _assemble_qdrant_url(self):  # noqa: D401
        """Build `QDRANT_URL` if not explicitly provided."""
        if self.QDRANT_URL:
            return self

        assembled = f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
        # Use model_copy to avoid mutating self in-place in case of immutability.
        return self.model_copy(update={"QDRANT_URL": assembled})

    # ------------------------------------------------------------------
    # Settings configuration (Pydantic v2)
    # ------------------------------------------------------------------
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )


# Create settings instance
settings = Settings()

# Ensure cache directory exists
os.makedirs(settings.CACHE_DIR, exist_ok=True)
