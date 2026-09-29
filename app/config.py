"""Application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ — parent of the `app` package
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _BACKEND_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=_DEFAULT_DATA_DIR, alias="DATA_DIR")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        alias="CORS_ORIGINS",
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="portfolio_knowledge",
        alias="QDRANT_COLLECTION",
    )

    # Model providers (LiteLLM). GEMINI_API_KEY is read by LiteLLM from the environment.
    embedding_model: str = Field(
        default="gemini/text-embedding-004",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimensions: int | None = Field(
        default=768,
        alias="EMBEDDING_DIMENSIONS",
    )
    llm_model: str = Field(
        default="gemini/gemini-2.0-flash",
        alias="LLM_MODEL",
    )
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, value: object) -> Path:
        if value is None or value == "":
            return _DEFAULT_DATA_DIR
        path = Path(str(value))
        if not path.is_absolute():
            path = (_BACKEND_ROOT / path).resolve()
        return path

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        if value is None or value == "":
            return ["http://localhost:3000"]
        if isinstance(value, str):
            origins = [part.strip() for part in value.split(",") if part.strip()]
            return origins or ["http://localhost:3000"]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("CORS_ORIGINS must be a comma-separated string or list")

    @field_validator("qdrant_api_key", "gemini_api_key", mode="before")
    @classmethod
    def empty_str_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("embedding_dimensions", mode="before")
    @classmethod
    def empty_dimensions_to_none(cls, value: object) -> object:
        if value == "" or value is None:
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
