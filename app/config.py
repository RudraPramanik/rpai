"""Application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ — parent of the `app` package
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _BACKEND_ROOT / "data"

# NVIDIA NVCF Parakeet CTC 1.1B ASR (OpenAI-compatible /v1/audio/transcriptions)
_DEFAULT_STT_API_BASE = (
    "https://1598d209-5e27-4d3c-8079-4751568b1081.invocation.api.nvcf.nvidia.com/v1"
)


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

    # Model providers (LiteLLM). Gemini keys cover gemini/* embed + chat.
    embedding_model: str = Field(
        default="gemini/gemini-embedding-001",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimensions: int | None = Field(
        default=768,
        alias="EMBEDDING_DIMENSIONS",
    )
    # Prefer GEMINI_LLM_API_MODEL; LLM_MODEL remains a supported alias.
    llm_model: str = Field(
        default="gemini/gemini-3.1-flash-lite",
        validation_alias=AliasChoices("GEMINI_LLM_API_MODEL", "LLM_MODEL"),
    )
    # Prefer GEMINI_API_KEY; accept GEMINAI_API_KEY / GENAI_API_KEY aliases.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "GEMINAI_API_KEY",
            "GENAI_API_KEY",
        ),
    )

    # Voice STT — NVIDIA NIM / NVCF OpenAI-compatible ASR (not Groq).
    nvidia_nim_api_key: str | None = Field(
        default=None,
        alias="NVIDIA_NIM_API_KEY",
    )
    nvidia_nim_api_base: str | None = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_NIM_API_BASE",
    )
    stt_model: str = Field(
        default="nvidia/parakeet-ctc-1.1b-asr",
        alias="STT_MODEL",
    )
    stt_api_base: str = Field(
        default=_DEFAULT_STT_API_BASE,
        alias="STT_API_BASE",
    )
    stt_language: str | None = Field(default="en-US", alias="STT_LANGUAGE")
    stt_max_upload_bytes: int = Field(
        default=5 * 1024 * 1024,
        alias="STT_MAX_UPLOAD_BYTES",
    )
    stt_send_model_field: bool = Field(
        default=False,
        alias="STT_SEND_MODEL_FIELD",
    )

    def resolved_gemini_api_key(self) -> str | None:
        """Return the configured Gemini/Google AI Studio key for LiteLLM."""
        return self.gemini_api_key

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

    @field_validator(
        "qdrant_api_key",
        "gemini_api_key",
        "nvidia_nim_api_key",
        "nvidia_nim_api_base",
        "stt_language",
        mode="before",
    )
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
