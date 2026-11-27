"""Configuration loader for ResearchAI.

Reads settings from `.env` and environment variables without hardcoding API keys.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent

_PLACEHOLDER_KEYS = {
    "",
    "your_gemini_api_key_here",
    "your_openai_api_key_here",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini/gemini-2.5-flash"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    tavily_api_key: str = ""

    default_num_sources: int = 5
    default_depth: str = "Intermediate"
    output_dir: str = "outputs"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        if not path.is_absolute():
            path = BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_valid_llm_key(
        self, provider: str | None = None, api_key_override: str | None = None
    ) -> bool:
        """Return True if a non-placeholder API key is available for the provider."""
        if api_key_override and api_key_override.strip() not in _PLACEHOLDER_KEYS:
            return True
        prov = (provider or self.llm_provider).lower().strip()
        if prov == "gemini":
            return self.gemini_api_key.strip() not in _PLACEHOLDER_KEYS
        if prov == "openai":
            return self.openai_api_key.strip() not in _PLACEHOLDER_KEYS
        return False


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
