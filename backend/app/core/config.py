from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "backend" / "data"


class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    INDEX_PATH: Path = DATA / "workspace.sqlite3"
    CHROMA_PERSIST_DIR: Path = DATA / "chroma_db"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


settings = Settings()
