"""
AutoDS Core Configuration
Pydantic Settings with support for environment variables, validation, and sensible defaults.
"""

from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    BASE_DIR: Path = BASE_DIR

    # App Info
    PROJECT_NAME: str = "AutoDS — Autonomous Data Science Platform"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # AI & Gemini API
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./autods.db"
    DATABASE_SYNC_URL: str = "sqlite:///./autods.db"

    # MLflow (Database-backed store: SQLite for local dev, PostgreSQL / server in production)
    MLFLOW_TRACKING_URI: str = "sqlite:///./data/mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "AutoDS_Default"

    # Security
    SECRET_KEY: str = "autods-insecure-development-secret-key-change-in-prod-32chars"
    BACKEND_CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ]

    # Storage Paths
    STORAGE_DIR: str = str(BASE_DIR / "data")
    REPORTS_DIR: str = str(BASE_DIR / "reports")
    EXPERIMENTS_DIR: str = str(BASE_DIR / "experiments")
    
    # Limits & Guards
    MAX_UPLOAD_SIZE_MB: int = 100
    SAFE_SQL_ROW_LIMIT: int = 1000
    MAX_CV_FOLDS: int = 5
    DEFAULT_RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip() and self.GEMINI_API_KEY != "your-gemini-api-key-here")

    @property
    def data_raw_dir(self) -> Path:
        p = Path(self.STORAGE_DIR) / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_processed_dir(self) -> Path:
        p = Path(self.STORAGE_DIR) / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_artifacts_dir(self) -> Path:
        p = Path(self.REPORTS_DIR) / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def experiments_artifacts_dir(self) -> Path:
        p = Path(self.EXPERIMENTS_DIR) / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
