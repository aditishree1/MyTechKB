from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    data_dir: Path = Path("./data")

    # LLM provider: "ollama" or "gemini"
    llm_provider: str = "ollama"

    # Ollama settings
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    # Google Gemini settings
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # Vector store
    vector_store: str = "chroma"
    chroma_collection: str = "mytechkb"

    # Chunking & retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    youtube_window_seconds: float = 100.0
    retrieve_k: int = 6
    enable_hybrid_search: bool = True
    enable_multi_query: bool = True
    rrf_k: int = 60

    # Cosmos DB (placeholder)
    cosmos_endpoint: str = ""
    cosmos_key: str = ""
    cosmos_database: str = "mytechkb"
    cosmos_container: str = "chunks"

    # Multi-tenancy & Authentication
    session_secret: str = "mytechkb-super-secret-key-change-in-production"

    # Cloud Deployment & Proxy Support
    youtube_proxy: str = ""
    render_external_url: str = ""

    @property
    def auth_db_path(self) -> Path:
        return self.data_dir / "auth.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
