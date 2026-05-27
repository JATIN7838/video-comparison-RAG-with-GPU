from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "video-rag-gpu"
    ENVIRONMENT: str = "dev"
    HOST: str = "0.0.0.0"
    PORT: int = 9000

    RETRIEVAL_API_KEY: str | None = None
    MODEL_BACKEND: str = "local"

    @property
    def reload(self) -> bool:
        return self.ENVIRONMENT.lower().startswith("dev")

    @property
    def use_hf(self) -> bool:
        return self.MODEL_BACKEND.lower() in ("huggingface", "hf")

    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 30
    # Used when MODEL_BACKEND=huggingface (no local model to query the dim from).
    EMBEDDING_DIM: int = 1024

    # Hugging Face Inference Endpoints (TEI). Required when MODEL_BACKEND=huggingface.
    HF_TOKEN: str | None = None
    HF_EMBEDDING_ENDPOINT_URL: str | None = None
    HF_RERANKER_ENDPOINT_URL: str | None = None
    HF_TIMEOUT: float = 60.0
    HF_MAX_RETRIES: int = 4  # retries on cold-start / 503 while endpoint scales up

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATE_K: int = 15
    RERANK_TOP_K: int = 3
    RRF_K: int = 60

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    WINDOW_SIZE: int = 1500

    # Qdrant: provide QDRANT_URL for a remote/server instance,
    # otherwise the client falls back to an embedded on-disk store at QDRANT_PATH.
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_PATH: str = "./qdrant_data"
    QDRANT_COLLECTION: str = "youtube_transcripts"


settings = Settings()
