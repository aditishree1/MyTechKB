from functools import lru_cache

import httpx
from langchain_core.embeddings import Embeddings

from mytechkb.config import Settings, get_settings
from mytechkb.store.base import KnowledgeStore
from mytechkb.store.chroma_store import ChromaKnowledgeStore
from mytechkb.store.cosmos_store import CosmosKnowledgeStore


class GeminiRestEmbeddings(Embeddings):
    """Lightweight LangChain Embeddings adapter for the Google Gemini REST API."""

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self._model}:embedContent"
        response = httpx.post(
            url,
            params={"key": self._api_key},
            json={"model": self._model, "content": {"parts": [{"text": text}]}},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]


def build_embeddings(settings: Settings):
    """Build embeddings model based on configured provider."""
    if settings.llm_provider.strip().lower() == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=settings.gemini_api_key,
            )
        except ImportError:
            return GeminiRestEmbeddings(
                model=settings.gemini_embedding_model,
                api_key=settings.gemini_api_key,
            )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


_SHARED_EMBEDDINGS: dict[str, Embeddings] = {}


def get_shared_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    cache_key = f"{settings.llm_provider}:{settings.gemini_embedding_model}:{settings.gemini_api_key}:{settings.embedding_model}:{settings.ollama_base_url}"
    if cache_key not in _SHARED_EMBEDDINGS:
        _SHARED_EMBEDDINGS[cache_key] = build_embeddings(settings)
    return _SHARED_EMBEDDINGS[cache_key]


def build_store(
    collection_name: str | None = None, settings: Settings | None = None
) -> KnowledgeStore:
    settings = settings or get_settings()
    coll = collection_name or settings.chroma_collection
    backend = settings.vector_store.strip().lower()
    if backend == "chroma":
        return ChromaKnowledgeStore(
            persist_dir=str(settings.chroma_dir),
            collection_name=coll,
            embeddings=get_shared_embeddings(settings),
        )
    if backend == "cosmos":
        return CosmosKnowledgeStore(
            endpoint=settings.cosmos_endpoint,
            key=settings.cosmos_key,
            database=settings.cosmos_database,
            container=settings.cosmos_container,
        )
    raise ValueError(
        f"Unknown VECTOR_STORE={settings.vector_store!r}. Use chroma or cosmos."
    )


def get_tenant_store(
    collection_name: str, settings: Settings | None = None
) -> KnowledgeStore:
    """Return an isolated vector store instance scoped to the specified tenant collection."""
    return build_store(collection_name=collection_name, settings=settings)


@lru_cache(maxsize=1)
def get_store() -> KnowledgeStore:
    return build_store()
