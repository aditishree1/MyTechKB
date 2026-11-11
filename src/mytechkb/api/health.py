"""Health-check helper — reports provider, vector store, and model readiness."""

import httpx

from mytechkb.config import Settings


def check_health(settings: Settings) -> dict:
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        gemini_ok = bool(settings.gemini_api_key)
        return {
            "status": "ok" if gemini_ok else "degraded",
            "provider": "gemini",
            "vector_store": settings.vector_store,
            "chroma_writable": settings.chroma_dir.exists() and settings.chroma_dir.is_dir(),
            "ollama": {"reachable": False, "llm_pulled": False, "embedding_pulled": False, "models": []},
            "gemini": {"configured": gemini_ok},
            "models": {
                "llm": settings.gemini_llm_model,
                "embedding": settings.gemini_embedding_model,
            },
        }

    # Default: Ollama
    ollama = _ollama_status(settings)
    chroma_ok = settings.chroma_dir.exists() and settings.chroma_dir.is_dir()
    status = "ok" if ollama["reachable"] and chroma_ok else "degraded"
    return {
        "status": status,
        "provider": "ollama",
        "vector_store": settings.vector_store,
        "chroma_writable": chroma_ok,
        "ollama": ollama,
        "models": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model,
        },
    }


def _ollama_status(settings: Settings) -> dict:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=3.0)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", [])]
        llm_ready = any(settings.llm_model in name for name in names)
        embed_ready = any(settings.embedding_model in name for name in names)
        return {
            "reachable": True,
            "llm_pulled": llm_ready,
            "embedding_pulled": embed_ready,
            "models": names,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "llm_pulled": False,
            "embedding_pulled": False,
            "models": [],
            "error": str(exc),
        }
