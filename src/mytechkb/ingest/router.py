"""Ingestion router — dispatches URLs and files to the right loader.

Supports:
* Single YouTube videos (`load_youtube`)
* YouTube playlists (`ingest_playlist` generator with per-video progress)
* Web articles (`load_web`)
* PDFs (`load_pdf`)
* Images via OCR (`load_image`)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

from mytechkb.config import Settings, get_settings
from mytechkb.ingest.chunking import chunk_documents
from mytechkb.ingest.ids import is_youtube_playlist_url, is_youtube_url
from mytechkb.ingest.image import load_image
from mytechkb.ingest.pdf import load_pdf
from mytechkb.ingest.web import load_web
from mytechkb.ingest.youtube import load_youtube, load_youtube_playlist
from mytechkb.models import SourceInfo
from mytechkb.store.base import KnowledgeStore

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}


# ---------------------------------------------------------------------------
# Single-URL ingestion
# ---------------------------------------------------------------------------

def ingest_url(
    url: str, store: KnowledgeStore, settings: Settings | None = None
) -> SourceInfo:
    """Ingest a single URL (YouTube video or web article) into *store*."""
    settings = settings or get_settings()
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("URL is required")
    if is_youtube_url(cleaned):
        documents = load_youtube(cleaned, proxy=settings.youtube_proxy)
    elif is_youtube_playlist_url(cleaned):
        # Pure playlist URL passed to single-URL endpoint — ingest all videos
        # and return the first indexed SourceInfo.
        sources: list[SourceInfo] = []
        for event in ingest_playlist(cleaned, store, settings):
            if event.get("event") == "complete":
                sources = event.get("sources", [])
            elif event.get("event") == "error":
                raise ValueError(event.get("message", "Playlist ingestion failed"))
        if not sources:
            raise ValueError("No videos could be indexed from that playlist")
        return sources[0]
    else:
        documents = load_web(cleaned)
    return _persist(documents, store, settings)


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------

def ingest_file(
    path: Path,
    original_name: str,
    store: KnowledgeStore,
    settings: Settings | None = None,
) -> SourceInfo:
    """Ingest an uploaded file (PDF or image) into *store*."""
    settings = settings or get_settings()
    suffix = Path(original_name).suffix.lower() or path.suffix.lower()
    if suffix == ".pdf":
        documents = load_pdf(path, original_name)
    elif suffix in IMAGE_SUFFIXES:
        documents = load_image(path, original_name)
    else:
        raise ValueError(f"Unsupported file type: {suffix or original_name}")
    return _persist(documents, store, settings)


# ---------------------------------------------------------------------------
# Playlist ingestion (generator with real-time progress events)
# ---------------------------------------------------------------------------

def ingest_playlist(
    url: str,
    store: KnowledgeStore,
    settings: Settings | None = None,
) -> Generator[dict, None, None]:
    """Ingest every video in a YouTube playlist, persisting each video as it loads.

    Yields progress dicts:
    * ``{"event": "progress", "video_index": i, "total": n, "title": …}``
    * ``{"event": "video_error", "video_index": i, "title": …, "error": …}``
    * ``{"event": "complete", "sources": [SourceInfo, …]}``
    * ``{"event": "error", "message": …}``
    """
    settings = settings or get_settings()
    cleaned = url.strip()
    if not cleaned:
        yield {"event": "error", "message": "Playlist URL is required"}
        return

    try:
        playlist_gen = load_youtube_playlist(cleaned, proxy=settings.youtube_proxy)
        sources: list[SourceInfo] = []
        persisted_ids: set[str] = set()

        try:
            while True:
                event = next(playlist_gen)
                ev_type = event.get("event")

                if ev_type == "video_loaded":
                    docs = event.get("documents") or []
                    if docs:
                        try:
                            info = _persist(docs, store, settings)
                            sources.append(info)
                            persisted_ids.add(info.source_id)
                        except Exception as exc:
                            logger.warning(
                                "Failed to persist source %s: %s",
                                docs[0].source_id,
                                exc,
                            )
                            yield {
                                "event": "video_error",
                                "video_index": event.get("video_index", -1),
                                "title": docs[0].title,
                                "error": str(exc),
                            }
                else:
                    yield event
        except StopIteration as stop:
            all_docs: list = stop.value or []

        # Fallback: if a custom/mocked generator returned docs via StopIteration
        # without emitting "video_loaded" events, persist any remaining ones now.
        docs_by_source: dict[str, list] = {}
        for doc in all_docs:
            if doc.source_id not in persisted_ids:
                docs_by_source.setdefault(doc.source_id, []).append(doc)

        for source_docs in docs_by_source.values():
            try:
                info = _persist(source_docs, store, settings)
                sources.append(info)
            except Exception as exc:
                logger.warning(
                    "Failed to persist source %s: %s",
                    source_docs[0].source_id,
                    exc,
                )
                yield {
                    "event": "video_error",
                    "video_index": -1,
                    "title": source_docs[0].title,
                    "error": str(exc),
                }

        yield {"event": "complete", "sources": sources}

    except Exception as exc:
        logger.exception("Playlist ingestion failed")
        yield {"event": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Shared persistence helper
# ---------------------------------------------------------------------------

def _persist(
    documents: list, store: KnowledgeStore, settings: Settings
) -> SourceInfo:
    """Chunk *documents*, upsert into *store*, and return a SourceInfo."""
    source_id = documents[0].source_id
    if store.has_source(source_id):
        for existing in store.list_sources():
            if existing.source_id == source_id:
                return existing
    store.delete_source(source_id)
    chunks = chunk_documents(documents, settings)
    if not chunks:
        raise ValueError("Nothing to index after chunking")
    store.upsert_chunks(chunks)
    first = documents[0]
    return SourceInfo(
        source_id=source_id,
        source_type=first.source_type,
        title=first.title,
        locator=first.locator,
        chunk_count=len(chunks),
        ingested_at=str(first.metadata.get("ingested_at", "")),
    )
