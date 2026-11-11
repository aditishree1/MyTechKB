"""Document chunking strategies for each source type.

Improvements:
* YouTube chunks are built on exact transcript segment boundaries with rolling
  overlap, preserving ``start_seconds``, ``end_seconds``, time-range
  ``timestamp`` (e.g. ``01:20–02:55``), and clickable ``video_deep_link``.
* Web-article chunking uses richer markdown-aware separators for better
  section and sentence-level splits.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mytechkb.config import Settings
from mytechkb.ingest.ids import stable_id, youtube_video_id
from mytechkb.ingest.youtube import format_timestamp
from mytechkb.models import Chunk, ExtractedDocument

# Markdown-aware separators used for web articles and general text.
# Splits at heading boundaries first, then paragraphs, lines, sentences, words.
WEB_SEPARATORS = ["\n## ", "\n### ", "\n# ", "\n\n", "\n", ". ", " ", ""]

# Generic heading separators (kept for non-web, non-youtube sources).
HEADING_SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Splitter factory
# ---------------------------------------------------------------------------

def _splitter(
    settings: Settings,
    chunk_size: int | None = None,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build a ``RecursiveCharacterTextSplitter`` with the given settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=separators or HEADING_SEPARATORS,
        length_function=len,
    )


# ---------------------------------------------------------------------------
# Metadata sanitiser
# ---------------------------------------------------------------------------

def _clean_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    """Return a copy of *metadata* with only JSON-safe scalar values."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[ExtractedDocument], settings: Settings
) -> list[Chunk]:
    """Chunk *documents* using the strategy appropriate for their source type."""
    if not documents:
        return []

    source_type = documents[0].source_type

    if source_type == "youtube":
        return _chunk_youtube(documents, settings)

    if source_type == "web":
        splitter = _splitter(settings, separators=WEB_SEPARATORS)
        return _split_extracted(documents, splitter)

    if source_type == "image":
        splitter = _splitter(settings, chunk_size=min(400, settings.chunk_size))
        return _split_extracted(documents, splitter)

    # Default path for PDFs and anything else.
    splitter = _splitter(settings)
    return _split_extracted(documents, splitter)


# ---------------------------------------------------------------------------
# Generic text splitting
# ---------------------------------------------------------------------------

def _split_extracted(
    documents: list[ExtractedDocument],
    splitter: RecursiveCharacterTextSplitter,
) -> list[Chunk]:
    """Split a sequence of documents using *splitter* and return chunks."""
    chunks: list[Chunk] = []
    for index, document in enumerate(documents):
        pieces = splitter.split_text(document.text)
        for piece_index, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            metadata = _clean_metadata(document.metadata)
            metadata["chunk_index"] = piece_index
            chunk_id = stable_id(
                document.source_id, str(index), str(piece_index), text[:80]
            )
            chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
    return chunks


# ---------------------------------------------------------------------------
# YouTube-specific chunking (segment-aligned time & character windows)
# ---------------------------------------------------------------------------

def _chunk_youtube(
    documents: list[ExtractedDocument], settings: Settings
) -> list[Chunk]:
    """Merge transcript snippets into segment-aligned windows with exact timestamps.

    Each chunk tracks:
    * ``start_seconds`` & ``end_seconds``
    * ``timestamp`` formatted as ``MM:SS`` or ``MM:SS–MM:SS``
    * ``video_deep_link`` pointing to ``https://youtu.be/<id>?t=<start>``
    """
    window_sec = settings.youtube_window_seconds
    max_chars = max(settings.chunk_size, 900)
    overlap_chars = max(settings.chunk_overlap, 100)

    groups: list[list[ExtractedDocument]] = []
    current: list[ExtractedDocument] = []
    current_chars = 0
    window_start: float | None = None

    for doc in documents:
        text = doc.text.strip()
        if not text:
            continue
        start = float(doc.metadata.get("start_seconds", 0))
        if window_start is None:
            window_start = start

        elapsed = start - window_start
        would_exceed_time = elapsed >= window_sec and len(current) > 0
        would_exceed_chars = (current_chars + len(text) + 1) > max_chars and len(current) > 0

        if would_exceed_time or would_exceed_chars:
            groups.append(current)
            # Carry over trailing snippets for overlap so context isn't cut mid-thought
            overlap_group: list[ExtractedDocument] = []
            overlap_len = 0
            for prev in reversed(current):
                prev_len = len(prev.text.strip()) + 1
                if overlap_len + prev_len > overlap_chars and overlap_group:
                    break
                overlap_group.insert(0, prev)
                overlap_len += prev_len

            current = overlap_group + [doc]
            current_chars = sum(len(x.text.strip()) + 1 for x in current)
            window_start = float(current[0].metadata.get("start_seconds", start))
        else:
            current.append(doc)
            current_chars += len(text) + 1

    if current:
        groups.append(current)

    merged: list[ExtractedDocument] = []
    for group in groups:
        first = group[0]
        last = group[-1]
        text = " ".join(item.text.strip() for item in group if item.text.strip())

        first_start = float(first.metadata.get("start_seconds", 0))
        last_start = float(last.metadata.get("start_seconds", first_start))
        last_dur = float(last.metadata.get("duration", 0))
        end_seconds = max(first_start, last_start + last_dur)

        start_ts = format_timestamp(first_start)
        end_ts = format_timestamp(end_seconds)
        timestamp_str = f"{start_ts}–{end_ts}" if end_ts != start_ts else start_ts

        vid_id = youtube_video_id(first.locator) or ""
        deep_link = (
            f"https://youtu.be/{vid_id}?t={int(first_start)}"
            if vid_id
            else first.locator
        )

        merged_meta = {k: v for k, v in first.metadata.items()}
        merged_meta["start_seconds"] = first_start
        merged_meta["end_seconds"] = end_seconds
        merged_meta["timestamp"] = timestamp_str
        merged_meta["video_deep_link"] = deep_link

        merged.append(
            ExtractedDocument(
                source_id=first.source_id,
                source_type=first.source_type,
                title=first.title,
                text=text,
                locator=first.locator,
                metadata=merged_meta,
            )
        )

    # Split any still-oversized merged documents (e.g. single giant transcript block)
    splitter = _splitter(settings, chunk_size=max(max_chars, 1000))
    return _split_extracted(merged, splitter)
