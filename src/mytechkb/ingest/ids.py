"""Stable identifier helpers for ingestion sources.

Provides deterministic hashing for source IDs, fingerprinting for uploaded
files, and YouTube URL parsing (single videos *and* playlists).
"""

import hashlib
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a single YouTube video ID (11 characters) from various URL shapes.
YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*v=|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

# Matches a YouTube playlist ID from URLs like:
#   https://www.youtube.com/playlist?list=PLxxxxx
#   https://www.youtube.com/watch?v=abc&list=PLxxxxx
YOUTUBE_PLAYLIST_RE = re.compile(
    r"(?:youtube\.com/(?:playlist|watch)\?(?:[^#]*&)?list=)([\w-]+)"
)


# ---------------------------------------------------------------------------
# Generic ID helpers
# ---------------------------------------------------------------------------

def stable_id(*parts: str) -> str:
    """Return a deterministic 40-char hex ID from the given string parts."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:40]


def file_fingerprint(path: Path) -> str:
    """SHA-256 hex digest of a file's contents (streamed in 1 MiB blocks)."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# YouTube single-video helpers
# ---------------------------------------------------------------------------

def youtube_video_id(url: str) -> str | None:
    """Extract the 11-character video ID from a YouTube URL, or ``None``."""
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def is_youtube_url(url: str) -> bool:
    """Return ``True`` if *url* points to a single YouTube video."""
    return youtube_video_id(url) is not None


# ---------------------------------------------------------------------------
# YouTube playlist helpers
# ---------------------------------------------------------------------------

def youtube_playlist_id(url: str) -> str | None:
    """Extract the playlist ID (``list=…`` parameter) from a YouTube URL.

    Recognises both ``/playlist?list=…`` and ``/watch?v=…&list=…`` shapes.
    Returns ``None`` when no playlist ID is found.
    """
    match = YOUTUBE_PLAYLIST_RE.search(url)
    return match.group(1) if match else None


def is_youtube_playlist_url(url: str) -> bool:
    """Return ``True`` when *url* contains a YouTube playlist ID."""
    return youtube_playlist_id(url) is not None
