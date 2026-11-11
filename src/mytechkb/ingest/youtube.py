"""YouTube ingestion — single videos *and* playlists.

Key features:
* Real video titles via the YouTube oEmbed API (zero API key required).
* Playlist support: normalizes playlist URLs and parses ``ytInitialData`` to
  enumerate all videos with their titles.
* ``video_deep_link`` metadata for clickable timestamped YouTube links.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator

import httpx
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable
except ImportError:  # pragma: no cover - version-dependent exports
    try:
        from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
    except ImportError:
        TranscriptsDisabled = VideoUnavailable = Exception

from mytechkb.ingest.ids import stable_id, youtube_playlist_id, youtube_video_id
from mytechkb.models import ExtractedDocument, utc_now

logger = logging.getLogger(__name__)

# Timeout for lightweight HTTP requests (oEmbed, playlist page).
_HTTP_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Transcript helper — compatible with multiple youtube-transcript-api versions
# ---------------------------------------------------------------------------

def _snippet_text(item: object) -> str:
    """Extract the ``text`` field from a transcript snippet."""
    if hasattr(item, "text"):
        return str(getattr(item, "text"))
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)


def _snippet_start(item: object) -> float:
    """Extract the ``start`` (seconds) field from a transcript snippet."""
    if hasattr(item, "start"):
        return float(getattr(item, "start"))
    if isinstance(item, dict):
        return float(item.get("start", 0))
    return 0.0


def _snippet_duration(item: object) -> float:
    """Extract the ``duration`` (seconds) field from a transcript snippet."""
    if hasattr(item, "duration"):
        return float(getattr(item, "duration"))
    if isinstance(item, dict):
        return float(item.get("duration", 0))
    return 0.0


def _fetch_transcript(video_id: str) -> list[object]:
    """Fetch the transcript for *video_id*, supporting API v1 & v2."""
    api = YouTubeTranscriptApi()
    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id)
        return list(fetched)
    get_transcript = getattr(YouTubeTranscriptApi, "get_transcript", None)
    if get_transcript is None:
        raise RuntimeError(
            "youtube-transcript-api does not expose fetch or get_transcript"
        )
    return list(get_transcript(video_id))


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS`` (or ``MM:SS`` when < 1 hour)."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


_format_ts = format_timestamp


# ---------------------------------------------------------------------------
# Video title via oEmbed
# ---------------------------------------------------------------------------

def fetch_video_title(video_id: str) -> str:
    """Return the real title for *video_id* using the YouTube oEmbed API.

    Falls back to ``"YouTube {video_id}"`` on any network or parsing error.
    """
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        response = httpx.get(oembed_url, timeout=_HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        title = data.get("title", "").strip()
        if title:
            return title
    except Exception:
        logger.debug("oEmbed lookup failed for %s; using fallback title", video_id)
    return f"YouTube {video_id}"


# ---------------------------------------------------------------------------
# Playlist video enumeration
# ---------------------------------------------------------------------------

# Regex to locate the ytInitialData JSON blob embedded in the playlist page.
_YT_INITIAL_DATA_RE = re.compile(
    r"(?:var\s+ytInitialData|window\[\"ytInitialData\"\])\s*=\s*(\{.*?\})\s*;",
    re.DOTALL,
)


def _extract_yt_initial_data(html: str) -> dict:
    """Parse the ``ytInitialData`` JSON object out of raw YouTube HTML."""
    match = _YT_INITIAL_DATA_RE.search(html)
    if not match:
        raise ValueError("Could not locate ytInitialData in the playlist page")
    return json.loads(match.group(1))


def _walk_playlist_renderers(data: dict) -> list[dict]:
    """Walk the ytInitialData JSON tree and return video info dicts.

    Each returned dict has ``{"id": "<video_id>", "title": "<title>"}``.
    Supports both ``playlistVideoRenderer`` (on /playlist pages) and
    ``playlistPanelVideoRenderer`` (on /watch?...&list=... pages).
    """
    results: list[dict] = []
    seen: set[str] = set()

    def _extract_title(title_obj: object, fallback: str) -> str:
        if isinstance(title_obj, dict):
            runs = title_obj.get("runs", [])
            text = (
                "".join(r.get("text", "") for r in runs)
                if runs
                else title_obj.get("simpleText", "")
            )
            return str(text).strip() or fallback
        return str(title_obj).strip() or fallback

    def _recurse(obj: object) -> None:
        if isinstance(obj, dict):
            # Modern YouTube playlist item format (lockupViewModel)
            if "lockupViewModel" in obj:
                vm = obj["lockupViewModel"]
                vid = vm.get("contentId", "")
                if not vid:
                    cmd = (
                        vm.get("rendererContext", {})
                        .get("commandContext", {})
                        .get("onTap", {})
                        .get("innertubeCommand", {})
                        .get("watchEndpoint", {})
                    )
                    vid = cmd.get("videoId", "")
                if vid and vid not in seen and len(vid) == 11:
                    seen.add(vid)
                    title = (
                        vm.get("metadata", {})
                        .get("lockupMetadataViewModel", {})
                        .get("title", {})
                        .get("content", "")
                    )
                    if not title:
                        title = (
                            vm.get("rendererContext", {})
                            .get("accessibilityContext", {})
                            .get("label", f"YouTube {vid}")
                        )
                    results.append({"id": vid, "title": title.strip() or f"YouTube {vid}"})
                return

            for key in ("playlistVideoRenderer", "playlistPanelVideoRenderer"):
                if key in obj:
                    renderer = obj[key]
                    vid = renderer.get("videoId", "")
                    if vid and vid not in seen:
                        seen.add(vid)
                        title = _extract_title(
                            renderer.get("title", {}), f"YouTube {vid}"
                        )
                        results.append({"id": vid, "title": title})
                    return
            for value in obj.values():
                _recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)

    _recurse(data)
    return results


def fetch_playlist_video_ids(playlist_url: str) -> list[dict]:
    """Fetch the playlist page and return a list of video info dicts.

    Returns ``[{"id": "abc123", "title": "Video Title"}, ...]``.

    Normalises ``/watch?v=...&list=...`` URLs to the canonical
    ``/playlist?list=...`` URL so YouTube returns the full playlist view.
    """
    playlist_id = youtube_playlist_id(playlist_url)
    target_url = (
        f"https://www.youtube.com/playlist?list={playlist_id}"
        if playlist_id
        else playlist_url
    )

    response = httpx.get(
        target_url,
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    response.raise_for_status()
    data = _extract_yt_initial_data(response.text)
    videos = _walk_playlist_renderers(data)
    if not videos:
        raise ValueError(
            "No videos found in the playlist. Make sure the playlist is public or unlisted."
        )
    return videos


# ---------------------------------------------------------------------------
# Single-video loader
# ---------------------------------------------------------------------------

def load_youtube(url: str, title_override: str | None = None) -> list[ExtractedDocument]:
    """Load transcript snippets for a single YouTube video.

    Each transcript segment becomes one :class:`ExtractedDocument`.  Metadata
    includes ``start_seconds``, ``duration``, a human-readable ``timestamp``,
    and a clickable ``video_deep_link`` for the segment.
    """
    video_id = youtube_video_id(url)
    if not video_id:
        raise ValueError("Not a YouTube URL")

    try:
        snippets = _fetch_transcript(video_id)
    except (TranscriptsDisabled, VideoUnavailable) as exc:
        raise ValueError(f"Could not load YouTube transcript: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not load YouTube transcript: {exc}") from exc

    if not snippets:
        raise ValueError("YouTube video has an empty transcript")

    # Fetch the real video title (falls back to "YouTube {id}" on error).
    title = (title_override or "").strip() or fetch_video_title(video_id)

    ingested_at = utc_now()
    source_id = stable_id("youtube", video_id)
    locator = f"https://www.youtube.com/watch?v={video_id}"

    documents: list[ExtractedDocument] = []
    for item in snippets:
        text = _snippet_text(item).strip()
        if not text:
            continue
        start = _snippet_start(item)
        duration = _snippet_duration(item)
        video_deep_link = f"https://youtu.be/{video_id}?t={int(start)}"
        documents.append(
            ExtractedDocument(
                source_id=source_id,
                source_type="youtube",
                title=title,
                text=text,
                locator=locator,
                metadata={
                    "source_id": source_id,
                    "source_type": "youtube",
                    "title": title,
                    "locator": locator,
                    "timestamp": format_timestamp(start),
                    "start_seconds": start,
                    "duration": duration,
                    "video_deep_link": video_deep_link,
                    "ingested_at": ingested_at,
                },
            )
        )

    if not documents:
        raise ValueError("YouTube transcript contained no text")
    return documents


# ---------------------------------------------------------------------------
# Playlist loader (generator)
# ---------------------------------------------------------------------------

def load_youtube_playlist(
    playlist_url: str,
) -> Generator[dict, None, list[ExtractedDocument]]:
    """Load every video in a YouTube playlist.

    Yields progress/error dicts and also yields ``video_docs`` so callers can
    persist videos incrementally as they arrive, while still returning the
    combined list of :class:`ExtractedDocument` objects at the end.

    Yielded dicts:
    * ``{"event": "progress", "video_index": i, "total": n, "title": …, "video_id": …}``
    * ``{"event": "video_loaded", "video_index": i, "total": n, "title": …, "documents": […]}``
    * ``{"event": "video_error", "video_index": i, "title": …, "error": …}``
    """
    videos = fetch_playlist_video_ids(playlist_url)
    total = len(videos)
    all_docs: list[ExtractedDocument] = []

    for i, info in enumerate(videos, start=1):
        vid_id = info["id"]
        title = info["title"]

        # Signal progress (1-based index) to the caller.
        yield {
            "event": "progress",
            "video_index": i,
            "total": total,
            "title": title,
            "video_id": vid_id,
        }

        try:
            docs = load_youtube(
                f"https://www.youtube.com/watch?v={vid_id}",
                title_override=title if not title.startswith("YouTube ") else None,
            )
            all_docs.extend(docs)
            yield {
                "event": "video_loaded",
                "video_index": i,
                "total": total,
                "title": docs[0].title if docs else title,
                "documents": docs,
            }
        except Exception as exc:
            logger.warning("Skipping video %s (%s): %s", vid_id, title, exc)
            yield {
                "event": "video_error",
                "video_index": i,
                "title": title,
                "error": str(exc),
            }

    return all_docs
