"""Web-article ingestion via trafilatura with readability fallback.

Minor improvement: trafilatura now uses ``output_format="txt"`` for cleaner
plain-text extraction.
"""

import re

import httpx
import trafilatura
from readability import Document as ReadabilityDocument

from mytechkb.ingest.ids import stable_id
from mytechkb.models import ExtractedDocument, utc_now

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    """Crude HTML-to-text: strip all tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", html)
    return re.sub(r"\s+", " ", text).strip()


def load_web(url: str) -> list[ExtractedDocument]:
    """Extract article text from *url*.

    1. Try **trafilatura** first (with ``output_format="txt"`` for cleaner
       plain-text output).
    2. Fall back to **readability-lxml** + manual HTML stripping if
       trafilatura fails or returns too little text.
    """
    downloaded = trafilatura.fetch_url(url)
    extracted = None
    title = url

    if downloaded:
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            output_format="txt",
        )
        metadata = trafilatura.extract_metadata(downloaded)
        if metadata and metadata.title:
            title = metadata.title

    # Readability fallback when trafilatura yields nothing useful.
    if not extracted:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "MyTechKB/0.1 (+local-rag)"},
        )
        response.raise_for_status()
        readable = ReadabilityDocument(response.text)
        title = readable.short_title() or title
        extracted = _strip_html(readable.summary() or "")

    text = (extracted or "").strip()
    if len(text) < 40:
        raise ValueError("Could not extract enough article text from that URL")

    ingested_at = utc_now()
    source_id = stable_id("web", url)
    return [
        ExtractedDocument(
            source_id=source_id,
            source_type="web",
            title=title,
            text=text,
            locator=url,
            metadata={
                "source_id": source_id,
                "source_type": "web",
                "title": title,
                "locator": url,
                "ingested_at": ingested_at,
            },
        )
    ]
