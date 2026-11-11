from pathlib import Path

import pymupdf

from mytechkb.ingest.ids import file_fingerprint, stable_id
from mytechkb.models import ExtractedDocument, utc_now


def load_pdf(path: Path, original_name: str) -> list[ExtractedDocument]:
    fingerprint = file_fingerprint(path)
    source_id = stable_id("pdf", fingerprint)
    ingested_at = utc_now()
    documents: list[ExtractedDocument] = []
    with pymupdf.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or original_name
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            documents.append(
                ExtractedDocument(
                    source_id=source_id,
                    source_type="pdf",
                    title=title,
                    text=text,
                    locator=original_name,
                    metadata={
                        "source_id": source_id,
                        "source_type": "pdf",
                        "title": title,
                        "locator": original_name,
                        "page": page_index,
                        "ingested_at": ingested_at,
                    },
                )
            )
    if not documents:
        raise ValueError(
            "PDF had no extractable text. Scanned PDFs need OCR (install Tesseract "
            "and convert, or use the optional heavy OCR image later)."
        )
    return documents
