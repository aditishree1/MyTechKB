from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from mytechkb.ingest.ids import file_fingerprint, stable_id
from mytechkb.models import ExtractedDocument, utc_now


def load_image(path: Path, original_name: str) -> list[ExtractedDocument]:
    fingerprint = file_fingerprint(path)
    source_id = stable_id("image", fingerprint)
    ingested_at = utc_now()
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("L")
        text = pytesseract.image_to_string(image).strip()
    if len(text) < 8:
        raise ValueError(
            "Image OCR found almost no text. Install Tesseract and use a clearer image."
        )
    return [
        ExtractedDocument(
            source_id=source_id,
            source_type="image",
            title=original_name,
            text=text,
            locator=original_name,
            metadata={
                "source_id": source_id,
                "source_type": "image",
                "title": original_name,
                "locator": original_name,
                "ingested_at": ingested_at,
            },
        )
    ]
