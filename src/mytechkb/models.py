from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExtractedDocument:
    source_id: str
    source_type: str
    title: str
    text: str
    locator: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass
class SourceInfo:
    source_id: str
    source_type: str
    title: str
    locator: str
    chunk_count: int
    ingested_at: str
