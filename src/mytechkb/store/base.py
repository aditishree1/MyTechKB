from typing import Protocol

from mytechkb.models import Chunk, SourceInfo


class KnowledgeStore(Protocol):
    def upsert_chunks(self, chunks: list[Chunk]) -> int: ...

    def similarity_search(
        self, query: str, k: int, source_ids: list[str] | None = None
    ) -> list[Chunk]: ...

    def get_all_chunks(
        self, source_ids: list[str] | None = None
    ) -> list[Chunk]: ...

    def delete_source(self, source_id: str) -> int: ...

    def list_sources(self) -> list[SourceInfo]: ...

    def has_source(self, source_id: str) -> bool: ...
