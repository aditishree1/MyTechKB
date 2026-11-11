"""Azure Cosmos DB adapter (not wired for v1 runtime).

When you opt into Azure, map this store onto Cosmos DB for NoSQL with
vector search:

- Database: ``Settings.cosmos_database`` (one KB).
- Container: ``Settings.cosmos_container`` with partition key ``/source_id``.
- Item shape::

    {
      "id": "<chunk_id>",
      "source_id": "...",
      "text": "...",
      "embedding": [float, ...],
      "source_type": "youtube|web|pdf|image",
      "title": "...",
      "locator": "url or filename",
      "page": 1,
      "timestamp": "00:01:20",
      "ingested_at": "ISO-8601"
    }

- Enable a vector embedding policy on ``embedding`` (cosine, dimensions
  matching ``nomic-embed-text``, typically 768).
- ``upsert_chunks``: bulk upsert items (RU-aware batches).
- ``similarity_search``: ``VectorDistance`` ORDER BY query, TOP k.
- ``delete_source``: delete by partition key ``source_id``.
- ``list_sources``: DISTINCT ``source_id`` with aggregated metadata.

Do not instantiate this class until Azure credentials and a vector-enabled
container exist. The factory raises if ``VECTOR_STORE=cosmos``.
"""

from mytechkb.models import Chunk, SourceInfo


class CosmosKnowledgeStore:
    def __init__(self, endpoint: str, key: str, database: str, container: str) -> None:
        del endpoint, key, database, container
        raise RuntimeError(
            "Cosmos DB is a later swap. Set VECTOR_STORE=chroma for local $0 use. "
            "See store/cosmos_store.py for the item shape and vector policy."
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        raise NotImplementedError

    def similarity_search(
        self, query: str, k: int, source_ids: list[str] | None = None
    ) -> list[Chunk]:
        raise NotImplementedError

    def get_all_chunks(
        self, source_ids: list[str] | None = None
    ) -> list[Chunk]:
        raise NotImplementedError

    def delete_source(self, source_id: str) -> int:
        raise NotImplementedError

    def list_sources(self) -> list[SourceInfo]:
        raise NotImplementedError

    def has_source(self, source_id: str) -> bool:
        raise NotImplementedError
