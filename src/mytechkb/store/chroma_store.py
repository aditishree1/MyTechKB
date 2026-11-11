from collections import defaultdict
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from mytechkb.models import Chunk, SourceInfo


class ChromaKnowledgeStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embeddings: Any,
    ) -> None:
        self._embeddings = embeddings
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        vectors = self._embeddings.embed_documents(texts)
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
        )
        return len(chunks)

    def similarity_search(
        self, query: str, k: int, source_ids: list[str] | None = None
    ) -> list[Chunk]:
        count = self._collection.count()
        if count == 0:
            return []

        # Build optional where-filter for source-scoped retrieval
        where: dict[str, Any] | None = None
        if source_ids:
            if len(source_ids) == 1:
                where = {"source_id": source_ids[0]}
            else:
                where = {"source_id": {"$in": source_ids}}

        query_vec = self._embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, count),
            where=where,
            include=["documents", "metadatas"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        chunks: list[Chunk] = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            chunks.append(
                Chunk(id=chunk_id, text=text or "", metadata=dict(metadata or {}))
            )
        return chunks

    def get_all_chunks(self, source_ids: list[str] | None = None) -> list[Chunk]:
        where: dict[str, Any] | None = None
        if source_ids:
            if len(source_ids) == 1:
                where = {"source_id": source_ids[0]}
            else:
                where = {"source_id": {"$in": source_ids}}
        data = self._collection.get(where=where, include=["documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            Chunk(id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(ids, documents, metadatas)
        ]

    def delete_source(self, source_id: str) -> int:
        existing = self._collection.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def has_source(self, source_id: str) -> bool:
        existing = self._collection.get(
            where={"source_id": source_id},
            include=["metadatas"],
            limit=1,
        )
        return bool(existing.get("ids"))

    def list_sources(self) -> list[SourceInfo]:
        data = self._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        grouped: dict[str, list[dict]] = defaultdict(list)
        for metadata in metadatas:
            if not metadata:
                continue
            source_id = str(metadata.get("source_id", ""))
            if source_id:
                grouped[source_id].append(dict(metadata))

        sources: list[SourceInfo] = []
        for source_id, rows in grouped.items():
            first = rows[0]
            sources.append(
                SourceInfo(
                    source_id=source_id,
                    source_type=str(first.get("source_type", "unknown")),
                    title=str(first.get("title", source_id)),
                    locator=str(first.get("locator", "")),
                    chunk_count=len(rows),
                    ingested_at=str(first.get("ingested_at", "")),
                )
            )
        sources.sort(key=lambda item: item.ingested_at, reverse=True)
        return sources
