"""Automated test suite for MyTechKB ingestion, chunking, vector store, and API."""

from pathlib import Path
import tempfile
import unittest

from mytechkb.config import Settings
from mytechkb.ingest.chunking import chunk_documents
from mytechkb.ingest.ids import (
    is_youtube_playlist_url,
    is_youtube_url,
    stable_id,
    youtube_playlist_id,
    youtube_video_id,
)
from mytechkb.ingest.youtube import _walk_playlist_renderers, format_timestamp
from mytechkb.models import Chunk, ExtractedDocument
from mytechkb.rag.prompts import (
    STUDY_NOTES_PROMPT,
    SYSTEM_PROMPT,
    citation_payload,
    format_context,
)
from mytechkb.rag.retrieval import BM25Index, reciprocal_rank_fusion
from mytechkb.store.chroma_store import ChromaKnowledgeStore


class FakeEmbeddings:
    """Deterministic lightweight embeddings for testing ChromaKnowledgeStore."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        t = text.lower()
        return [
            float(t.count("rag") + 1),
            float(t.count("chunk") + 1),
            float(t.count("vector") + 1),
            float(len(t) % 10 + 1),
        ]


class TestMyTechKB(unittest.TestCase):
    def test_youtube_url_and_playlist_parsing(self) -> None:
        # Single video URLs
        self.assertEqual(
            youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            youtube_video_id("https://youtu.be/dQw4w9WgXcQ?t=42"),
            "dQw4w9WgXcQ",
        )
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertFalse(is_youtube_url("https://example.com/article"))

        # Playlist URLs
        self.assertEqual(
            youtube_playlist_id("https://www.youtube.com/playlist?list=PL12345_abc-DEF"),
            "PL12345_abc-DEF",
        )
        self.assertEqual(
            youtube_playlist_id(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL12345_abc-DEF&index=2"
            ),
            "PL12345_abc-DEF",
        )
        self.assertTrue(
            is_youtube_playlist_url("https://www.youtube.com/playlist?list=PL12345_abc-DEF")
        )
        self.assertFalse(
            is_youtube_playlist_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        )

    def test_playlist_renderer_extraction(self) -> None:
        sample_yt_data = {
            "contents": {
                "twoColumnBrowseResultsRenderer": {
                    "tabs": [
                        {
                            "tabRenderer": {
                                "content": {
                                    "sectionListRenderer": {
                                        "contents": [
                                            {
                                                "itemSectionRenderer": {
                                                    "contents": [
                                                        {
                                                            "playlistVideoListRenderer": {
                                                                "contents": [
                                                                    {
                                                                        "playlistVideoRenderer": {
                                                                            "videoId": "vid00000001",
                                                                            "title": {
                                                                                "runs": [
                                                                                    {"text": "Part 1: Intro to RAG"}
                                                                                ]
                                                                            },
                                                                        }
                                                                    },
                                                                    {
                                                                        "playlistVideoRenderer": {
                                                                            "videoId": "vid00000002",
                                                                            "title": {
                                                                                "simpleText": "Part 2: Chunking Deep Dive"
                                                                            },
                                                                        }
                                                                    },
                                                                ]
                                                            }
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
        videos = _walk_playlist_renderers(sample_yt_data)
        self.assertEqual(len(videos), 2)
        self.assertEqual(videos[0], {"id": "vid00000001", "title": "Part 1: Intro to RAG"})
        self.assertEqual(
            videos[1], {"id": "vid00000002", "title": "Part 2: Chunking Deep Dive"}
        )

    def test_youtube_chunking_preserves_timestamps_and_deep_links(self) -> None:
        settings = Settings(youtube_window_seconds=60.0, chunk_size=300, chunk_overlap=50)
        source_id = stable_id("youtube", "dQw4w9WgXcQ")
        locator = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        docs = [
            ExtractedDocument(
                source_id=source_id,
                source_type="youtube",
                title="RAG Masterclass Ep 1",
                text="Welcome to the RAG architecture course where we build vector search.",
                locator=locator,
                metadata={
                    "source_id": source_id,
                    "source_type": "youtube",
                    "title": "RAG Masterclass Ep 1",
                    "locator": locator,
                    "start_seconds": 15.0,
                    "duration": 10.0,
                    "timestamp": format_timestamp(15.0),
                },
            ),
            ExtractedDocument(
                source_id=source_id,
                source_type="youtube",
                title="RAG Masterclass Ep 1",
                text="Next we discuss document chunking and overlap strategies in LangChain.",
                locator=locator,
                metadata={
                    "source_id": source_id,
                    "source_type": "youtube",
                    "title": "RAG Masterclass Ep 1",
                    "locator": locator,
                    "start_seconds": 25.0,
                    "duration": 12.0,
                    "timestamp": format_timestamp(25.0),
                },
            ),
            ExtractedDocument(
                source_id=source_id,
                source_type="youtube",
                title="RAG Masterclass Ep 1",
                text="Finally we evaluate strict grounding so the model never hallucinates.",
                locator=locator,
                metadata={
                    "source_id": source_id,
                    "source_type": "youtube",
                    "title": "RAG Masterclass Ep 1",
                    "locator": locator,
                    "start_seconds": 95.0,
                    "duration": 15.0,
                    "timestamp": format_timestamp(95.0),
                },
            ),
        ]

        chunks = chunk_documents(docs, settings)
        self.assertGreaterEqual(len(chunks), 2)
        first_meta = chunks[0].metadata
        self.assertEqual(first_meta["start_seconds"], 15.0)
        self.assertEqual(first_meta["video_deep_link"], "https://youtu.be/dQw4w9WgXcQ?t=15")
        self.assertIn("00:15", str(first_meta["timestamp"]))

        cites = citation_payload(chunks)
        self.assertEqual(cites[0]["video_deep_link"], "https://youtu.be/dQw4w9WgXcQ?t=15")
        formatted = format_context(chunks)
        self.assertIn("[1] RAG Masterclass Ep 1", formatted)
        self.assertIn("ONLY", SYSTEM_PROMPT)
        self.assertIn("Mermaid.js", STUDY_NOTES_PROMPT)

    def test_chroma_store_scoped_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaKnowledgeStore(
                persist_dir=tmpdir,
                collection_name="test_kb",
                embeddings=FakeEmbeddings(),
            )
            chunks = [
                Chunk(
                    id="c1",
                    text="RAG chunking strategies in video 1",
                    metadata={
                        "source_id": "src_1",
                        "source_type": "youtube",
                        "title": "Video 1",
                        "locator": "https://youtu.be/11111111111",
                        "ingested_at": "2026-11-03T10:00:00Z",
                    },
                ),
                Chunk(
                    id="c2",
                    text="Vector database indexing in PDF 2",
                    metadata={
                        "source_id": "src_2",
                        "source_type": "pdf",
                        "title": "Paper 2",
                        "locator": "paper2.pdf",
                        "ingested_at": "2026-11-03T11:00:00Z",
                    },
                ),
            ]
            store.upsert_chunks(chunks)

            sources = store.list_sources()
            self.assertEqual(len(sources), 2)

            # Unscoped search returns from both sources
            all_results = store.similarity_search("RAG chunking vector", k=4)
            self.assertEqual(len(all_results), 2)

            # Scoped search to src_1 returns only src_1
            scoped_1 = store.similarity_search(
                "RAG chunking vector", k=4, source_ids=["src_1"]
            )
            self.assertEqual(len(scoped_1), 1)
            self.assertEqual(scoped_1[0].metadata["source_id"], "src_1")

            # Scoped search to src_2 returns only src_2
            scoped_2 = store.similarity_search(
                "RAG chunking vector", k=4, source_ids=["src_2"]
            )
            self.assertEqual(len(scoped_2), 1)
            self.assertEqual(scoped_2[0].metadata["source_id"], "src_2")

            # Test get_all_chunks
            all_chunks = store.get_all_chunks()
            self.assertEqual(len(all_chunks), 2)
            scoped_chunks = store.get_all_chunks(source_ids=["src_1"])
            self.assertEqual(len(scoped_chunks), 1)
            self.assertEqual(scoped_chunks[0].id, "c1")

    def test_bm25_lexical_search_and_rrf(self) -> None:
        chunks = [
            Chunk(id="c1", text="Python LangChain RecursiveCharacterTextSplitter tutorial", metadata={"title": "Doc 1"}),
            Chunk(id="c2", text="FastAPI streaming SSE with server sent events", metadata={"title": "Doc 2"}),
            Chunk(id="c3", text="ChromaDB cosine similarity vector distance metric", metadata={"title": "Doc 3"}),
        ]
        bm25 = BM25Index().fit(chunks)
        results = bm25.search("FastAPI SSE", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.id, "c2")
        self.assertGreater(score, 0.0)

        # Test Reciprocal Rank Fusion
        dense_ranking = [chunks[0], chunks[1], chunks[2]]  # c1 is 1st, c2 is 2nd, c3 is 3rd
        bm25_ranking = [chunks[1], chunks[2], chunks[0]]   # c2 is 1st, c3 is 2nd, c1 is 3rd

        # c2 has rank 2 in dense and rank 1 in bm25: score = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0163 = 0.0324
        # c1 has rank 1 in dense and rank 3 in bm25: score = 1/(60+1) + 1/(60+3) = 0.0163 + 0.0158 = 0.0321
        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60, top_n=3)
        self.assertEqual(fused[0].id, "c2")
        self.assertEqual(len(fused), 3)


if __name__ == "__main__":
    unittest.main()
