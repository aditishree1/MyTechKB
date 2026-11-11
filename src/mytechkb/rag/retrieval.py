"""Advanced Information Retrieval algorithms for SDE 2/3 grade RAG:
* In-memory BM25 Okapi lexical search
* Multi-Query Expansion via LLM rewriting
* Reciprocal Rank Fusion (RRF) combining dense vector and sparse lexical results
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from mytechkb.models import Chunk


# ---------------------------------------------------------------------------
# BM25 Okapi Implementation (Zero-dependency, sub-millisecond in-memory index)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_\-\.\:\/]+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms and identifiers."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class BM25Index:
    """Okapi BM25 index over Chunk documents for exact keyword and identifier matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lens: list[int] = []
        self.chunks: list[Chunk] = []
        self.doc_freqs: dict[str, int] = Counter()
        self.term_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}

    def fit(self, chunks: list[Chunk]) -> BM25Index:
        """Build the inverted index and compute IDF statistics for all chunks."""
        self.chunks = list(chunks)
        self.doc_count = len(chunks)
        if self.doc_count == 0:
            self.avg_dl = 0.0
            return self

        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            self.doc_lens.append(len(tokens))
            freqs = Counter(tokens)
            self.term_freqs.append(freqs)
            for token in freqs.keys():
                self.doc_freqs[token] += 1

        self.avg_dl = sum(self.doc_lens) / self.doc_count if self.doc_count else 0.0

        # Precompute Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Score all documents against query tokens using the BM25 formula."""
        if not self.chunks or self.doc_count == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: list[float] = [0.0] * self.doc_count

        for token in q_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i, freqs in enumerate(self.term_freqs):
                if token not in freqs:
                    continue
                tf = freqs[token]
                doc_len = self.doc_lens[i]
                norm = 1.0 - self.b + self.b * (doc_len / (self.avg_dl or 1.0))
                score = idf_val * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                scores[i] += score

        ranked_indices = sorted(
            range(self.doc_count), key=lambda idx: scores[idx], reverse=True
        )

        results: list[tuple[Chunk, float]] = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], scores[idx]))
        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = 60,
    top_n: int = 6,
) -> list[Chunk]:
    """Fuse multiple ranked candidate lists into a single consensus ranking.

    RRF avoids score calibration issues between different retrieval metrics
    (e.g. unbounded BM25 scores vs. cosine distances in [0, 2]):
        RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk.id
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_chunk_ids[:top_n]]


# ---------------------------------------------------------------------------
# Multi-Query Expansion Helper
# ---------------------------------------------------------------------------

async def generate_query_perspectives(query: str, llm: Any) -> list[str]:
    """Generate 2-3 query variations to address vocabulary and syntax mismatch."""
    clean_q = query.strip()
    if not clean_q:
        return []

    # Prompt designed to extract diverse lexical and conceptual variants
    prompt = (
        f"You are an AI search query optimizer. Given the following user question, "
        f"generate 2 alternative search queries that use different technical keywords, "
        f"synonyms, and perspectives to find relevant technical documentation.\n\n"
        f"Original question: {clean_q}\n\n"
        f"Output ONLY 2 numbered queries (one per line). Do not include any introduction or commentary."
    )

    queries = [clean_q]
    try:
        from langchain_core.messages import HumanMessage
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        content = getattr(resp, "content", "") or str(resp)
        for line in content.split("\n"):
            cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            if cleaned and cleaned.lower() != clean_q.lower() and len(cleaned) > 5:
                queries.append(cleaned)
    except Exception:
        # Graceful fallback: return original query if expansion fails or times out
        pass

    return queries[:3]
