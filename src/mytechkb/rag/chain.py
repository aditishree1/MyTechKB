from collections.abc import AsyncIterator
import json

import httpx
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from mytechkb.config import Settings, get_settings
from mytechkb.rag.prompts import (
    STUDY_NOTES_PROMPT,
    SYSTEM_PROMPT,
    citation_payload,
    format_context,
)
from mytechkb.rag.retrieval import (
    BM25Index,
    generate_query_perspectives,
    reciprocal_rank_fusion,
)
from mytechkb.store.base import KnowledgeStore


class GeminiRestChat:
    """Lightweight async streaming client for the Google Gemini REST API."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.1) -> None:
        self._model = model.removeprefix("models/")
        self._api_key = api_key
        self._temperature = temperature

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        system_parts: list[dict] = []
        contents: list[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append({"text": str(msg.content)})
            else:
                contents.append({"role": "user", "parts": [{"text": str(msg.content)}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": self._temperature},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:streamGenerateContent"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                params={"alt": "sse", "key": self._api_key},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    data = json.loads(raw)
                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            text = part.get("text", "")
                            if text:
                                yield AIMessageChunk(content=text)


def build_llm(settings: Settings | None = None):
    """Build LLM based on configured provider."""
    settings = settings or get_settings()
    if settings.llm_provider.strip().lower() == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.gemini_llm_model,
                google_api_key=settings.gemini_api_key,
                temperature=0.1,
                convert_system_message_to_human=True,
            )
        except ImportError:
            return GeminiRestChat(
                model=settings.gemini_llm_model,
                api_key=settings.gemini_api_key,
                temperature=0.1,
            )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )


async def stream_answer(
    question: str,
    store: KnowledgeStore,
    settings: Settings | None = None,
    source_ids: list[str] | None = None,
    is_notes: bool = False,
) -> AsyncIterator[dict]:
    """Stream an answer to the user's question, grounded in the knowledge base."""
    settings = settings or get_settings()
    question = question.strip()
    if not question:
        yield {"event": "error", "data": {"message": "Ask a question first."}}
        return

    # Build LLM client
    llm = build_llm(settings)

    # Detect if user is requesting comprehensive study notes
    notes_triggers = ("/notes", "generate notes", "create notes", "make notes", "study notes", "revision notes")
    wants_notes = is_notes or any(t in question.lower() for t in notes_triggers)

    if wants_notes:
        # Notes Synthesis Mode: Retrieve comprehensive chunks across target source(s)
        try:
            corpus = store.get_all_chunks(source_ids=source_ids)
        except Exception:
            corpus = []

        if not corpus:
            yield {
                "event": "empty",
                "data": {
                    "message": "No materials found in the knowledge base to generate notes from. Add or select a source first."
                },
            }
            return

        # Sample up to 24 chunks uniformly across the corpus to cover full beginning-to-end scope
        max_notes_chunks = 24
        if len(corpus) > max_notes_chunks:
            step = len(corpus) / max_notes_chunks
            chunks = [corpus[int(i * step)] for i in range(max_notes_chunks)]
        else:
            chunks = corpus

        yield {"event": "sources", "data": {"sources": citation_payload(chunks)}}

        context = format_context(chunks)
        notes_prompt = (
            f"User Request: {question}\n\n"
            f"Context Material:\n{context}\n\n"
            "Generate comprehensive, modular revision study notes based STRICTLY on the context above. "
            "Include a Mermaid diagram, core technical principles, a markdown comparison table, "
            "an implementation blueprint, and critical pitfalls."
        )

        async for message in llm.astream(
            [SystemMessage(content=STUDY_NOTES_PROMPT), HumanMessage(content=notes_prompt)]
        ):
            text = getattr(message, "text", None) or message.content
            if isinstance(text, list):
                text = "".join(
                    part if isinstance(part, str) else str(part.get("text", ""))
                    if isinstance(part, dict)
                    else ""
                    for part in text
                )
            if isinstance(text, str) and text:
                yield {"event": "token", "data": {"text": text}}
        yield {"event": "done", "data": {}}
        return

    # Standard Q&A Mode: Multi-Query Expansion + Hybrid BM25 & Dense Search + RRF
    queries = [question]
    if settings.enable_multi_query:
        try:
            perspectives = await generate_query_perspectives(question, llm)
            if perspectives:
                queries = perspectives
        except Exception:
            queries = [question]

    ranked_lists: list[list[Chunk]] = []

    # 1. Dense Semantic Vector Search across all expanded queries
    for q in queries:
        try:
            dense_results = store.similarity_search(
                q, k=settings.retrieve_k, source_ids=source_ids
            )
            if dense_results:
                ranked_lists.append(dense_results)
        except Exception:
            pass

    # 2. Sparse Lexical Search (BM25 Okapi) for exact keyword/token matching
    if settings.enable_hybrid_search:
        try:
            corpus = store.get_all_chunks(source_ids=source_ids)
            if corpus:
                bm25 = BM25Index().fit(corpus)
                bm25_matches = [
                    chunk for chunk, _ in bm25.search(question, top_k=settings.retrieve_k)
                ]
                if bm25_matches:
                    ranked_lists.append(bm25_matches)
        except Exception:
            pass

    # 3. Reciprocal Rank Fusion (RRF) to merge candidate lists into consensus top-k
    if ranked_lists:
        chunks = reciprocal_rank_fusion(
            ranked_lists, k=settings.rrf_k, top_n=settings.retrieve_k
        )
    else:
        chunks = []

    if not chunks:
        yield {
            "event": "empty",
            "data": {
                "message": "The knowledge base is empty or nothing matched. Add a source first."
            },
        }
        return

    # Emit source citations to the frontend
    yield {"event": "sources", "data": {"sources": citation_payload(chunks)}}

    # Build the grounded prompt
    context = format_context(chunks)
    prompt = (
        f"Question: {question}\n\n"
        f"Context from the knowledge base:\n{context}\n\n"
        "Answer the question using ONLY the context above. "
        "Cite sources using [n] notation. If the context doesn't cover the question, say so."
    )

    async for message in llm.astream(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    ):
        text = getattr(message, "text", None) or message.content
        if isinstance(text, list):
            text = "".join(
                part if isinstance(part, str) else str(part.get("text", ""))
                if isinstance(part, dict)
                else ""
                for part in text
            )
        if isinstance(text, str) and text:
            yield {"event": "token", "data": {"text": text}}
    yield {"event": "done", "data": {}}
