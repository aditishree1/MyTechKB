# MyTechKB

**Personal AI Knowledge Base for Technical Revision** — Add YouTube playlists, YouTube videos, web articles, PDFs, and images; ask questions and revise with **strict grounding** and **clickable timestamp/page citations**.

Built with **LangChain**, **ChromaDB** ($0 persistent vector database), and **FastAPI**.

---

## Key Features

- **🎬 YouTube Playlist & Video Ingestion**: Paste a single video URL or an entire course playlist (`youtube.com/playlist?list=...`). Automatically fetches real video titles (via YouTube oEmbed / playlist metadata) and transcripts with real-time per-video progress.
- **⏱️ Segment-Aligned Chunking & Clickable Deep Links**: Transcripts are chunked on exact segment boundaries with rolling overlap, preserving `MM:SS–MM:SS` time ranges and generating clickable `https://youtu.be/<id>?t=<seconds>` links.
- **📄 Web Articles, PDFs & Images**: Markdown-aware splitting for technical articles (`trafilatura` + `readability` fallback), page-tracked PDF parsing (`PyMuPDF`), and image OCR (`Tesseract`).
- **🎯 Strict Grounding & Source Scoping**: Answers are strictly constrained to your uploaded material (no outside hallucinations). Filter your chat to **All Sources** or select specific videos/documents when revising a course.
- **💸 100% Free Vector Store & Dual LLM Support**:
  - **Local ($0, 100% offline)**: Uses **Ollama** (`llama3.2` + `nomic-embed-text`) and local **ChromaDB**.
  - **Cloud Deployment ($0 free tier)**: Switch `LLM_PROVIDER=gemini` with a free Google AI Studio API key (`gemini-2.0-flash` + `models/text-embedding-004`) to deploy on Render, Railway, or Hugging Face Spaces without needing an 8 GB RAM/GPU server.

---

## Quick Start (Local with Ollama)

1. Pull the local models in [Ollama](https://ollama.com):
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. Configure & run with [uv](https://docs.astral.sh/uv/):
   ```bash
   cp .env.example .env
   uv sync
   uv run mytechkb
   ```
3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Cloud / Zero-GPU Mode (Google Gemini Free Tier)

To run or deploy without local Ollama:

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Set in `.env`:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_key_here
   ```
3. Start the server:
   ```bash
   uv run mytechkb
   ```

---

## Docker Compose

```bash
docker compose up --build
```

Starts Ollama, pulls `llama3.2` and `nomic-embed-text`, and serves MyTechKB on port `8000` with persistent volumes (`chroma_data`, `ollama_data`).

---

## Running Tests

```bash
uv run python -m unittest discover -s tests -v
```
