"""FastAPI application — routes for ingestion, chat, and source management."""

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from mytechkb.api.health import check_health
from mytechkb.auth import (
    SESSION_COOKIE_NAME,
    TenantContext,
    clear_session_cookie,
    create_guest_token,
    create_user_token,
    get_auth_db,
    set_session_cookie,
    verify_token,
)
from mytechkb.config import Settings, get_settings
from mytechkb.ingest.ids import is_youtube_playlist_url
from mytechkb.ingest.router import ingest_file, ingest_playlist, ingest_url
from mytechkb.rag.chain import stream_answer
from mytechkb.store.factory import get_tenant_store

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ChatBody(BaseModel):
    question: str
    source_ids: list[str] | None = None
    is_notes: bool = False


class AuthBody(BaseModel):
    username: str
    password: str


def get_current_tenant(request: Request) -> TenantContext:
    tenant = getattr(request.state, "tenant", None)
    if tenant is not None:
        return tenant
    # Fallback if accessed outside middleware
    return TenantContext(
        tenant_type="guest",
        tenant_id="default",
        username=None,
        collection_name="kb_guest_default",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(title="MyTechKB", version="0.3.0")

    # ── Tenant Context Middleware ────────────────────────────

    @app.middleware("http")
    async def tenant_session_middleware(request: Request, call_next):
        s = app_settings
        token = request.cookies.get(SESSION_COOKIE_NAME)
        payload = verify_token(token, s.session_secret) if token else None

        cookie_to_set: str | None = None

        if payload and payload.get("type") == "user":
            user_id = str(payload.get("user_id", ""))
            username = str(payload.get("username", ""))
            tenant = TenantContext(
                tenant_type="user",
                tenant_id=user_id,
                username=username,
                collection_name=f"kb_user_{user_id}",
            )
        elif payload and payload.get("type") == "guest":
            guest_id = str(payload.get("guest_id", ""))
            tenant = TenantContext(
                tenant_type="guest",
                tenant_id=guest_id,
                username=None,
                collection_name=f"kb_guest_{guest_id}",
            )
        else:
            # First visit or invalid session: provision fresh isolated guest sandbox
            guest_id = uuid.uuid4().hex[:12]
            cookie_to_set = create_guest_token(s.session_secret, guest_id=guest_id)
            tenant = TenantContext(
                tenant_type="guest",
                tenant_id=guest_id,
                username=None,
                collection_name=f"kb_guest_{guest_id}",
            )

        request.state.tenant = tenant
        response = await call_next(request)

        # Set guest cookie if new and not overridden by route (e.g. login/logout)
        if cookie_to_set and SESSION_COOKIE_NAME not in response.headers.get("set-cookie", ""):
            set_session_cookie(response, cookie_to_set)

        return response

    # ── Pages ────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        s = app_settings
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, s)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "health": check_health(s),
                "sources": store.list_sources(),
                "tenant": tenant,
            },
        )

    # ── Authentication ───────────────────────────────────────

    @app.post("/auth/register")
    def register(payload: AuthBody) -> JSONResponse:
        s = app_settings
        auth_db = get_auth_db(s)
        try:
            user = auth_db.create_user(payload.username, payload.password)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        token = create_user_token(user, s.session_secret)
        response = JSONResponse(
            {"ok": True, "user": {"id": user.id, "username": user.username}}
        )
        set_session_cookie(response, token)
        return response

    @app.post("/auth/login")
    def login(payload: AuthBody) -> JSONResponse:
        s = app_settings
        auth_db = get_auth_db(s)
        user = auth_db.authenticate_user(payload.username, payload.password)
        if not user:
            return JSONResponse(
                {"ok": False, "error": "Invalid username or password"}, status_code=401
            )

        token = create_user_token(user, s.session_secret)
        response = JSONResponse(
            {"ok": True, "user": {"id": user.id, "username": user.username}}
        )
        set_session_cookie(response, token)
        return response

    @app.post("/auth/logout")
    def logout() -> JSONResponse:
        s = app_settings
        # On logout, assign a clean fresh guest sandbox
        new_guest_token = create_guest_token(s.session_secret)
        response = JSONResponse({"ok": True, "message": "Logged out"})
        set_session_cookie(response, new_guest_token)
        return response

    @app.get("/auth/me")
    def me(request: Request) -> dict:
        tenant = get_current_tenant(request)
        return {
            "authenticated": tenant.is_authenticated,
            "username": tenant.username,
            "tenant_type": tenant.tenant_type,
            "tenant_id": tenant.tenant_id,
            "collection": tenant.collection_name,
        }

    # ── Health ───────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict:
        return check_health(app_settings)

    # ── Sources ──────────────────────────────────────────────

    @app.get("/sources")
    def list_sources(request: Request) -> dict:
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, app_settings)
        return {"sources": [_source_dict(s) for s in store.list_sources()]}

    @app.get("/sources/partial", response_class=HTMLResponse)
    def sources_partial(request: Request) -> HTMLResponse:
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, app_settings)
        return templates.TemplateResponse(
            request,
            "sources.html",
            {"sources": store.list_sources()},
        )

    @app.delete("/sources/{source_id}")
    def delete_source(source_id: str, request: Request) -> dict:
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, app_settings)
        deleted = store.delete_source(source_id)
        return {"ok": True, "deleted": deleted}

    # ── Ingestion ────────────────────────────────────────────

    @app.post("/ingest/url")
    def add_url(request: Request, url: str = Form(...)) -> JSONResponse:
        s = app_settings
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, s)
        try:
            source = ingest_url(url, store, s)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "source": _source_dict(source)})

    @app.post("/ingest/file")
    async def add_file(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        s = app_settings
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, s)
        filename = Path(file.filename or "upload").name
        dest = s.uploads_dir / filename
        try:
            with dest.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            source = ingest_file(dest, filename, store, s)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        finally:
            dest.unlink(missing_ok=True)
        return JSONResponse({"ok": True, "source": _source_dict(source)})

    @app.post("/ingest/playlist")
    def add_playlist(request: Request, url: str = Form(...)) -> StreamingResponse:
        """Index every video in a YouTube playlist into current tenant's store."""
        s = app_settings
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, s)

        def events():
            try:
                for event in ingest_playlist(url, store, s):
                    event_name = event.pop("event", "progress")
                    # Serialize SourceInfo objects inside "complete" events
                    if event_name == "complete" and "sources" in event:
                        event["sources"] = [_source_dict(s) for s in event["sources"]]
                    yield f"event: {event_name}\ndata: {json.dumps(event)}\n\n"
            except Exception as exc:
                payload = json.dumps({"message": str(exc)})
                yield f"event: error\ndata: {payload}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    # ── Chat ─────────────────────────────────────────────────

    @app.post("/chat")
    async def chat(request: Request, payload: ChatBody) -> StreamingResponse:
        s = app_settings
        tenant = get_current_tenant(request)
        store = get_tenant_store(tenant.collection_name, s)

        question = payload.question
        source_ids = payload.source_ids
        is_notes = payload.is_notes

        async def events():
            async for item in stream_answer(
                question,
                store,
                s,
                source_ids=source_ids,
                is_notes=is_notes,
            ):
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _source_dict(source) -> dict:
    return {
        "source_id": source.source_id,
        "source_type": source.source_type,
        "title": source.title,
        "locator": source.locator,
        "chunk_count": source.chunk_count,
        "ingested_at": source.ingested_at,
    }


app = create_app()
