"""Multi-tenant authentication and session management.

Provides:
* PBKDF2-HMAC-SHA256 password hashing (zero external dependencies)
* Cryptographically signed session tokens (HMAC-SHA256)
* SQLite user registry with transactional schema initialization
* Dual-tier TenantContext: authenticated User vs. isolated Guest Sandbox
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request, Response

from mytechkb.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Password Hashing & Verification
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a 16-byte cryptographically secure salt."""
    salt = secrets.token_bytes(16)
    kdf = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${kdf.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hash in constant time."""
    try:
        salt_hex, key_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(derived, expected_key)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Signed Session Tokens
# ---------------------------------------------------------------------------

def sign_token(payload: dict[str, Any], secret: str) -> str:
    """Sign JSON payload with HMAC-SHA256: returns base64_payload.base64_sig."""
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    sig = hmac.new(secret.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).digest()
    b64_sig = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"{b64_payload}.{b64_sig}"


def verify_token(token: str, secret: str) -> dict[str, Any] | None:
    """Verify HMAC-SHA256 signed token and return the decoded payload."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        b64_payload, b64_sig = parts

        # Verify signature
        expected_sig = hmac.new(
            secret.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256
        ).digest()
        sig_padding = "=" * (-len(b64_sig) % 4)
        actual_sig = base64.urlsafe_b64decode(b64_sig + sig_padding)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Decode payload
        payload_padding = "=" * (-len(b64_payload) % 4)
        raw_payload = base64.urlsafe_b64decode(b64_payload + payload_padding)
        return json.loads(raw_payload.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SQLite User Database
# ---------------------------------------------------------------------------

@dataclass
class User:
    id: str
    username: str
    created_at: str


class AuthDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_user(self, username: str, password: str) -> User:
        clean_user = username.strip().lower()
        if len(clean_user) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")

        user_id = uuid.uuid4().hex[:12]
        pwd_hash = hash_password(password)
        created_at = uuid.uuid1().hex  # stable chronological marker

        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (user_id, clean_user, pwd_hash),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"Username '{clean_user}' is already taken")

        return User(id=user_id, username=clean_user, created_at=created_at)

    def authenticate_user(self, username: str, password: str) -> User | None:
        clean_user = username.strip().lower()
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
                (clean_user,),
            ).fetchone()
            if not row:
                return None
            if not verify_password(password, row["password_hash"]):
                return None
            return User(id=row["id"], username=row["username"], created_at=row["created_at"])

    def get_user(self, user_id: str) -> User | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            return User(id=row["id"], username=row["username"], created_at=row["created_at"])


# ---------------------------------------------------------------------------
# Tenant Context
# ---------------------------------------------------------------------------

@dataclass
class TenantContext:
    tenant_type: str  # "user" or "guest"
    tenant_id: str
    username: str | None
    collection_name: str

    @property
    def is_authenticated(self) -> bool:
        return self.tenant_type == "user"


SESSION_COOKIE_NAME = "mytechkb_session"


def create_user_token(user: User, secret: str) -> str:
    """Create a signed session token for an authenticated user."""
    return sign_token(
        {"type": "user", "user_id": user.id, "username": user.username}, secret
    )


def create_guest_token(secret: str, guest_id: str | None = None) -> str:
    """Create a signed session token for an ephemeral guest sandbox."""
    return sign_token(
        {"type": "guest", "guest_id": guest_id or uuid.uuid4().hex[:12]}, secret
    )


def set_session_cookie(response: Response, token: str) -> None:
    """Set standard HTTP-only session cookie across the application."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,  # 30 days
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Clear session cookie."""
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def get_auth_db(settings: Settings | None = None) -> AuthDB:
    settings = settings or get_settings()
    return AuthDB(settings.auth_db_path)


def resolve_tenant(
    request: Request, response: Response | None = None, settings: Settings | None = None
) -> TenantContext:
    """Resolve current tenant from HTTP-only cookie, or provision an isolated guest session."""
    settings = settings or get_settings()
    token = request.cookies.get(SESSION_COOKIE_NAME)
    payload = verify_token(token, settings.session_secret) if token else None

    if payload and payload.get("type") == "user":
        user_id = str(payload.get("user_id", ""))
        username = str(payload.get("username", ""))
        if user_id:
            return TenantContext(
                tenant_type="user",
                tenant_id=user_id,
                username=username,
                collection_name=f"kb_user_{user_id}",
            )

    if payload and payload.get("type") == "guest":
        guest_id = str(payload.get("guest_id", ""))
        if guest_id:
            return TenantContext(
                tenant_type="guest",
                tenant_id=guest_id,
                username=None,
                collection_name=f"kb_guest_{guest_id}",
            )

    # Provision a new isolated guest session ID
    guest_id = uuid.uuid4().hex[:12]
    new_payload = {"type": "guest", "guest_id": guest_id}
    new_token = sign_token(new_payload, settings.session_secret)

    if response:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=new_token,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 30,  # 30 days
        )

    return TenantContext(
        tenant_type="guest",
        tenant_id=guest_id,
        username=None,
        collection_name=f"kb_guest_{guest_id}",
    )
