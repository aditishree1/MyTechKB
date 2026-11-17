"""Comprehensive tests for multi-tenant account isolation and authentication."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from mytechkb.api.app import create_app
from mytechkb.auth import (
    SESSION_COOKIE_NAME,
    AuthDB,
    clear_session_cookie,
    create_guest_token,
    create_user_token,
    hash_password,
    set_session_cookie,
    sign_token,
    verify_password,
    verify_token,
)
from mytechkb.config import Settings
from mytechkb.models import Chunk
from mytechkb.store.chroma_store import ChromaKnowledgeStore


class FakeEmbeddings:
    """Lightweight deterministic embeddings for testing Chroma collections."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        t = text.lower()
        return [
            float(t.count("tenant") + 1),
            float(t.count("isolation") + 1),
            float(t.count("security") + 1),
            float(len(t) % 10 + 1),
        ]


class TestMultiTenancy(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.db_path = self.dir_path / "auth.db"
        self.chroma_dir = self.dir_path / "chroma"
        self.secret = "test-super-secret-key-12345"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_password_hashing_and_verification(self) -> None:
        pwd = "MySecretPassword123!"
        h1 = hash_password(pwd)
        h2 = hash_password(pwd)

        # Different salts must yield different hashes
        self.assertNotEqual(h1, h2)
        # Both must verify correctly
        self.assertTrue(verify_password(pwd, h1))
        self.assertTrue(verify_password(pwd, h2))
        # Wrong password fails
        self.assertFalse(verify_password("WrongPassword", h1))
        self.assertFalse(verify_password("", h1))

    def test_token_signing_and_verification(self) -> None:
        payload = {"type": "user", "user_id": "usr_abc123", "username": "demo_user"}
        token = sign_token(payload, self.secret)

        decoded = verify_token(token, self.secret)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["user_id"], "usr_abc123")
        self.assertEqual(decoded["username"], "demo_user")

        # Wrong secret fails
        self.assertIsNone(verify_token(token, "wrong-secret"))

        # Tampered payload fails
        tampered = "tampered" + token[8:]
        self.assertIsNone(verify_token(tampered, self.secret))

        # Malformed token fails
        self.assertIsNone(verify_token("invalid_token_no_dot", self.secret))

    def test_auth_db_crud_and_uniqueness(self) -> None:
        auth_db = AuthDB(self.db_path)

        # Register user
        user = auth_db.create_user("DemoUser", "securePass123")
        self.assertEqual(user.username, "demouser")  # normalized to lowercase
        self.assertTrue(user.id)

        # Duplicate username must fail
        with self.assertRaises(ValueError):
            auth_db.create_user("demouser", "anotherPass")

        # Validation rules
        with self.assertRaises(ValueError):
            auth_db.create_user("ab", "validPass123")  # too short username (<3)

        with self.assertRaises(ValueError):
            auth_db.create_user("validuser", "123")  # too short password (<6)

        # Authentication
        auth_success = auth_db.authenticate_user("demouser", "securePass123")
        self.assertIsNotNone(auth_success)
        self.assertEqual(auth_success.id, user.id)

        auth_fail = auth_db.authenticate_user("demouser", "wrongPassword")
        self.assertIsNone(auth_fail)

        auth_missing = auth_db.authenticate_user("nonexistent", "securePass123")
        self.assertIsNone(auth_missing)

    def test_vector_store_hard_tenant_isolation(self) -> None:
        """Verify two users and a guest have zero data cross-leakage."""
        embeddings = FakeEmbeddings()

        store_alice = ChromaKnowledgeStore(
            persist_dir=str(self.chroma_dir),
            collection_name="kb_user_alice123",
            embeddings=embeddings,
        )
        store_bob = ChromaKnowledgeStore(
            persist_dir=str(self.chroma_dir),
            collection_name="kb_user_bob456",
            embeddings=embeddings,
        )
        store_guest = ChromaKnowledgeStore(
            persist_dir=str(self.chroma_dir),
            collection_name="kb_guest_ephemeral789",
            embeddings=embeddings,
        )

        chunk_alice = Chunk(
            id="alice_c1",
            text="Alice's proprietary system design notes on distributed caching.",
            metadata={
                "source_id": "src_alice",
                "source_type": "web",
                "title": "Alice's Private Tech Guide",
                "locator": "https://alice.internal/guide",
                "chunk_index": 0,
                "confidential": "true",
            },
        )
        chunk_bob = Chunk(
            id="bob_c1",
            text="Bob's summary of binary search trees.",
            metadata={
                "source_id": "src_bob",
                "source_type": "web",
                "title": "Bob's Public Notes",
                "locator": "https://bob.internal/notes",
                "chunk_index": 0,
            },
        )

        store_alice.upsert_chunks([chunk_alice])
        store_bob.upsert_chunks([chunk_bob])

        # Alice's store lists only Alice's source
        alice_sources = store_alice.list_sources()
        self.assertEqual(len(alice_sources), 1)
        self.assertEqual(alice_sources[0].source_id, "src_alice")

        # Bob's store lists only Bob's source
        bob_sources = store_bob.list_sources()
        self.assertEqual(len(bob_sources), 1)
        self.assertEqual(bob_sources[0].source_id, "src_bob")

        # Guest store has 0 sources
        guest_sources = store_guest.list_sources()
        self.assertEqual(len(guest_sources), 0)

        # Searching Alice's store returns Alice's chunk, not Bob's
        alice_hits = store_alice.similarity_search("distributed caching", k=5)
        self.assertEqual(len(alice_hits), 1)
        self.assertEqual(alice_hits[0].metadata["source_id"], "src_alice")

        # Searching Bob's store for Alice's content returns Bob's (or empty), never Alice's
        bob_hits = store_bob.similarity_search("distributed caching", k=5)
        for hit in bob_hits:
            self.assertNotEqual(hit.metadata["source_id"], "src_alice")

    def test_fastapi_multi_tenancy_endpoints_and_session_isolation(self) -> None:
        """Full end-to-end test of registration, login, logout, and guest fallback."""
        test_settings = Settings(
            data_dir=self.dir_path,
            session_secret=self.secret,
            llm_provider="gemini",
            gemini_api_key="test_dummy_key",
        )

        app = create_app(settings=test_settings)
        client = TestClient(app)

        # 1. First visit as anonymous visitor
        res_home = client.get("/")
        self.assertEqual(res_home.status_code, 200)
        # Should have set a guest session cookie
        self.assertIn(SESSION_COOKIE_NAME, res_home.cookies)

        # Check /auth/me for this guest
        res_me = client.get("/auth/me")
        self.assertEqual(res_me.status_code, 200)
        guest_me = res_me.json()
        self.assertFalse(guest_me["authenticated"])
        self.assertEqual(guest_me["tenant_type"], "guest")
        self.assertTrue(guest_me["collection"].startswith("kb_guest_"))

        # 2. Register user
        res_reg = client.post(
            "/auth/register",
            json={"username": "test_user_unique", "password": "password123"},
        )
        self.assertEqual(res_reg.status_code, 200, res_reg.json() if res_reg.status_code != 200 else None)
        self.assertTrue(res_reg.json()["ok"])
        self.assertEqual(res_reg.json()["user"]["username"], "test_user_unique")

        # Check /auth/me after registration
        res_me_user = client.get("/auth/me")
        user_me = res_me_user.json()
        self.assertTrue(user_me["authenticated"])
        self.assertEqual(user_me["username"], "test_user_unique")
        self.assertTrue(user_me["collection"].startswith("kb_user_"))

        # 3. Log out -> back to fresh guest sandbox
        res_logout = client.post("/auth/logout")
        self.assertEqual(res_logout.status_code, 200)
        self.assertTrue(res_logout.json()["ok"])

        res_me_guest2 = client.get("/auth/me")
        guest2_me = res_me_guest2.json()
        self.assertFalse(guest2_me["authenticated"])
        self.assertEqual(guest2_me["tenant_type"], "guest")

        # 4. Log in again
        res_login = client.post(
            "/auth/login",
            json={"username": "test_user_unique", "password": "password123"},
        )
        self.assertEqual(res_login.status_code, 200)
        self.assertTrue(res_login.json()["ok"])

        res_me_user_again = client.get("/auth/me")
        self.assertTrue(res_me_user_again.json()["authenticated"])
        self.assertEqual(res_me_user_again.json()["collection"], user_me["collection"])

        # 5. Ingest mock article into user's workspace
        with patch("mytechkb.api.app.ingest_url") as mock_ingest:
            from mytechkb.models import SourceInfo
            mock_ingest.return_value = SourceInfo(
                source_id="src_user_rag",
                source_type="web",
                title="Advanced RAG Architecture Guide",
                locator="https://example.internal/rag",
                chunk_count=12,
                ingested_at="2026-11-06T12:00:00Z",
            )
            res_ingest = client.post("/ingest/url", data={"url": "https://example.internal/rag"})
            self.assertEqual(res_ingest.status_code, 200)
            self.assertTrue(res_ingest.json()["ok"])

        # 6. User sees source in /sources
        with patch("mytechkb.store.chroma_store.ChromaKnowledgeStore.list_sources") as mock_list:
            mock_list.return_value = [mock_ingest.return_value]
            res_sources = client.get("/sources")
            self.assertEqual(res_sources.status_code, 200)
            self.assertEqual(len(res_sources.json()["sources"]), 1)
            self.assertEqual(res_sources.json()["sources"][0]["source_id"], "src_user_rag")

        # 7. Logout to guest and verify isolated guest collection is queried
        res_logout2 = client.post("/auth/logout")
        self.assertEqual(res_logout2.status_code, 200)

        # In guest session, list_sources on empty guest collection returns 0
        res_guest_sources = client.get("/sources")
        self.assertEqual(res_guest_sources.status_code, 200)
        self.assertEqual(len(res_guest_sources.json()["sources"]), 0)


if __name__ == "__main__":
    unittest.main()
