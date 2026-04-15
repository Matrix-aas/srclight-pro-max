"""Tests for embedding providers, vector search, and hybrid RRF."""

from unittest.mock import patch

import pytest

from srclight.embeddings import (
    DEFAULT_OLLAMA_EMBED_MODEL,
    CohereProvider,
    EmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
    VoyageProvider,
    _embed_request_timeout,
    _index_embed_request_timeout,
    bytes_to_vector,
    cosine_similarity,
    embed_symbols,
    get_provider,
    prepare_embedding_text,
    rrf_merge,
    vector_to_bytes,
    vectors_to_bytes,
)

# --- Embed request timeout (Cursor/IDE tool timeout) ---


def test_embed_request_timeout_default():
    import os
    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("SRCLIGHT_EMBED_REQUEST_TIMEOUT", None)
        assert _embed_request_timeout() == 20


def test_embed_request_timeout_from_env():
    with patch.dict("os.environ", {"SRCLIGHT_EMBED_REQUEST_TIMEOUT": "45"}, clear=False):
        assert _embed_request_timeout() == 45


def test_index_embed_request_timeout_default():
    import os

    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("SRCLIGHT_INDEX_EMBED_REQUEST_TIMEOUT", None)
        assert _index_embed_request_timeout() == 120


def test_index_embed_request_timeout_from_env():
    with patch.dict("os.environ", {"SRCLIGHT_INDEX_EMBED_REQUEST_TIMEOUT": "240"}, clear=False):
        assert _index_embed_request_timeout() == 240


# --- Fixtures ---


class MockProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dims: int = 4):
        self._dims = dims

    @property
    def name(self) -> str:
        return "mock:test-model"

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings based on text hash."""
        results = []
        for text in texts:
            h = hash(text) & 0xFFFFFFFF
            vec = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self._dims)]
            # Normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


# --- Test prepare_embedding_text ---


def test_prepare_embedding_text_full():
    sym = {
        "qualified_name": "MyClass::my_method",
        "name": "my_method",
        "signature": "void my_method(int x)",
        "doc_comment": "Does something useful.",
        "content": "void my_method(int x) { return x + 1; }",
    }
    text = prepare_embedding_text(sym)
    assert "MyClass::my_method" in text
    assert "void my_method(int x)" in text
    assert "Does something useful." in text
    assert "return x + 1" in text


def test_prepare_embedding_text_minimal():
    sym = {"name": "foo", "content": "int foo() { return 42; }"}
    text = prepare_embedding_text(sym)
    assert "foo" in text
    assert "return 42" in text


def test_prepare_embedding_text_truncation():
    sym = {"name": "big", "content": "x" * 5000}
    text = prepare_embedding_text(sym)
    # Content should be truncated to ~2000 chars
    assert len(text) < 2100


def test_prepare_embedding_text_vue_component_summary():
    sym = {
        "name": "TemplateStyleSignals",
        "qualified_name": "TemplateStyleSignals",
        "signature": (
            "component TemplateStyleSignals | BaseCard, NuxtLink, slot | "
            "v-for, v-if | postcss, module, .card, .hero, --accent-color"
        ),
        "doc_comment": (
            "Vue component TemplateStyleSignals. "
            "Template components: BaseCard (Base Card), NuxtLink (Nuxt Link), slot. "
            "Style vars: --accent-color (accent color). Style mode: postcss, module."
        ),
        "content": "",
    }

    text = prepare_embedding_text(sym)

    assert "BaseCard" in text
    assert "Base Card" in text
    assert "accent color" in text
    assert "postcss" in text


def test_prepare_embedding_text_vue_script_hint_summary():
    sym = {
        "name": "ScriptHints",
        "qualified_name": "ScriptHints",
        "signature": (
            "component ScriptHints | definePageMeta, defineProps | useFetch, useRoute"
        ),
        "doc_comment": (
            "Vue component ScriptHints. "
            "Script macros: definePageMeta (define Page Meta), defineProps (define Props). "
            "Script composables: useFetch (use Fetch), useRoute (use Route). "
            "Fetch paths: /api/items. Page meta values: auth, dashboard."
        ),
        "content": "",
    }

    text = prepare_embedding_text(sym)

    assert "define Page Meta" in text
    assert "use Fetch" in text
    assert "/api/items" in text
    assert "dashboard" in text


def test_prepare_embedding_text_mongoose_schema_metadata():
    sym = {
        "name": "UserSchema",
        "kind": "schema",
        "signature": "mongoose schema | User | users",
        "metadata": {
            "framework": "mongoose",
            "resource": "schema",
            "entity_name": "User",
            "collection_name": "users",
        },
    }

    text = prepare_embedding_text(sym)

    assert "mongoose" in text
    assert "schema" in text
    assert "User" in text
    assert "users" in text


def test_prepare_embedding_text_mikroorm_repository_metadata():
    sym = {
        "name": "UserRepository",
        "kind": "repository",
        "signature": "mikroorm repository | UserRepository",
        "metadata": {
            "framework": "mikroorm",
            "resource": "repository",
            "entity_name": "User",
            "table_name": "users",
            "repository_owner": "User",
        },
    }

    text = prepare_embedding_text(sym)

    assert "mikroorm" in text
    assert "repository" in text
    assert "User" in text
    assert "users" in text


def test_prepare_embedding_text_persistence_aggregate_metadata():
    sym = {
        "name": "accountModels",
        "kind": "database",
        "signature": "mongoose model registry | Account | accounts",
        "metadata": {
            "framework": "mongoose",
            "resource": "database",
            "entity_names": ["Account", "Session"],
            "collection_names": ["accounts", "sessions"],
        },
    }

    text = prepare_embedding_text(sym)

    assert "Account" in text
    assert "Session" in text
    assert "accounts" in text
    assert "sessions" in text


def test_prepare_embedding_text_wave1_backend_symbols():
    route_handler = {
        "name": "getMe",
        "kind": "route_handler",
        "signature": "GET /auth/me",
        "metadata": {
            "framework": "nestjs",
            "resource": "route_handler",
            "http_method": "GET",
            "route_path": "/auth/me",
            "controller_path": "/auth",
        },
    }
    module = {
        "name": "AuthModule",
        "kind": "module",
        "signature": "Nest module AuthModule",
        "metadata": {
            "framework": "nestjs",
            "resource": "module",
            "imports": ["AuthModule", "ConfigModule", "TypeOrmModule"],
            "controllers": ["AuthController"],
            "providers": ["AuthService", "JwtAuthGuard"],
            "exports": ["AuthService"],
        },
    }
    schema = {
        "name": "UserSchema",
        "kind": "schema",
        "signature": "mongoose schema | User | users",
        "metadata": {
            "framework": "mongoose",
            "resource": "schema",
            "entity_name": "User",
            "collection_name": "users",
        },
    }
    entity = {
        "name": "User",
        "kind": "entity",
        "signature": "mikroorm entity | User | users",
        "metadata": {
            "framework": "mikroorm",
            "resource": "entity",
            "entity_name": "User",
            "table_name": "users",
            "fields": [
                {"name": "id", "field_name": "id", "kind": "primary_key"},
                {"name": "email", "field_name": "email_address", "kind": "property"},
            ],
        },
    }

    prepared = "\n".join(
        prepare_embedding_text(symbol)
        for symbol in (route_handler, module, schema, entity)
    )

    assert "GET /auth/me" in prepared
    assert "module AuthModule" in prepared
    assert "schema users" in prepared
    assert "mikroorm entity User" in prepared


def test_prepare_embedding_text_wave2_messagepattern_bullmq_redis_rabbitmq_metadata():
    microservice_handler = {
        "name": "handlePrediction",
        "kind": "microservice_handler",
        "signature": "MessagePattern prediction.run",
        "metadata": {
            "framework": "nestjs",
            "resource": "microservice_handler",
            "message_pattern": "prediction.run",
            "pattern": "prediction.run",
            "transport": "rmq",
            "role": "consumer",
        },
    }
    queue_processor = {
        "name": "notificationWorker",
        "kind": "queue_processor",
        "signature": "BullMQ worker notifications",
        "metadata": {
            "framework": "bullmq",
            "resource": "queue_processor",
            "queue_name": "notifications",
            "job_name": "deliver-email",
            "role": "consumer",
        },
    }
    redis_transport = {
        "name": "redisClient",
        "kind": "transport",
        "signature": "Redis client",
        "metadata": {
            "framework": "redis",
            "resource": "transport",
            "transport": "redis",
            "connection_url": "process.env.REDIS_URL",
            "role": "client",
        },
    }
    rabbitmq_consumer = {
        "name": "startAchievementConsumer",
        "kind": "microservice_handler",
        "signature": "RabbitMQ consumer achievement-events",
        "metadata": {
            "framework": "amqplib",
            "resource": "microservice_handler",
            "queue_name": "achievement-events",
            "transport": "rmq",
            "role": "consumer",
        },
    }

    prepared = "\n".join(
        prepare_embedding_text(symbol)
        for symbol in (microservice_handler, queue_processor, redis_transport, rabbitmq_consumer)
    )

    assert "prediction.run" in prepared
    assert "rmq" in prepared
    assert "notifications" in prepared
    assert "deliver-email" in prepared
    assert "process.env.REDIS_URL" in prepared
    assert "achievement-events" in prepared


def test_prepare_embedding_text_wrapper_backed_microservice_handler_uses_resolved_metadata():
    wrapped_handler = {
        "name": "handleDiaryPush",
        "kind": "microservice_handler",
        "signature": "RpcRequest diary.note.push",
        "metadata": {
            "framework": "nestjs",
            "resource": "microservice_handler",
            "message_pattern": "diary.note.push",
            "pattern": "diary.note.push",
            "transport": "rmq",
            "role": "consumer",
        },
    }

    prepared = prepare_embedding_text(wrapped_handler)

    assert "diary.note.push" in prepared
    assert "message pattern: diary.note.push" in prepared
    assert "transport: rmq" in prepared
    assert "nestjs microservice_handler" in prepared


def test_prepare_embedding_text_async_symbols_include_transport_role_and_queue_hints():
    publisher = {
        "name": "publishDiaryPush",
        "kind": "microservice_handler",
        "signature": "ClientProxy emit diary.note.push",
        "metadata": {
            "framework": "nestjs",
            "resource": "microservice_handler",
            "event_pattern": "diary.note.push",
            "pattern": "diary.note.push",
            "transport": "rabbitmq",
            "role": "producer",
            "queue_name": "diary-events",
        },
    }
    worker = {
        "name": "notificationWorker",
        "kind": "queue_processor",
        "signature": "BullMQ worker notifications",
        "metadata": {
            "framework": "bullmq",
            "resource": "queue_processor",
            "queue_name": "notifications",
            "job_name": "deliver-email",
            "role": "consumer",
            "transport": "redis",
        },
    }

    prepared = "\n".join(prepare_embedding_text(symbol) for symbol in (publisher, worker))

    assert "event pattern: diary.note.push" in prepared
    assert "transport: rabbitmq" in prepared
    assert "producer" in prepared
    assert "queue: diary-events" in prepared
    assert "transport: redis" in prepared
    assert "consumer" in prepared


@pytest.mark.parametrize(
    "connection_url",
    [
        "redis://user:secret@host:6379/0",
        "'redis://user:secret@host:6379/0'",
        '"redis://user:secret@host:6379/0"',
        "`redis://user:secret@host:6379/0`",
    ],
)
def test_prepare_embedding_text_redacts_transport_connection_url(connection_url):
    sym = {
        "name": "redisClient",
        "kind": "transport",
        "signature": "Redis client",
        "metadata": {
            "framework": "redis",
            "resource": "transport",
            "transport": "redis",
            "connection_url": connection_url,
            "role": "client",
        },
    }

    text = prepare_embedding_text(sym)

    assert "connection: redis://host:6379/0" in text
    assert "secret" not in text
    assert "user@" not in text


def test_indexer_build_embeddings_defers_dimensions_probe_until_after_embed_batch(
    tmp_path, monkeypatch
):
    from srclight.indexer import IndexConfig, Indexer

    class _LazyDimensionsProvider:
        def __init__(self):
            self._ready = False
            self._dimensions = None

        @property
        def name(self) -> str:
            return "mock:lazy-dimensions"

        @property
        def dimensions(self) -> int:
            if not self._ready:
                raise RuntimeError("dimensions unavailable before batch")
            return 4

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self._ready = True
            self._dimensions = 4
            return [[0.0, 1.0, 0.0, 0.0] for _ in texts]

    class _FakeDB:
        def __init__(self):
            self.upserted = []
            self.committed = 0
            self.conn = object()

        def get_symbols_needing_embeddings(self, model):
            return [
                {"id": 1, "body_hash": "h1"},
                {"id": 2, "body_hash": "h2"},
            ]

        def upsert_embedding(self, symbol_id, provider_name, dims, emb_bytes, body_hash):
            self.upserted.append((symbol_id, provider_name, dims, body_hash))

        def commit(self):
            self.committed += 1

    provider = _LazyDimensionsProvider()
    events = []

    monkeypatch.setattr(
        "srclight.embeddings.get_provider",
        lambda model_spec, **kwargs: provider,
    )
    monkeypatch.setattr(
        "srclight.vector_cache.VectorCache",
        type(
            "_NoopVectorCache",
            (),
            {
                "__init__": lambda self, *args, **kwargs: None,
                "build_from_db": lambda self, conn: None,
            },
        ),
    )

    indexer = Indexer(_FakeDB(), IndexConfig(root=tmp_path))
    assert indexer._build_embeddings(DEFAULT_OLLAMA_EMBED_MODEL, on_event=events.append) == 2
    assert events[0]["phase"] == "embeddings"
    assert "mock:lazy-dimensions" in events[0]["detail"]
    assert any("4d" in event.get("detail", "") for event in events)


# --- Test vector math ---


def test_vector_to_bytes_roundtrip():
    vec = [1.0, 2.0, 3.0, 4.0]
    b = vector_to_bytes(vec)
    assert isinstance(b, bytes)
    assert len(b) == 16  # 4 floats * 4 bytes

    recovered = bytes_to_vector(b)
    assert recovered == pytest.approx(vec)


def test_vectors_to_bytes_batch():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    batch = vectors_to_bytes(vecs)
    assert len(batch) == 2
    for b in batch:
        assert len(b) == 8  # 2 floats * 4 bytes


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- Test RRF merge ---


def test_rrf_merge_basic():
    fts = [
        {"symbol_id": 1, "name": "foo", "rank": -20},
        {"symbol_id": 2, "name": "bar", "rank": -10},
    ]
    emb = [
        {"symbol_id": 2, "name": "bar", "similarity": 0.95},
        {"symbol_id": 3, "name": "baz", "similarity": 0.80},
    ]
    merged = rrf_merge(fts, emb)

    # bar (id=2) should be top — it appears in both lists
    assert merged[0]["symbol_id"] == 2
    assert "fts" in merged[0]["sources"]
    assert "embedding" in merged[0]["sources"]
    assert merged[0]["rrf_score"] > 0

    # All 3 symbols should appear
    ids = {r["symbol_id"] for r in merged}
    assert ids == {1, 2, 3}


def test_rrf_merge_weights():
    fts = [{"symbol_id": 1, "name": "a", "rank": -20}]
    emb = [{"symbol_id": 2, "name": "b", "similarity": 0.9}]

    # With heavy embedding weight, embedding result should rank higher
    merged = rrf_merge(fts, emb, embedding_weight=5.0, fts_weight=1.0)
    assert merged[0]["symbol_id"] == 2


def test_rrf_merge_empty():
    assert rrf_merge([], []) == []
    result = rrf_merge([{"symbol_id": 1, "name": "a"}], [])
    assert len(result) == 1


# --- Test provider factory ---


def test_get_provider_ollama():
    provider = get_provider(DEFAULT_OLLAMA_EMBED_MODEL)
    assert isinstance(provider, OllamaProvider)
    assert "ollama" in provider.name


def test_get_provider_ollama_explicit():
    provider = get_provider("ollama:qwen3-embedding:4b")
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama:qwen3-embedding:4b"


def test_get_provider_ollama_tagged_without_prefix():
    provider = get_provider("qwen3-embedding:4b")
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama:qwen3-embedding:4b"


def test_get_provider_ollama_hyphenated_without_prefix():
    provider = get_provider("all-minilm")
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama:all-minilm"


@pytest.mark.parametrize(
    ("model", "expected_name"),
    [
        ("bge-m3", "ollama:bge-m3"),
        ("snowflake-arctic-embed", "ollama:snowflake-arctic-embed"),
        ("nomic-embed-text-v2-moe", "ollama:nomic-embed-text-v2-moe"),
    ],
)
def test_get_provider_supported_embedding_families_without_prefix(model, expected_name):
    provider = get_provider(model)
    assert isinstance(provider, OllamaProvider)
    assert provider.name == expected_name


def test_get_provider_ollama_latest_tag_without_prefix():
    provider = get_provider("nomic-embed-text:latest")
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama:nomic-embed-text:latest"


def test_ollama_is_available_matches_tagged_model(monkeypatch):
    provider = OllamaProvider(model="qwen3-embedding:4b", timeout=19)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"models":[{"name":"qwen3-embedding:4b"}]}'

    def fake_urlopen(req, timeout=None):
        assert timeout == 19
        return _Response()

    monkeypatch.setattr("srclight.embeddings.urllib.request.urlopen", fake_urlopen)
    assert provider.is_available() is True


def test_get_provider_voyage():
    with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
        provider = get_provider("voyage-code-3")
        assert isinstance(provider, VoyageProvider)
        assert "voyage" in provider.name


def test_get_provider_voyage_explicit():
    provider = get_provider("voyage:voyage-code-3", api_key="test-key")
    assert isinstance(provider, VoyageProvider)


def test_get_provider_openai():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider("openai:text-embedding-3-small")
        assert isinstance(provider, OpenAIProvider)
        assert "openai" in provider.name
        assert "text-embedding-3-small" in provider.name


def test_get_provider_openai_inferred():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider("text-embedding-3-small")
        assert isinstance(provider, OpenAIProvider)


def test_get_provider_openai_custom_base_url():
    provider = get_provider(
        "openai:my-model",
        api_key="test-key",
        base_url="https://api.together.xyz",
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider._base_url == "https://api.together.xyz"


def test_get_provider_openai_no_key():
    with patch.dict("os.environ", {}, clear=True):
        # Remove OPENAI_API_KEY if set
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="API key required"):
            get_provider("openai:text-embedding-3-small")


def test_get_provider_cohere():
    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        provider = get_provider("cohere:embed-v4.0")
        assert isinstance(provider, CohereProvider)
        assert "cohere" in provider.name


def test_get_provider_cohere_inferred():
    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        provider = get_provider("embed-v4.0")
        assert isinstance(provider, CohereProvider)


def test_get_provider_cohere_no_key():
    with patch.dict("os.environ", {}, clear=True):
        import os
        os.environ.pop("COHERE_API_KEY", None)
        with pytest.raises(ValueError, match="Cohere API key required"):
            get_provider("cohere:embed-v4.0")


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider("unknown:model")


def test_get_provider_unknown_bare_model():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider("banana")


@pytest.mark.parametrize("model", ["foo-embedding", "myembed", "embedder-123"])
def test_get_provider_unknown_embeddingish_bare_model(model):
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider(model)


@pytest.mark.parametrize("model", ["foo:4b", "banana:latest", "foo-bar:1"])
def test_get_provider_unknown_tagged_model(model):
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider(model)


@pytest.mark.parametrize(
    ("provider", "method_name", "call_args", "expected_timeout"),
    [
        (OllamaProvider(model="qwen3-embedding:4b", timeout=33), "embed_batch", [["hello"]], 33),
        (
            OpenAIProvider(api_key="test-key", model="text-embedding-3-small", timeout=34),
            "embed_batch",
            [["hello"]],
            34,
        ),
        (
            CohereProvider(api_key="test-key", model="embed-v4.0", timeout=35),
            "embed_batch",
            [["hello"]],
            35,
        ),
        (
            VoyageProvider(api_key="test-key", model="voyage-code-3", timeout=36),
            "embed_batch",
            [["hello"]],
            36,
        ),
    ],
)
def test_provider_timeout_used(provider, method_name, call_args, expected_timeout, monkeypatch):
    """Provider constructors should pass their timeout through to HTTP requests."""
    import json

    responses = {
        OllamaProvider: {"embeddings": [[1.0, 2.0, 3.0, 4.0]]},
        OpenAIProvider: {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]},
        CohereProvider: {"embeddings": {"float": [[1.0, 2.0, 3.0, 4.0]]}},
        VoyageProvider: {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]},
    }
    seen_timeouts: list[int | None] = []

    class _Response:
        def __init__(self, payload: dict):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode()

    def fake_urlopen(req, timeout=None):
        seen_timeouts.append(timeout)
        return _Response(responses[type(provider)])

    monkeypatch.setattr("srclight.embeddings.urllib.request.urlopen", fake_urlopen)

    result = getattr(provider, method_name)(*call_args)
    assert result
    assert seen_timeouts == [expected_timeout]


def test_indexer_build_embeddings_uses_index_timeout(tmp_path, monkeypatch):
    from srclight.indexer import IndexConfig, Indexer

    seen: dict[str, object] = {}

    class _FakeProvider:
        name = "ollama:qwen3-embedding:4b"
        dimensions = 4

    class _FakeDB:
        def get_symbols_needing_embeddings(self, model):
            seen["model_name"] = model
            return []

    def fake_get_provider(model_spec, **kwargs):
        seen["model_spec"] = model_spec
        seen["timeout"] = kwargs.get("timeout")
        return _FakeProvider()

    monkeypatch.setattr("srclight.embeddings.get_provider", fake_get_provider)
    monkeypatch.setenv("SRCLIGHT_INDEX_EMBED_REQUEST_TIMEOUT", "240")

    indexer = Indexer(_FakeDB(), IndexConfig(root=tmp_path))
    assert indexer._build_embeddings(DEFAULT_OLLAMA_EMBED_MODEL) == 0
    assert seen == {
        "model_spec": DEFAULT_OLLAMA_EMBED_MODEL,
        "timeout": 240,
        "model_name": "ollama:qwen3-embedding:4b",
    }


def test_indexer_build_embeddings_emits_provider_dims_rates_and_smoothed_eta(tmp_path, monkeypatch):
    from srclight.indexer import IndexConfig, Indexer

    class _FakeProvider:
        name = "ollama:qwen3-embedding:4b"
        dimensions = 2560

    class _FakeDB:
        def __init__(self):
            self.upserted = []
            self.committed = 0
            self.conn = object()

        def get_symbols_needing_embeddings(self, model):
            return [
                {"id": i, "body_hash": f"h{i}"}
                for i in range(1, 66)
            ]

        def upsert_embedding(self, symbol_id, provider_name, dims, emb_bytes, body_hash):
            self.upserted.append((symbol_id, provider_name, dims, body_hash))

        def commit(self):
            self.committed += 1

    timeline = iter([100.0, 101.0, 103.0, 106.0, 106.0])
    events = []

    def fake_get_provider(model_spec, **kwargs):
        return _FakeProvider()

    def fake_embed_symbols(provider, symbols, on_progress=None, batch_size=32):
        assert batch_size == 32
        if on_progress is not None:
            on_progress(1, 3)
            on_progress(2, 3)
            on_progress(3, 3)
        return [(symbol["id"], b"vec") for symbol in symbols]

    monkeypatch.setattr("srclight.embeddings.get_provider", fake_get_provider)
    monkeypatch.setattr("srclight.embeddings.embed_symbols", fake_embed_symbols)
    monkeypatch.setattr("srclight.indexer.time.monotonic", lambda: next(timeline))
    monkeypatch.setattr("srclight.vector_cache.VectorCache.build_from_db", lambda self, conn: None)

    db = _FakeDB()
    indexer = Indexer(db, IndexConfig(root=tmp_path))

    assert indexer._build_embeddings(DEFAULT_OLLAMA_EMBED_MODEL, on_event=events.append) == 65
    assert len(db.upserted) == 65
    assert db.committed >= 1

    assert events[0]["detail"] == "65 symbols | ollama:qwen3-embedding:4b | 2560d"
    assert "32.0 sym/s" in events[1]["detail"]
    assert "1.00 batch/s" in events[1]["detail"]
    assert events[1]["remaining_seconds"] == 0
    assert "21.3 sym/s" in events[2]["detail"]
    assert "0.67 batch/s" in events[2]["detail"]
    assert events[2]["remaining_seconds"] == 2
    assert "10.8 sym/s" in events[3]["detail"]
    assert "0.50 batch/s" in events[3]["detail"]
    assert "0.50 batch/s" in events[4]["detail"]


# --- Test embed_symbols ---


def test_embed_symbols_batch():
    provider = MockProvider(dims=4)
    symbols = [
        {"id": 1, "name": "foo", "content": "int foo() {}"},
        {"id": 2, "name": "bar", "content": "int bar() {}"},
        {"id": 3, "name": "baz", "content": "int baz() {}"},
    ]
    results = embed_symbols(provider, symbols, batch_size=2)
    assert len(results) == 3
    for sym_id, emb_bytes in results:
        assert isinstance(sym_id, int)
        assert isinstance(emb_bytes, bytes)
        assert len(emb_bytes) == 16  # 4 dims * 4 bytes


# --- Test DB embedding methods ---


def test_db_embeddings(tmp_path):
    """Test embedding storage and vector search in the database."""
    from srclight.db import Database, FileRecord, SymbolRecord

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    # Insert a file and symbols
    file_id = db.upsert_file(FileRecord(
        path="test.py", content_hash="abc123", mtime=1.0,
        language="python", size=100, line_count=10,
    ))

    sym1_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="hello",
        start_line=1, end_line=5, content="def hello(): pass",
        body_hash="h1",
    ), "test.py")

    sym2_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="world",
        start_line=6, end_line=10, content="def world(): pass",
        body_hash="h2",
    ), "test.py")

    db.commit()

    # Generate mock embeddings
    provider = MockProvider(dims=4)
    vec1 = provider.embed_one("hello function")
    vec2 = provider.embed_one("world function")

    db.upsert_embedding(sym1_id, provider.name, 4, vector_to_bytes(vec1), "h1")
    db.upsert_embedding(sym2_id, provider.name, 4, vector_to_bytes(vec2), "h2")
    db.commit()

    # Check stats
    stats = db.embedding_stats()
    assert stats["total_symbols"] == 2
    assert stats["embedded_symbols"] == 2
    assert stats["coverage_pct"] == 100.0
    assert stats["model"] == provider.name
    assert stats["dimensions"] == 4

    # Vector search
    query_vec = provider.embed_one("hello function")
    query_bytes = vector_to_bytes(query_vec)
    results = db.vector_search(query_bytes, 4, limit=5)
    assert len(results) == 2
    # "hello" should be top result (identical embedding)
    assert results[0]["name"] == "hello"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=0.01)

    # Check symbols needing embeddings (none — both are done)
    needing = db.get_symbols_needing_embeddings(provider.name)
    assert len(needing) == 0

    db.close()


def test_db_embeddings_incremental(tmp_path):
    """Test that changed symbols get re-embedded."""
    from srclight.db import Database, FileRecord, SymbolRecord

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    file_id = db.upsert_file(FileRecord(
        path="test.py", content_hash="abc", mtime=1.0,
        language="python", size=50, line_count=5,
    ))

    sym_id = db.insert_symbol(SymbolRecord(
        file_id=file_id, kind="function", name="func",
        start_line=1, end_line=3, content="def func(): pass",
        body_hash="v1",
    ), "test.py")
    db.commit()

    # Embed it
    provider = MockProvider(dims=4)
    vec = provider.embed_one("func")
    db.upsert_embedding(sym_id, provider.name, 4, vector_to_bytes(vec), "v1")
    db.commit()

    # No symbols needing embedding
    assert len(db.get_symbols_needing_embeddings(provider.name)) == 0

    # Now "change" the symbol (update body_hash to v2 via direct SQL)
    db.conn.execute("UPDATE symbols SET body_hash = 'v2' WHERE id = ?", (sym_id,))
    db.commit()

    # Now it should need re-embedding
    needing = db.get_symbols_needing_embeddings(provider.name)
    assert len(needing) == 1
    assert needing[0]["id"] == sym_id

    db.close()


def test_db_symbols_needing_embeddings_decode_metadata_for_wave1_backend_text(tmp_path):
    """DB embedding fetches should preserve JSON metadata for backend summary text."""
    from srclight.db import Database, FileRecord, SymbolRecord

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.open()
    db.initialize()

    file_id = db.upsert_file(FileRecord(
        path="server/src/modules/auth/auth.controller.ts",
        content_hash="backend-hash",
        mtime=1.0,
        language="typescript",
        size=200,
        line_count=20,
    ))

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="route_handler",
        name="getMe",
        signature="GET /auth/me",
        start_line=1,
        end_line=3,
        content="getMe() { return this.auth.me(); }",
        body_hash="route-v1",
        metadata={
            "framework": "nestjs",
            "resource": "route_handler",
            "http_method": "GET",
            "route_path": "/auth/me",
            "controller_path": "/auth",
        },
    ), "server/src/modules/auth/auth.controller.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="module",
        name="AuthModule",
        signature="Nest module AuthModule",
        start_line=5,
        end_line=12,
        content="export class AuthModule {}",
        body_hash="module-v1",
        metadata={
            "framework": "nestjs",
            "resource": "module",
            "imports": ["AuthModule", "ConfigModule", "TypeOrmModule"],
            "controllers": ["AuthController"],
            "providers": ["AuthService", "JwtAuthGuard"],
            "exports": ["AuthService"],
        },
    ), "server/src/modules/auth/auth.controller.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="schema",
        name="UserSchema",
        signature="mongoose schema | User | users",
        start_line=14,
        end_line=18,
        content="export const UserSchema = SchemaFactory.createForClass(User);",
        body_hash="schema-v1",
        metadata={
            "framework": "mongoose",
            "resource": "schema",
            "entity_name": "User",
            "collection_name": "users",
        },
    ), "server/src/modules/auth/auth.controller.ts")

    db.insert_symbol(SymbolRecord(
        file_id=file_id,
        kind="entity",
        name="User",
        signature="mikroorm entity | User | users",
        start_line=20,
        end_line=28,
        content="export class User { id!: number; email!: string; }",
        body_hash="entity-v1",
        metadata={
            "framework": "mikroorm",
            "resource": "entity",
            "entity_name": "User",
            "table_name": "users",
            "fields": [
                {"name": "id", "field_name": "id", "kind": "primary_key"},
                {"name": "email", "field_name": "email_address", "kind": "property"},
            ],
        },
    ), "server/src/modules/auth/auth.controller.ts")
    db.commit()

    needing = db.get_symbols_needing_embeddings("mock:test-model")

    assert len(needing) == 4
    assert all(isinstance(symbol["metadata"], dict) for symbol in needing)

    prepared = "\n".join(prepare_embedding_text(symbol) for symbol in needing)

    assert "GET /auth/me" in prepared
    assert "module AuthModule" in prepared
    assert "schema users" in prepared
    assert "mikroorm entity User" in prepared

    db.close()
