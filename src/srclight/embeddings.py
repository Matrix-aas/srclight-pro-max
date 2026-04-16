"""Embedding providers and hybrid search for Srclight Layer 7.

Supports:
- Ollama (local, default) — zero Python ML deps, just HTTP
- OpenAI-compatible (any provider speaking /v1/embeddings) — OpenAI, Together,
  Fireworks, Mistral, Jina, vLLM, HuggingFace TEI, LiteLLM, DeepInfra, etc.
- Voyage Code 3 (API, optional) — best code retrieval quality

Architecture:
- Providers generate embeddings via HTTP APIs (no torch/transformers needed)
- Embeddings stored as float32 BLOB in SQLite symbol_embeddings table
- Cosine similarity computed in Python (numpy fast path if available)
- Hybrid search: RRF fusion of FTS5 keyword results + embedding similarity
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("srclight.embeddings")

# Default Ollama embedding model used by the bare `--embed` CLI UX.
DEFAULT_OLLAMA_EMBED_MODEL = "ollama:qwen3-embedding:4b"


def _transport_aliases(transport: str) -> list[str]:
    transport_lower = (transport or "").strip().lower()
    if not transport_lower:
        return []
    if transport_lower == "rmq":
        return ["rabbitmq"]
    return []


def _redact_connection_url(connection_url: str) -> str:
    """Strip credentials from a transport connection URL."""
    normalized = connection_url.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"', "`"}:
        normalized = normalized[1:-1].strip()

    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return normalized

    if not parsed.scheme or not parsed.hostname:
        return normalized

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _timeout_from_env(var_name: str, default: int) -> int:
    try:
        return int(os.environ.get(var_name, str(default)))
    except ValueError:
        return default


# Timeout for interactive embedding API requests (MCP tool path). Kept short so
# Cursor/IDE tool calls don't hang; indexing uses a longer timeout.
def _embed_request_timeout() -> int:
    return _timeout_from_env("SRCLIGHT_EMBED_REQUEST_TIMEOUT", 20)


def _index_embed_request_timeout() -> int:
    return _timeout_from_env("SRCLIGHT_INDEX_EMBED_REQUEST_TIMEOUT", 120)


_OLLAMA_MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_KNOWN_OLLAMA_EMBED_MODEL_TOKENS = {
    "arctic",
    "bge",
    "e5",
    "embeddinggemma",
    "gte",
    "jina",
    "minilm",
    "mxbai",
    "nomic",
    "qwen",
    "qwen3",
    "snowflake",
}


def _looks_like_ollama_model(model: str) -> bool:
    """Heuristically recognize unprefixed Ollama model strings.

    Explicit prefixes are handled separately. For unprefixed strings, accept
    plain model names like `qwen3-embedding` and tagged names like
    `qwen3-embedding:4b` for known Ollama embedding families. Arbitrary/custom
    Ollama names should use the explicit `ollama:` prefix.
    """
    if not model:
        return False
    base, sep, tag = model.partition(":")
    if not _OLLAMA_MODEL_NAME_RE.match(base):
        return False

    tokens = {token for token in re.split(r"[-_.]", base) if token}
    if not any(token in _KNOWN_OLLAMA_EMBED_MODEL_TOKENS for token in tokens):
        return False
    if not sep:
        return True
    if not tag:
        return False
    if tag == "latest":
        return True
    return any(ch.isdigit() for ch in tag)


# --- Embedding text preparation ---


def prepare_embedding_text(symbol: dict) -> str:
    """Build the text to embed for a symbol.

    Combines name, signature, doc comment, and truncated content
    to give the embedding model rich context without exceeding limits.
    """
    parts = []

    # Qualified name or plain name
    qname = symbol.get("qualified_name") or symbol.get("name") or ""
    if qname:
        parts.append(qname)

    # Signature (if different from name)
    sig = symbol.get("signature") or ""
    if sig and sig != qname:
        parts.append(sig)

    # Doc comment (natural language context)
    doc = symbol.get("doc_comment") or ""
    if doc:
        parts.append(doc.strip())

    # Content (truncated — most embedding models handle 512-8192 tokens)
    content = symbol.get("content") or ""
    if content:
        # Truncate to ~2000 chars (~500 tokens) to stay within limits
        parts.append(content[:2000])

    metadata = symbol.get("metadata") or {}
    if isinstance(metadata, dict):
        metadata_parts = []
        symbol_name = symbol.get("name") or ""
        resource = metadata.get("resource")
        framework = metadata.get("framework")

        if framework or resource:
            metadata_parts.append(" ".join(part for part in (framework, resource) if part))

        if resource == "route_handler":
            method = metadata.get("http_method")
            route_path = metadata.get("route_path")
            controller_path = metadata.get("controller_path")
            if framework and method and route_path:
                metadata_parts.append(f"{framework} route {method} {route_path}")
            if controller_path:
                metadata_parts.append(f"controller {controller_path}")

        if resource == "module" and symbol_name:
            metadata_parts.append(f"module {symbol_name}")
            for key, label in (
                ("imports", "imports"),
                ("controllers", "controllers"),
                ("providers", "providers"),
                ("exports", "exports"),
            ):
                values = metadata.get(key)
                if isinstance(values, list) and values:
                    metadata_parts.append(f"{label}: " + ", ".join(str(value) for value in values))

        if framework == "vue" and resource == "component":
            for key, label in (
                ("props", "props"),
                ("emits", "emits"),
                ("slots", "slots"),
                ("composables_used", "composables"),
                ("stores_used", "stores"),
                ("graphql_ops_used", "graphql ops"),
                ("routes_used", "routes"),
                ("css_modules", "css modules"),
                ("scoped_styles", "scoped styles"),
            ):
                values = metadata.get(key)
                if isinstance(values, list) and values:
                    metadata_parts.append(f"{label}: " + ", ".join(str(value) for value in values))

        for key, label in (
            ("entity_name", "entity"),
            ("table_name", "table"),
            ("collection_name", "collection"),
            ("repository_owner", "repository owner"),
        ):
            value = metadata.get(key)
            if value:
                metadata_parts.append(f"{label}: {value}")

        entity_name = metadata.get("entity_name") or symbol_name
        collection_name = metadata.get("collection_name")
        if framework and resource == "entity" and entity_name:
            metadata_parts.append(f"{framework} entity {entity_name}")
        if framework and resource == "schema":
            if entity_name:
                metadata_parts.append(f"{framework} schema {entity_name}")
            if collection_name:
                metadata_parts.append(f"schema {collection_name}")
        if framework and resource == "repository" and entity_name:
            metadata_parts.append(f"{framework} repository {entity_name}")
        if framework and resource == "database":
            names = metadata.get("entity_names")
            if isinstance(names, list) and names:
                metadata_parts.append(f"{framework} database " + ", ".join(str(value) for value in names))

        if resource == "microservice_handler":
            for key, label in (
                ("pattern", "pattern"),
                ("message_pattern", "message pattern"),
                ("event_pattern", "event pattern"),
                ("queue_name", "queue"),
                ("transport", "transport"),
                ("role", "role"),
            ):
                value = metadata.get(key)
                if value:
                    metadata_parts.append(f"{label}: {value}")
            role = metadata.get("role")
            queue_name = metadata.get("queue_name")
            transport = metadata.get("transport")
            if role:
                metadata_parts.append(f"{role} message handler")
            if queue_name and role:
                metadata_parts.append(f"{role} queue {queue_name}")
            if transport:
                for alias in _transport_aliases(str(transport)):
                    metadata_parts.append(f"transport: {alias}")

        if resource == "queue_processor":
            for key, label in (
                ("queue_name", "queue"),
                ("job_name", "job"),
                ("transport", "transport"),
                ("role", "role"),
            ):
                value = metadata.get(key)
                if value:
                    metadata_parts.append(f"{label}: {value}")
            role = metadata.get("role")
            queue_name = metadata.get("queue_name")
            if role:
                metadata_parts.append(f"{role} worker")
            if queue_name and role:
                metadata_parts.append(f"{role} queue {queue_name}")

        if resource == "scheduled_job":
            schedule_type = metadata.get("schedule_type")
            if schedule_type:
                metadata_parts.append(f"schedule: {schedule_type}")
            for key, label in (
                ("cron", "cron"),
                ("interval_name", "interval"),
                ("every_ms", "every ms"),
                ("timeout_name", "timeout"),
                ("delay_ms", "delay ms"),
            ):
                value = metadata.get(key)
                if value is not None:
                    metadata_parts.append(f"{label}: {value}")

        if resource == "transport":
            for key, label in (
                ("transport", "transport"),
                ("queue_name", "queue"),
                ("connection_url", "connection"),
                ("role", "role"),
            ):
                value = metadata.get(key)
                if value:
                    if key == "connection_url":
                        value = _redact_connection_url(str(value))
                    metadata_parts.append(f"{label}: {value}")

        for key, label in (
            ("entity_names", "entities"),
            ("collection_names", "collections"),
        ):
            values = metadata.get(key)
            if isinstance(values, list) and values:
                metadata_parts.append(f"{label}: " + ", ".join(str(value) for value in values))

        fields = metadata.get("fields")
        if isinstance(fields, list) and fields:
            field_summaries = []
            for field in fields:
                if not isinstance(field, dict):
                    continue
                field_name = field.get("name")
                storage_name = field.get("field_name")
                field_kind = field.get("kind")
                if field_name and storage_name and storage_name != field_name:
                    field_summaries.append(f"{field_name} -> {storage_name} ({field_kind})")
                elif field_name:
                    field_summaries.append(
                        f"{field_name} ({field_kind})" if field_kind else str(field_name)
                    )
            if field_summaries:
                metadata_parts.append("fields: " + ", ".join(field_summaries))

        if metadata_parts:
            parts.append("\n".join(metadata_parts))

    return "\n".join(parts)


# --- Provider protocol ---


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier string (stored in DB)."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output embedding dimensions."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]


# --- Ollama provider ---


class OllamaProvider(EmbeddingProvider):
    """Embed via Ollama's HTTP API (local, zero Python ML deps).

    Default model: qwen3-embedding:4b (best quality available locally).
    Alternative: nomic-embed-text-v2-moe (strong multilingual fallback).

    Ollama endpoint: http://localhost:11434 (accessible from WSL to Windows Ollama).
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:4b",
        base_url: str = "http://localhost:11434",
        timeout: int | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Probe by embedding a test string
            vec = self.embed_one("test")
            self._dimensions = len(vec)
        return self._dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama /api/embed endpoint."""
        url = f"{self._base_url}/api/embed"
        payload = json.dumps({"model": self._model, "input": texts}).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            timeout = self._timeout if self._timeout is not None else _embed_request_timeout()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            if e.code == 404 and "model" in body and "not found" in body:
                raise ConnectionError(
                    f"Ollama model '{self._model}' is not available at {self._base_url}. "
                    f"Pull it first: ollama pull {self._model}"
                ) from e
            raise ConnectionError(
                f"Ollama request to {self._base_url}/api/embed failed with HTTP {e.code}: {body}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        # Cache dimensions from first result
        if self._dimensions is None and embeddings:
            self._dimensions = len(embeddings[0])

        return embeddings

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            timeout = self._timeout if self._timeout is not None else _embed_request_timeout()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(self._matches_model(name) for name in models)
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            timeout = self._timeout if self._timeout is not None else _embed_request_timeout()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def pull_model(self) -> None:
        """Pull the model via Ollama API."""
        url = f"{self._base_url}/api/pull"
        payload = json.dumps({"name": self._model, "stream": False}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # This can take a while for large models
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()

    def _matches_model(self, tag_name: str) -> bool:
        """Match Ollama tag names against the configured model string."""
        if not tag_name:
            return False
        if tag_name == self._model:
            return True
        if self._model.startswith("ollama:") and tag_name == self._model.split(":", 1)[1]:
            return True
        if tag_name.endswith(":latest") and tag_name[:-7] == self._model:
            return True
        if tag_name.endswith(":latest") and self._model == tag_name[:-7]:
            return True
        if ":" not in self._model and tag_name.split(":", 1)[0] == self._model:
            return True
        return False


# --- OpenAI-compatible provider ---


class OpenAIProvider(EmbeddingProvider):
    """Embed via any OpenAI-compatible /v1/embeddings endpoint.

    Covers: OpenAI, Together, Fireworks, Mistral, Jina, vLLM,
    HuggingFace TEI, LiteLLM, DeepInfra, Anyscale, and more.

    Config via environment variables:
        OPENAI_API_KEY  — API key (required)
        OPENAI_BASE_URL — Base URL (default: https://api.openai.com)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        import os
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")).rstrip("/")
        self._timeout = timeout
        self._dimensions: int | None = None
        if not self._api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Probe by embedding a test string
            vec = self.embed_one("test")
            self._dimensions = len(vec)
        return self._dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via OpenAI-compatible /v1/embeddings endpoint."""
        url = f"{self._base_url}/v1/embeddings"
        payload = json.dumps({
            "model": self._model,
            "input": texts,
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            timeout = self._timeout if self._timeout is not None else _embed_request_timeout()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            raise ConnectionError(f"OpenAI API error ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach {self._base_url}. Error: {e}"
            ) from e

        results = data.get("data", [])
        embeddings = [r["embedding"] for r in sorted(results, key=lambda x: x["index"])]

        if len(embeddings) != len(texts):
            raise ValueError(
                f"API returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        # Cache dimensions from first result
        if self._dimensions is None and embeddings:
            self._dimensions = len(embeddings[0])

        return embeddings


# --- Cohere provider ---


class CohereProvider(EmbeddingProvider):
    """Embed via Cohere's v2 embed API.

    Models: embed-v4.0 (1024 dims), embed-english-v3.0, embed-multilingual-v3.0.

    Config via environment variable:
        COHERE_API_KEY — API key (required)
    """

    API_URL = "https://api.cohere.com/v2/embed"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-v4.0",
        timeout: int | None = None,
    ):
        import os
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._model = model
        self._timeout = timeout
        if not self._api_key:
            raise ValueError(
                "Cohere API key required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    @property
    def dimensions(self) -> int:
        # embed-v4.0 and embed-*-v3.0 all output 1024 dimensions
        return 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Cohere v2 embed API."""
        payload = json.dumps({
            "model": self._model,
            "texts": texts,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }).encode()

        req = urllib.request.Request(
            self.API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            timeout = self._timeout if self._timeout is not None else _embed_request_timeout()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            raise ConnectionError(f"Cohere API error ({e.code}): {body}") from e

        # v2 response: {"embeddings": {"float": [[...], [...]]}}
        embeddings_data = data.get("embeddings", {})
        if isinstance(embeddings_data, dict):
            embeddings = embeddings_data.get("float", [])
        else:
            embeddings = embeddings_data

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Cohere returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        return embeddings


# --- Voyage Code 3 provider ---


class VoyageProvider(EmbeddingProvider):
    """Embed via Voyage AI API (best code retrieval quality).

    Requires VOYAGE_API_KEY environment variable.
    Model: voyage-code-3 (1024 dims, 32K context).
    """

    API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-code-3",
        timeout: int | None = None,
    ):
        import os
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self._model = model
        self._timeout = timeout
        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return f"voyage:{self._model}"

    @property
    def dimensions(self) -> int:
        # voyage-code-3 outputs 1024 dimensions
        return 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Voyage API."""
        payload = json.dumps({
            "model": self._model,
            "input": texts,
            "input_type": "document",
        }).encode()

        req = urllib.request.Request(
            self.API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            timeout = self._timeout if self._timeout is not None else _embed_request_timeout()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            raise ConnectionError(f"Voyage API error ({e.code}): {body}") from e

        results = data.get("data", [])
        return [r["embedding"] for r in sorted(results, key=lambda x: x["index"])]


# --- Vector math (pure Python, numpy fast path) ---


def vectors_to_bytes(vectors: list[list[float]]) -> list[bytes]:
    """Convert float vectors to bytes for SQLite BLOB storage."""
    return [struct.pack(f'{len(v)}f', *v) for v in vectors]


def vector_to_bytes(vector: list[float]) -> bytes:
    """Convert a single float vector to bytes."""
    return struct.pack(f'{len(vector)}f', *vector)


def bytes_to_vector(data: bytes) -> list[float]:
    """Convert bytes back to float vector."""
    n = len(data) // 4
    return list(struct.unpack(f'{n}f', data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Uses numpy/cupy when available."""
    from .vector_math import _backend, _np

    if _np is not None:
        va = _np.asarray(a, dtype=_np.float32)
        vb = _np.asarray(b, dtype=_np.float32)
        na, nb = _np.linalg.norm(va), _np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        result = _np.dot(va, vb) / (na * nb)
        return float(result.get()) if _backend == "cupy" else float(result)
    # Pure Python fallback
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Reciprocal Rank Fusion (RRF) ---


def rrf_merge(
    fts_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    k: int = 60,
    fts_weight: float = 1.0,
    embedding_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge FTS5 and embedding search results using Reciprocal Rank Fusion.

    RRF score = sum(weight / (k + rank)) for each result list.
    k=60 is the standard value from the RRF paper (Cormack et al., 2009).

    Args:
        fts_results: Results from FTS5 search (must have 'symbol_id')
        embedding_results: Results from embedding search (must have 'symbol_id')
        k: RRF parameter (default 60)
        fts_weight: Weight for FTS results (default 1.0)
        embedding_weight: Weight for embedding results (default 1.0)

    Returns:
        Merged results sorted by combined RRF score (descending).
    """
    scores: dict[int, float] = {}
    data: dict[int, dict] = {}

    # Score FTS results by rank position
    for rank, result in enumerate(fts_results):
        sid = result["symbol_id"]
        scores[sid] = scores.get(sid, 0.0) + fts_weight / (k + rank + 1)
        if sid not in data:
            data[sid] = dict(result)
            data[sid]["sources"] = []
        data[sid]["sources"].append("fts")

    # Score embedding results by rank position
    for rank, result in enumerate(embedding_results):
        sid = result["symbol_id"]
        scores[sid] = scores.get(sid, 0.0) + embedding_weight / (k + rank + 1)
        if sid not in data:
            data[sid] = dict(result)
            data[sid]["sources"] = []
        data[sid]["sources"].append("embedding")
        # Preserve similarity score
        if "similarity" in result:
            data[sid]["similarity"] = result["similarity"]

    # Build merged results sorted by RRF score
    merged = []
    for sid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = data[sid]
        entry["rrf_score"] = round(score, 6)
        merged.append(entry)

    return merged


# --- Provider factory ---


def get_provider(model: str, **kwargs) -> EmbeddingProvider:
    """Create an embedding provider from a model specifier.

    Formats:
        "ollama:qwen3-embedding:4b" or "qwen3-embedding:4b" -> OllamaProvider
        "openai:text-embedding-3-small" -> OpenAIProvider
        "cohere:embed-v4.0" or "embed-v4.0" -> CohereProvider
        "voyage:voyage-code-3" or "voyage-code-3" -> VoyageProvider

    OpenAI-compatible providers (Together, Fireworks, Mistral, vLLM, etc.)
    use the "openai:" prefix with OPENAI_BASE_URL env var to set the endpoint.
    """
    explicit_prefixes = ("openai:", "cohere:", "voyage:", "ollama:")
    if model.startswith(explicit_prefixes):
        provider_type, model_name = model.split(":", 1)
    elif model.startswith("voyage"):
        provider_type = "voyage"
        model_name = model
    elif model.startswith("embed-") and ("v3" in model or "v4" in model):
        provider_type = "cohere"
        model_name = model
    elif model.startswith("text-embedding"):
        provider_type = "openai"
        model_name = model
    elif _looks_like_ollama_model(model):
        provider_type = "ollama"
        model_name = model
    else:
        raise ValueError(f"Unknown embedding provider: {model}")

    if provider_type == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        timeout = kwargs.get("timeout")
        return OllamaProvider(model=model_name, base_url=base_url, timeout=timeout)
    elif provider_type == "openai":
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        timeout = kwargs.get("timeout")
        return OpenAIProvider(model=model_name, api_key=api_key, base_url=base_url, timeout=timeout)
    elif provider_type == "cohere":
        api_key = kwargs.get("api_key")
        timeout = kwargs.get("timeout")
        return CohereProvider(api_key=api_key, model=model_name, timeout=timeout)
    elif provider_type == "voyage":
        api_key = kwargs.get("api_key")
        timeout = kwargs.get("timeout")
        return VoyageProvider(api_key=api_key, model=model_name, timeout=timeout)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


# --- Batch embedding for indexing ---


def embed_symbols(
    provider: EmbeddingProvider,
    symbols: list[dict],
    batch_size: int = 32,
    on_progress: Any | None = None,
) -> list[tuple[int, bytes]]:
    """Embed a list of symbols in batches.

    Args:
        provider: Embedding provider to use
        symbols: List of symbol dicts (from db.get_symbols_needing_embeddings)
        batch_size: Number of symbols per batch
        on_progress: Optional callback(batch_num, total_batches)

    Returns:
        List of (symbol_id, embedding_bytes) tuples
    """
    results = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(symbols), batch_size):
        batch = symbols[batch_idx:batch_idx + batch_size]
        texts = [prepare_embedding_text(sym) for sym in batch]

        if on_progress:
            on_progress(batch_idx // batch_size + 1, total_batches)

        try:
            vectors = provider.embed_batch(texts)
            for sym, vec in zip(batch, vectors):
                results.append((sym["id"], vector_to_bytes(vec)))
        except Exception as e:
            logger.error("Embedding batch %d failed: %s", batch_idx // batch_size + 1, e)
            # Skip failed batch, continue with next
            continue

    return results
