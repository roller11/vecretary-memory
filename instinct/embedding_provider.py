# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Embedding Provider Abstraction

Modular, switchable embedding system. Each provider is independent and
can be enabled/disabled via config. The factory tries providers in priority
order and falls back gracefully.

Design goals:
  - Full-featured local: Ollama with nomic-embed-text (GPU, strong model)
  - Optional local sentence-transformers provider for explicitly configured adapters
  - No persisted provider selection or backend-specific embedding reconstruction

Provider priority (default):
  1. ollama  — local GPU server, strongest model, lightweight process
  2. sentence_transformers — optional local model, needs torch + transformers

Each provider is a class with:
  - name() -> str
  - is_available() -> bool  (cheap probe, no side effects)
  - __call__(input: List[str]) -> List[List[float]]
  - embed_query(text) -> List[float]  (single-query convenience)
  - embed_documents(texts) -> List[List[float]]

The factory get_embedding_provider(config) returns the first available
provider, or raises if none are available. Never delegate provider selection
to persisted collection configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from .provider_privacy import valid_query_embedding

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Provider selection: ordered list of provider names to try.
    # Set via env CE_EMBEDDING_PROVIDERS or override in code.
    "providers": ["ollama", "sentence_transformers"],

    # Ollama settings
    "ollama": {
        "model_name": os.environ.get("CE_OLLAMA__EMBED_MODEL", "nomic-embed-text:latest"),
        "base_url": os.environ.get("CE_OLLAMA__HOST", "http://localhost:11434"),
        "timeout": float(os.environ.get("CE_OLLAMA__EMBED_TIMEOUT", "30.0")),
    },

    # Sentence-transformers settings (heavy, needs torch)
    "sentence_transformers": {
        "model_name": os.environ.get("CE_ST__EMBED_MODEL", "all-MiniLM-L6-v2"),
    },

    # Remote HTTP embedding API (future, for cloud deployments)
    "remote": {
        "url": os.environ.get("CE_EMBED__REMOTE_URL", ""),
        "api_key": os.environ.get("CE_EMBED__REMOTE_KEY", ""),
        "model_name": os.environ.get("CE_EMBED__REMOTE_MODEL", ""),
        "timeout": float(os.environ.get("CE_EMBED__REMOTE_TIMEOUT", "30.0")),
    },
}


def get_config() -> Dict[str, Any]:
    """Get the embedding config, allowing env overrides."""
    config = DEFAULT_CONFIG.copy()

    # Allow env to override the provider priority list
    env_providers = os.environ.get("CE_EMBEDDING_PROVIDERS")
    if env_providers:
        config["providers"] = [
            p.strip() for p in env_providers.split(",") if p.strip()
        ]

    return config


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update the global config in-place. Returns the new config."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = _deep_merge(DEFAULT_CONFIG, updates)
    return DEFAULT_CONFIG


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# =============================================================================
# PROVIDER INTERFACE
# =============================================================================

class EmbeddingProvider:
    """Base interface for explicitly configured embedding providers."""

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def is_legacy() -> bool:
        return False

    @staticmethod
    def supported_spaces() -> list:
        return ["cosine"]

    def get_config(self) -> dict:
        return {"provider": self.name()}

    @staticmethod
    def build_from_config(config: dict) -> "EmbeddingProvider":
        raise NotImplementedError

    def is_available(self) -> bool:
        """Cheap probe — can this provider serve embeddings right now?
        Should NOT raise. Should NOT load heavy models."""
        return True

    def __call__(self, input: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, input: str | List[str]) -> List[List[float]]:
        """Return embeddings for one or more query strings."""
        if isinstance(input, str):
            input = [input]
        return self(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """Return embeddings for one or more documents."""
        return self(input)


# =============================================================================
# PROVIDER: OLLAMA (local GPU server, lightweight process)
# =============================================================================

class OllamaProvider(EmbeddingProvider):
    """Embedding via local Ollama server. Lightweight: the model lives
    in the Ollama process, not in ours. Best for local dev with a GPU."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        from .provider_privacy import validated_provider_url
        self.base_url = validated_provider_url(base_url)
        self.timeout = timeout

    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> dict:
        return {
            "provider": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    @staticmethod
    def build_from_config(config: dict) -> "OllamaProvider":
        return OllamaProvider(
            model_name=config.get("model_name", "nomic-embed-text:latest"),
            base_url=config.get("base_url", "http://localhost:11434"),
            timeout=config.get("timeout", 30.0),
        )

    def is_available(self) -> bool:
        """Check if Ollama server is reachable. Does NOT check if the
        model is pulled (that's a heavier probe done in check_readiness)."""
        try:
            import httpx
            r = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5.0,
                trust_env=False,
            )
            return r.status_code == 200
        except Exception:
            return False

    def check_readiness(self) -> dict:
        """Full readiness probe: server reachable + model pulled +
        model responds to an embed call. Returns a diagnostic dict."""
        result = {
            "ready": False,
            "server_reachable": False,
            "model_pulled": False,
            "model_responds": False,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "error": None,
            "remediation": None,
        }
        try:
            import httpx

            # 1. Server reachable
            try:
                r = httpx.get(f"{self.base_url}/api/tags", timeout=10.0, trust_env=False)
                r.raise_for_status()
                result["server_reachable"] = True
            except Exception as e:
                result["error"] = "Ollama server unavailable; provider details withheld"
                result["remediation"] = (
                    f"Start Ollama: ollama serve  (then: ollama pull {self.model_name})"
                )
                return result

            # 2. Model pulled
            try:
                tags = r.json().get("models", []) or []
                base_name = self.model_name.split(":")[0]
                model_pulled = any(
                    m.get("name", "").split(":")[0] == base_name for m in tags
                )
                result["model_pulled"] = model_pulled
                if not model_pulled:
                    result["error"] = "Configured model is not pulled; provider inventory withheld"
                    result["remediation"] = f"ollama pull {self.model_name}"
                    return result
            except Exception as e:
                result["error"] = "Invalid provider model response; details withheld"
                return result

            # 3. Model responds to embed call
            try:
                er = httpx.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": "ping"},
                    timeout=self.timeout,
                    trust_env=False,
                )
                er.raise_for_status()
                embedding = er.json().get("embedding") or []
                if not valid_query_embedding([embedding]):
                    result["error"] = f"Model responded but returned no embedding"
                    return result
                result["model_responds"] = True
                result["ready"] = True
            except Exception as e:
                result["error"] = "Embedding probe failed; details withheld"
                result["remediation"] = f"Try: ollama pull {self.model_name}"
                return result

            return result
        except Exception as e:
            result["error"] = "Provider readiness failed; details withheld"
            return result

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings. RAISES on failure — never returns empty
        vectors (which violate the SQLite embedding identity).

        Uses the modern /api/embed endpoint with truncate=True so long
        documents are truncated SERVER-SIDE instead of failing with
        "input length exceeds the context length" (nomic-embed-text has
        a 2048-token context; a 7KB doc used to 500 the whole upsert).
        Falls back to the legacy /api/embeddings endpoint (no truncation)
        if /api/embed is unavailable (Ollama < 0.3).
        """
        import httpx

        embeddings = []
        errors = []
        legacy_fallback = False
        for text in input:
            try:
                # Modern endpoint: batch-of-one with server-side truncation.
                try:
                    response = httpx.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model_name,
                            "input": text,
                            "truncate": True,
                        },
                        timeout=self.timeout,
                        trust_env=False,
                    )
                    response.raise_for_status()
                    data = response.json()
                    batch = data.get("embeddings") or []
                    embedding = batch[0] if batch else None
                except httpx.HTTPStatusError as e:
                    # 404 => old Ollama without /api/embed; retry legacy.
                    if e.response.status_code == 404:
                        legacy_fallback = True
                        response = httpx.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": self.model_name, "prompt": text},
                            timeout=self.timeout,
                            trust_env=False,
                        )
                        response.raise_for_status()
                        data = response.json()
                        embedding = data.get("embedding")
                    else:
                        raise
                if not valid_query_embedding([embedding]):
                    raise RuntimeError(
                        "Ollama returned no embedding; check the configured model."
                    )
                embeddings.append(embedding)
            except Exception as e:
                errors.append(type(e).__name__)
                logger.error("Ollama embedding failed; content and provider details withheld")

        if errors:
            raise RuntimeError(
                f"Ollama embedding failed for {len(errors)}/{len(input)} "
                f"input(s) ({errors[0]}). Check provider/model availability; details withheld."
            )
        return embeddings


# =============================================================================
# PROVIDER: SENTENCE TRANSFORMERS (optional, needs torch)
# =============================================================================

class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence-transformers embedder. Heavy: loads torch + transformers
    into the process. Use only when explicitly selected and adequately resourced."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._fn = None  # lazy init

    @staticmethod
    def name() -> str:
        return "sentence_transformers"

    def get_config(self) -> dict:
        return {"provider": "sentence_transformers", "model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "SentenceTransformerProvider":
        return SentenceTransformerProvider(
            model_name=config.get("model_name", "all-MiniLM-L6-v2")
        )

    def is_available(self) -> bool:
        """Check if sentence_transformers is importable."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._fn is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._fn = SentenceTransformer(self.model_name, trust_remote_code=False)
            logger.info(f"SentenceTransformer embedding loaded ({self.model_name})")
        except Exception:
            raise RuntimeError(
                "Failed to load SentenceTransformer; details withheld. "
                f"Install: pip install sentence-transformers torch"
            )

    def __call__(self, input: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        return self._fn.encode(input, convert_to_numpy=True).tolist()


# =============================================================================
# FACTORY
# =============================================================================

# Registry of provider classes by name
_PROVIDERS: Dict[str, type] = {
    "ollama": OllamaProvider,
    "sentence_transformers": SentenceTransformerProvider,
}


def register_provider(name: str, cls: type) -> None:
    """Register a custom provider class."""
    _PROVIDERS[name] = cls


def get_embedding_provider(
    config: Optional[Dict[str, Any]] = None,
) -> EmbeddingProvider:
    """Get the first available embedding provider, in priority order.

    Tries each provider in config["providers"] order. Returns the first
    one whose is_available() returns True. Fails closed if none is available.

    Args:
        config: Optional config override. Uses get_config() if None.

    Returns:
        An explicit EmbeddingProvider instance.
    """
    cfg = config or get_config()
    provider_names = cfg.get("providers", ["ollama", "sentence_transformers"])

    tried = []
    for name in provider_names:
        cls = _PROVIDERS.get(name)
        if cls is None:
            logger.warning("Unknown embedding provider configured; value withheld")
            continue

        provider_config = cfg.get(name, {})
        try:
            provider = cls.build_from_config(provider_config)
        except Exception as e:
            logger.warning("Failed to build registered embedding provider; details withheld")
            tried.append(f"{name} (build failed)")
            continue

        try:
            if provider.is_available():
                logger.info(f"Using embedding provider: {name}")
                return provider
            else:
                logger.info(f"Embedding provider '{name}' not available, trying next")
                tried.append(f"{name} (not available)")
        except Exception as e:
            logger.warning("Registered provider availability failed; details withheld")
            tried.append(f"{name} (probe failed)")

    logger.warning("No configured embedding provider available; refusing implicit collection fallback")
    raise RuntimeError("No configured embedding provider available. Review local provider readiness.")


def list_providers() -> List[Dict[str, Any]]:
    """List all registered providers with their availability status.
    Useful for the dashboard /api/health endpoint."""
    cfg = get_config()
    result = []
    for name, cls in _PROVIDERS.items():
        provider_config = cfg.get(name, {})
        try:
            provider = cls.build_from_config(provider_config)
            available = provider.is_available()
        except Exception as e:
            available = False
        result.append({
            "name": name,
            "available": available,
            "class": cls.__name__,
        })
    return result
