# SPDX-License-Identifier: MPL-2.0
"""
Ollama-based embedding function for ChromaDB.

Computes embeddings via a local Ollama server instead of loading
sentence-transformers into the Context Engine process. This keeps the
Context Engine process lightweight while allowing stronger, more capable
local models for embeddings.
"""

import logging
from .provider_privacy import valid_query_embedding
from typing import List, Optional

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    """Embedding function that calls a local Ollama server."""

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

    @staticmethod
    def is_legacy() -> bool:
        return False

    @staticmethod
    def supported_spaces() -> list:
        return ["cosine"]

    def get_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    @staticmethod
    def build_from_config(config: dict):
        return OllamaEmbeddingFunction(
            model_name=config.get("model_name", "nomic-embed-text:latest"),
            base_url=config.get("base_url", "http://localhost:11434"),
            timeout=config.get("timeout", 30.0),
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Hardening B (2026-06-17): on any failure (network, model
        missing, empty response), RAISE a clear error instead of
        returning an empty vector. An empty vector breaks the chroma
        collection on the next add() with "Collection expecting
        embedding with dimension of 768, got 0", which masks the
        real cause. Fail loud, fail early.
        """
        import httpx

        embeddings = []
        errors = []
        for text in input:
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=self.timeout,
                    trust_env=False,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not valid_query_embedding([embedding]):
                    raise RuntimeError(
                        "Ollama returned no embedding; check the configured model."
                    )
                embeddings.append(embedding)
            except Exception as e:
                # Collect errors instead of swallowing them. Raise at
                # the end so a partial batch surfaces every failure.
                errors.append(type(e).__name__)
                logger.error("Ollama embedding failed; content and provider details withheld")

        if errors:
            # Surface all failures clearly. Don't write garbage to the
            # vector store.
            raise RuntimeError(
                f"Ollama embedding failed for {len(errors)}/{len(input)} "
                f"input(s) ({errors[0]}). Check provider/model availability; details withheld."
            )
        return embeddings

    def embed_query(self, input: str | List[str]) -> List[List[float]]:
        """ChromaDB-compatible query embedding entry point."""
        if isinstance(input, str):
            input = [input]
        return self(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """ChromaDB-compatible document embedding entry point."""
        return self(input)

    @staticmethod
    def check_availability(base_url: str = "http://localhost:11434", timeout: float = 5.0) -> bool:
        """Check if Ollama server is reachable."""
        try:
            import httpx
            from .provider_privacy import validated_provider_url
            response = httpx.get(f"{validated_provider_url(base_url)}/api/tags", timeout=timeout, trust_env=False)
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def check_readiness(
        model_name: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        timeout: float = 10.0,
    ) -> dict:
        """Hardening E: probe whether the embedding model is actually
        ready to serve. Returns a dict the MCP server can log and the
        dashboard can display.

        Checks in order:
          1. Ollama server reachable at base_url
          2. Model listed in /api/tags (was it pulled?)
          3. Model responds to a tiny /api/embeddings call

        Returns: {
            "ready": bool,
            "server_reachable": bool,
            "model_pulled": bool,
            "model_responds": bool,
            "model_name": str,
            "base_url": str,
            "error": str | None,
            "remediation": str | None,
        }
        """
        result = {
            "ready": False,
            "server_reachable": False,
            "model_pulled": False,
            "model_responds": False,
            "model_name": model_name,
            "base_url": "withheld until validated",
            "error": None,
            "remediation": None,
        }
        try:
            import httpx

            from .provider_privacy import validated_provider_url
            base_url = validated_provider_url(base_url)
            result['base_url'] = base_url

            # 1. Server reachable
            try:
                r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout, trust_env=False)
                r.raise_for_status()
                result["server_reachable"] = True
            except Exception as e:
                result["error"] = "Ollama server unavailable; details withheld"
                result["remediation"] = (
                    f"Start the Ollama server: ollama serve  "
                    f"(then in another terminal: ollama pull {model_name})"
                )
                return result

            # 2. Model pulled
            try:
                tags = r.json().get("models", []) or []
                # Match by either the full name or the name without tag
                base_name = model_name.split(":")[0]
                model_pulled = any(
                    m.get("name", "").split(":")[0] == base_name for m in tags
                )
                result["model_pulled"] = model_pulled
                if not model_pulled:
                    result["error"] = "Configured model is not pulled; provider inventory withheld"
                    result["remediation"] = f"ollama pull {model_name}"
                    return result
            except Exception as e:
                result["error"] = "Invalid provider model response; details withheld"
                return result

            # 3. Model actually responds to an embed call
            try:
                er = httpx.post(
                    f"{base_url.rstrip('/')}/api/embeddings",
                    json={"model": model_name, "prompt": "ping"},
                    timeout=timeout,
                    trust_env=False,
                )
                er.raise_for_status()
                embedding = er.json().get("embedding") or []
                if not valid_query_embedding([embedding]):
                    result["error"] = (
                        f"Model '{model_name}' responded but returned no "
                        f"embedding vector"
                    )
                    return result
                result["model_responds"] = True
                result["ready"] = True
            except Exception as e:
                result["error"] = (
                    "Embedding probe failed; details withheld"
                )
                result["remediation"] = (
                    f"Try: ollama pull {model_name}  (the model may be "
                    f"corrupted or partially downloaded)"
                )
                return result

            return result
        except Exception as e:
            result["error"] = "Provider readiness failed; details withheld"
            return result
