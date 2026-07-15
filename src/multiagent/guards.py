"""No-fallback / no-hiding runtime guards (plan §8, R1).

Small, loud assertions that make the honesty contract executable rather than
aspirational. They raise typed errors that the CLI surfaces with the exact cause;
nothing here silently swallows a problem into a default.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


def ensure_local_no_proxy(base_url: str) -> None:
    """Exclude the (local) Ollama host from any configured HTTP(S) proxy.

    On corporate networks HTTP_PROXY/HTTPS_PROXY are set and intercept *every*
    request — including localhost — which makes the local Ollama server return
    403 Forbidden. urllib, httpx and requests all honour the ``no_proxy`` env var,
    so we append the Ollama host (and the loopback aliases) to it before any LLM
    call. Idempotent; only ever *adds* bypass entries, never touches the proxy URLs.
    """
    host = urlparse(base_url).hostname or "localhost"
    wanted = {host, "localhost", "127.0.0.1", "::1"}
    for var in ("no_proxy", "NO_PROXY"):
        current = {h.strip() for h in os.environ.get(var, "").split(",") if h.strip()}
        merged = current | wanted
        os.environ[var] = ",".join(sorted(merged))


class EvalModeLLMError(RuntimeError):
    """Raised when an LLM call is reached while evaluation_mode is on.

    Eval numbers must be byte-reproducible and free of non-determinism, so any LLM
    invocation during eval is a hard error (§8.5), not a silent skip.
    """


def assert_llm_allowed(config, where: str) -> None:
    """Guard placed immediately before every LLM invocation."""
    if getattr(config, "evaluation_mode", False):
        raise EvalModeLLMError(
            f"LLM call reached in evaluation_mode at {where!r}. Eval mode is LLM-free; "
            f"this indicates a node took its LLM branch when it should have been "
            f"deterministic — fix the branch, do not relax the guard."
        )
