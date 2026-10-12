"""OpenRouter behind the door (ARC-17): one POST per request, and the public listings.

Plain httpx rather than an SDK: it is one POST, and the agent's dependency closure is something
the product defends. The transport sends exactly the payload the door built and checked; it
adds nothing to it. Retrying, budgeting and reading the answer are the door's.

The model listing and the zero-data-retention endpoint listing are public: reading them needs
no key and costs nothing, which is how the allowlist is proposed and refreshed (ARC-10).
"""

from __future__ import annotations

from courant.llm.errors import TransportError
from courant.llm.transport import RawResponse

API = "https://openrouter.ai/api/v1"
CHAT_URL = f"{API}/chat/completions"
KEY_URL = f"{API}/key"
MODELS_URL = f"{API}/models"
ZDR_URL = f"{API}/endpoints/zdr"

REQUEST_TIMEOUT = 120.0
LISTING_TIMEOUT = 30.0

#: OpenRouter attributes usage to these; being identifiable is also how a free-tier account
#: avoids looking abusive.
ATTRIBUTION = {"HTTP-Referer": "https://courant.dev", "X-Title": "Courant"}


def _env_key(name: str) -> str:
    """A key from the environment through ``courant.config``, the one definition of
    "configured": the same `.env` loading and placeholder rejection as everywhere else."""
    from courant.config import load_dotenv, real_key

    load_dotenv()
    return real_key(name) or ""


class OpenRouterTransport:
    """OpenRouter's OpenAI-compatible chat API."""

    name = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else _env_key("OPENROUTER_API_KEY")

    def __repr__(self) -> str:
        return "OpenRouterTransport()"                  # never the key (SEC-1)

    def send(self, payload: dict) -> RawResponse:
        if not self.api_key:
            raise TransportError("no OpenRouter key is configured", retryable=False)
        import httpx

        try:
            response = httpx.post(CHAT_URL, json=payload, headers=self._headers(),
                                  timeout=REQUEST_TIMEOUT)
        except httpx.HTTPError as exc:
            raise TransportError(f"could not reach OpenRouter ({type(exc).__name__})") from None
        return RawResponse(status=response.status_code, body=_body(response),
                           headers=dict(response.headers))

    def key_info(self) -> dict:
        """GET /api/v1/key: this key's limits, including ``free_model_daily_requests``."""
        if not self.api_key:
            raise TransportError("no OpenRouter key is configured", retryable=False)
        return _get(KEY_URL, headers=self._headers())

    def list_models(self) -> dict:
        """GET /api/v1/models, keylessly."""
        return _get(MODELS_URL)

    def zdr_endpoints(self) -> dict:
        """GET /api/v1/endpoints/zdr, keylessly: every zero-data-retention endpoint."""
        return _get(ZDR_URL)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                **ATTRIBUTION}


def public_listing() -> dict:
    """OpenRouter's public model listing and its zero-data-retention endpoint listing, read
    without a key: ``{"models": body, "zdr": body or None}``. ``zdr`` is None when that listing
    could not be read, which leaves every model's retention class unknown."""
    models = _get(MODELS_URL)
    try:
        zdr = _get(ZDR_URL)
    except TransportError:
        zdr = None
    return {"models": models, "zdr": zdr}


def _get(url: str, headers: dict | None = None) -> dict:
    import httpx

    path = url.removeprefix(API)
    try:
        response = httpx.get(url, headers=headers or {}, timeout=LISTING_TIMEOUT)
    except httpx.HTTPError as exc:
        raise TransportError(f"could not reach OpenRouter for {path} "
                             f"({type(exc).__name__})") from None
    if response.status_code != 200:
        raise TransportError(f"OpenRouter answered {response.status_code} for {path}",
                             status=response.status_code,
                             retryable=response.status_code >= 500)
    body = _body(response)
    if not isinstance(body, dict):
        raise TransportError(f"OpenRouter's answer for {path} is not a JSON object",
                             retryable=False)
    return body


def _body(response):
    try:
        return response.json()
    except ValueError:
        return {"error": {"code": response.status_code, "message": "the body is not JSON"}}
