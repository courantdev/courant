"""Claude behind the door (ARC-17), through the official SDK: the one importer of ``anthropic``.

Anthropic has no free tier, so nothing of it is on an allowlist. Claude is reached only when a
caller pins a Claude id and S has set ``COURANT_ALLOW_PAID_LLM=1``, and never with
customer-derived data: its data-retention class is not recorded anywhere in this repository,
and ARC-9 treats unknown as training on inputs.

The transport translates both ways. The door's OpenAI-compatible payload becomes a Messages
request: the system prompt is cached (it is byte-identical on every call), thinking is
adaptive, and the schema goes in ``output_config.format`` so it constrains generation rather
than being corrected afterwards. The answer comes back in the shape the door reads from every
provider, with cache reads as ``usage.prompt_tokens_details.cached_tokens``. A refusal is an
HTTP 200 with ``stop_reason: "refusal"``; it is translated into a refusal the door reads as
such, never into an empty answer someone might call a diagnosis.
"""

from __future__ import annotations

from courant.llm.errors import TransportError
from courant.llm.transport import RawResponse

#: Anthropic's stop reasons, as OpenAI finish reasons.
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
           "refusal": "content_filter", "tool_use": "tool_calls"}


def _env_key(name: str) -> str:
    from courant.config import load_dotenv, real_key

    load_dotenv()
    return real_key(name) or ""


class AnthropicTransport:
    """Claude via the Messages API. Paid: never chosen by the door on its own."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, *, client=None) -> None:
        self.api_key = api_key if api_key is not None else _env_key("ANTHROPIC_API_KEY")
        self._client = client

    def __repr__(self) -> str:
        return "AnthropicTransport()"                   # never the key (SEC-1)

    def send(self, payload: dict) -> RawResponse:
        system, messages = _split(payload.get("messages") or [])
        output_config: dict = {"effort": "high"}
        schema = ((payload.get("response_format") or {}).get("json_schema") or {}).get("schema")
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        client = self._get_client()
        try:
            response = client.messages.create(
                model=payload["model"],
                max_tokens=payload["max_tokens"],
                # Adaptive on this model family: the model decides how much to think.
                # `budget_tokens`, `temperature`, `top_p` and `top_k` return a 400 here, so
                # their absence is correctness, not tidiness.
                thinking={"type": "adaptive"},
                output_config=output_config,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 — mapped onto an HTTP answer below
            return _as_answer(exc)
        return RawResponse(status=200, body=_openai_shape(response, payload["model"]))

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # lazy: only when Claude is actually called
            except ImportError:
                raise TransportError("the anthropic package is not installed",
                                     retryable=False) from None
            self._client = anthropic.Anthropic(api_key=self.api_key or None)
        return self._client


def _split(messages: list) -> tuple[str, list[dict]]:
    """The OpenAI message list as Anthropic's ``system`` plus ``messages``."""
    system = "\n\n".join(str(m.get("content") or "") for m in messages
                         if m.get("role") == "system")
    rest = [{"role": m.get("role"), "content": m.get("content")} for m in messages
            if m.get("role") != "system"]
    return system, rest


def _as_answer(exc: Exception) -> RawResponse:
    """An SDK exception as the HTTP answer it was. With no answer at all, a TransportError.

    The message is the exception's class, not its text, which may quote the request.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        raise TransportError(f"could not reach Anthropic ({type(exc).__name__})") from None
    headers = {}
    response = getattr(exc, "response", None)
    retry_after = getattr(response, "headers", {}).get("retry-after") if response is not None \
        else None
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    return RawResponse(status=status, headers=headers,
                       body={"error": {"code": status, "message": type(exc).__name__}})


def _openai_shape(response, model: str) -> dict:
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        message = {"role": "assistant", "content": "",
                   "refusal": f"declined ({category or 'unspecified'})"}
    else:
        text = "".join(getattr(block, "text", "") or "" for block in
                       (getattr(response, "content", None) or ())
                       if getattr(block, "type", None) == "text")
        message = {"role": "assistant", "content": text}

    usage = getattr(response, "usage", None)

    def tokens(name: str) -> int:
        value = getattr(usage, name, 0) if usage is not None else 0
        return value if isinstance(value, int) else 0

    prompt, completion = tokens("input_tokens"), tokens("output_tokens")
    return {
        "id": getattr(response, "id", None),
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": _FINISH.get(stop, stop)}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion,
                  "prompt_tokens_details": {"cached_tokens": tokens("cache_read_input_tokens")},
                  "cost": None},
    }
