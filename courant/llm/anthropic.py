"""Claude behind the door (ARC-17), through the official SDK: the one importer of ``anthropic``.

Anthropic has no free tier, so nothing of it is on an allowlist. Claude is reached only when a
caller pins a Claude id and S has set ``COURANT_ALLOW_PAID_LLM=1``, and never with
customer-derived data: its data-retention class is not recorded anywhere in this repository,
and ARC-9 treats unknown as training on inputs.

The transport translates both ways. The door's OpenAI-compatible payload becomes a Messages
request: the system prompt is cached (it is byte-identical on every call), thinking is
adaptive, and the schema goes in ``output_config.format`` so it constrains generation rather
than being corrected afterwards, with every object closed (``closed_schema``), as the API
requires of a structured-output schema. The answer comes back in the shape the door reads from every
provider, with ``usage.prompt_tokens`` the uncached input (Anthropic's ``input_tokens``), cache
reads as ``usage.prompt_tokens_details.cached_tokens`` and cache writes as
``usage.prompt_tokens_details.cache_write_tokens``: the counts the door prices a paid Claude
request from (``prices.cost``, ARC-21), since Anthropic reports no cost. A refusal is an
HTTP 200 with ``stop_reason: "refusal"``; it is translated into a refusal the door reads as
such, never into an empty answer someone might call a diagnosis.

A key that is not scoped to a workspace needs every request to name one: with
``ANTHROPIC_WORKSPACE_ID`` set (``courant.config.anthropic_workspace_id``), the client sends it
as the ``anthropic-workspace-id`` header. An error answer is handed to the door with its status,
the SDK's error type (``invalid_request_error``) and, when the transport recognises why, a key
of ``errors.REJECTION_REASONS``; never the provider's message, which may quote the request.
"""

from __future__ import annotations

from courant.llm.errors import REJECTION_REASONS, TransportError
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

    def __init__(self, api_key: str | None = None, *, workspace_id: str | None = None,
                 client=None) -> None:
        self.api_key = api_key if api_key is not None else _env_key("ANTHROPIC_API_KEY")
        self.workspace_id = workspace_id if workspace_id is not None else _workspace()
        self._client = client

    def __repr__(self) -> str:
        return "AnthropicTransport()"                   # never the key (SEC-1)

    def send(self, payload: dict) -> RawResponse:
        system, messages = _split(payload.get("messages") or [])
        output_config: dict = {"effort": "high"}
        schema = ((payload.get("response_format") or {}).get("json_schema") or {}).get("schema")
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": closed_schema(schema)}
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

    def headers(self) -> dict[str, str]:
        """The headers every request carries besides the SDK's own: the workspace, if set."""
        return {WORKSPACE_HEADER: self.workspace_id} if self.workspace_id else {}

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # lazy: only when Claude is actually called
            except ImportError:
                raise TransportError("the anthropic package is not installed",
                                     retryable=False) from None
            # max_retries=0: the door's retry policy is the only one, so every attempt is
            # reserved in the ledger (and under a spend cap, bounded) before it leaves (ARC-18,
            # ARC-21). The SDK's own retries would send requests the ledger never saw.
            self._client = anthropic.Anthropic(api_key=self.api_key or None,
                                               default_headers=self.headers(), max_retries=0)
        return self._client


#: The header naming the workspace a request acts in.
WORKSPACE_HEADER = "anthropic-workspace-id"

#: Why Anthropic rejected a request, recognised by a phrase its error message uses (any one of
#: them), each mapped to a key of ``REJECTION_REASONS`` so only fixed words leave this module
#: (SEC-1). The workspace message, as Anthropic answered it on 12 Oct 2026, begins "This API
#: key is not scoped to a workspace, so this request must include the anthropic-workspace-id
#: header".
_REASONS = (("workspace_required", ("not scoped to a workspace", "anthropic-workspace-id")),)


def _workspace() -> str | None:
    from courant.config import anthropic_workspace_id

    return anthropic_workspace_id()


def closed_schema(schema):
    """``schema`` as Anthropic's structured outputs accept it: every object closed
    (``additionalProperties: false``, which the API requires of each object and answers a 400
    without). A copy; the door still validates the answer against the caller's own schema.
    An object that states ``additionalProperties`` keeps what it states."""
    if isinstance(schema, list):
        return [closed_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {key: closed_schema(value) if key in ("properties", "items", "anyOf", "allOf",
                                                 "$defs", "definitions") else value
           for key, value in schema.items()}
    if isinstance(out.get("properties"), dict):
        out["properties"] = {name: closed_schema(sub)
                             for name, sub in schema["properties"].items()}
    if isinstance(out.get("$defs"), dict):
        out["$defs"] = {name: closed_schema(sub) for name, sub in schema["$defs"].items()}
    kinds = out.get("type")
    is_object = kinds == "object" or (isinstance(kinds, list) and "object" in kinds) \
        or "properties" in out
    if is_object and "additionalProperties" not in out:
        out["additionalProperties"] = False
    return out


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
    error = {"code": status, "message": type(exc).__name__}
    kind = getattr(exc, "type", None)
    if isinstance(kind, str) and kind:
        error["type"] = kind                            # the door keeps it only if well-formed
    reason = _reason(exc)
    if reason is not None:
        error["courant_reason"] = reason
    return RawResponse(status=status, headers=headers, body={"error": error})


def _reason(exc: Exception) -> str | None:
    """A key of ``REJECTION_REASONS`` for why Anthropic rejected the request, or None. Read
    from the error body's message, which is matched and never passed on."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return None
    lowered = message.lower()
    for reason, phrases in _REASONS:
        if reason in REJECTION_REASONS and any(phrase in lowered for phrase in phrases):
            return reason
    return None


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
                  "prompt_tokens_details": {
                      "cached_tokens": tokens("cache_read_input_tokens"),
                      "cache_write_tokens": tokens("cache_creation_input_tokens")},
                  "cost": None},
    }
