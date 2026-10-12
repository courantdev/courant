"""The one interface between the model door and a provider (ARC-17).

A transport is plumbing: it turns the door's payload into one exchange with a vendor and hands
back what came back. Choosing a model, retrying, budgeting, validating and refusing are the
door's, so a provider added later inherits every rule without restating one.

The door builds one OpenAI-compatible chat payload for every provider and reads one
OpenAI-shaped body back. OpenRouter speaks that shape natively; the Anthropic transport
translates both ways.

Two optional methods, which the door uses when a transport has them:

    key_info()     the parsed body of the provider's key endpoint (OpenRouter: GET
                   /api/v1/key), from which a batch reads the day's allowance (ARC-8)
    list_models()  the provider's public model listing, for an allowlist refresh (ARC-10)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RawResponse:
    """One HTTP answer: status, parsed JSON body, headers."""

    status: int
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def header(self, name: str) -> str | None:
        """A header by name, case-insensitively (HTTP header names are)."""
        wanted = name.lower()
        for key, value in (self.headers or {}).items():
            if str(key).lower() == wanted:
                return str(value)
        return None


class Transport(Protocol):
    """A provider behind the door. ``send`` raises only when no HTTP answer arrived at all."""

    name: str

    def send(self, payload: dict) -> RawResponse:
        ...
