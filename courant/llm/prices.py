"""What a paid request costs (ARC-21): one price table, the cost of an answered request from its
usage, and an upper bound on a request's cost before it is sent.

OpenRouter reports each request's cost in its answer, and the door records that. Anthropic does
not, so a paid Claude request's cost is computed here from the four token counts its answer
carries, each at its own rate:

    uncached input      ``usage.prompt_tokens`` (Anthropic's ``input_tokens``)
    cache writes        ``usage.prompt_tokens_details.cache_write_tokens``
                        (``cache_creation_input_tokens``, 5-minute entries)
    cache reads         ``usage.prompt_tokens_details.cached_tokens``
                        (``cache_read_input_tokens``: a fraction of input that differs by model)
    output              ``usage.completion_tokens`` (thinking included)

The Anthropic transport marks its system prompt ``cache_control: ephemeral`` (5 minutes), so a
request may write or read the cache; nothing it sends asks for the 1-hour entry. The prompt's size
is the sum of the three input counts; a model with a long-context tier (Haiku 5.5, above 100,000
prompt tokens) bills the whole request at that tier's rates.

The table is Anthropic's own pricing page, https://platform.claude.com/docs/en/about-claude/pricing,
read 13 Oct 2026 (``SOURCE``), in US dollars per million tokens: input, 5-minute and 1-hour cache
writes, cache hits, output, and Haiku 5.5's tier above 100,000 prompt tokens. The page's 1.1 times
multiplier for US-only inference (``inference_geo``) does not apply: the transport never sets it.
A model that is not in the table has no price: a paid Anthropic request to it is refused before it
is sent, so no paid request is ever recorded without its cost (ARC-21).

The bound (``request_bound``) is what the spend cap reserves before a request leaves: every byte
of the payload's text as one input token (a token is at least one byte), plus a fixed overhead for
what the provider adds (role markers, the structured-output instruction), all priced as a
5-minute cache write, the dearest input class the transport can be billed (it never writes a
1-hour entry), and at the long-context tier whenever the bound is over its threshold, so a prompt
that could exceed 100,000 tokens is always bound at the higher tier;
and the request's whole ``max_tokens`` as output. It is never below the request's real cost while
the table is right.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

#: Where every number of ``PRICES`` was read, and when.
SOURCE = ("Anthropic pricing page, https://platform.claude.com/docs/en/about-claude/pricing, "
          "read 13 Oct 2026")

#: Tokens a request is assumed to carry beyond its own text, in the bound only.
OVERHEAD_TOKENS = 4096


@dataclass(frozen=True)
class Rates:
    """US dollars per million tokens."""

    input: float
    cache_write: float              # a 5-minute entry: what the transport writes
    cache_write_1h: float           # a 1-hour entry: never written by the transport
    cache_read: float
    output: float


@dataclass(frozen=True)
class Price:
    """A model's rates, and the long-context tier's above ``long_above`` prompt tokens."""

    base: Rates
    long: Rates | None = None
    long_above: int | None = None

    def rates(self, prompt_tokens: int) -> Rates:
        if self.long is not None and self.long_above is not None \
                and prompt_tokens > self.long_above:
            return self.long
        return self.base


#: (provider, model id) -> price. Source: the module docstring.
PRICES: dict[tuple[str, str], Price] = {
    ("anthropic", "claude-haiku-5-5"): Price(
        base=Rates(input=0.10, cache_write=0.125, cache_write_1h=0.20, cache_read=0.01,
                   output=0.50),
        long=Rates(input=0.50, cache_write=0.625, cache_write_1h=1.00, cache_read=0.05,
                   output=2.50),
        long_above=100_000),
    ("anthropic", "claude-sonnet-5-5"): Price(
        base=Rates(input=2.00, cache_write=2.50, cache_write_1h=4.00, cache_read=0.10,
                   output=10.00)),
    ("anthropic", "claude-opus-5-5"): Price(
        base=Rates(input=4.00, cache_write=5.00, cache_write_1h=8.00, cache_read=0.20,
                   output=20.00)),
}

#: Providers whose answers carry no cost, so whose paid requests are priced here.
PRICED_HERE = frozenset({"anthropic"})


def price_of(provider: str, model: str) -> Price | None:
    return PRICES.get((provider, model))


def needs_table(provider: str) -> bool:
    """Whether a paid request to ``provider`` is costed from this table (its answers say no
    cost of their own)."""
    return provider in PRICED_HERE


def _count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def usage_tokens(usage: Mapping | None) -> dict[str, int]:
    """The four counts of an answer's ``usage`` (the door's OpenAI shape), missing ones 0."""
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    return {"input": _count(usage.get("prompt_tokens")),
            "cache_write": _count(details.get("cache_write_tokens")),
            "cache_read": _count(details.get("cached_tokens")),
            "output": _count(usage.get("completion_tokens"))}


def cost(price: Price, usage: Mapping | None) -> float:
    """What one answered request cost, in US dollars, from its usage."""
    n = usage_tokens(usage)
    rates = price.rates(n["input"] + n["cache_write"] + n["cache_read"])
    dollars = (n["input"] * rates.input + n["cache_write"] * rates.cache_write
               + n["cache_read"] * rates.cache_read + n["output"] * rates.output) / 1e6
    return round(dollars, 9)


def payload_bytes(payload: Mapping) -> int:
    """Every byte of text a payload sends: its messages and its schema, as UTF-8."""
    text = [str(m.get("content") or "") for m in payload.get("messages") or ()
            if isinstance(m, Mapping)]
    schema = ((payload.get("response_format") or {}).get("json_schema") or {}).get("schema")
    if schema is not None:
        text.append(json.dumps(schema, separators=(",", ":")))
    return sum(len(t.encode("utf-8")) for t in text)


def request_bound(price: Price, payload: Mapping) -> float:
    """The most one request can cost, in US dollars (module docstring). Rounded up."""
    tokens_in = payload_bytes(payload) + OVERHEAD_TOKENS
    out = payload.get("max_tokens")
    out = out if isinstance(out, int) and not isinstance(out, bool) and out > 0 else 0
    rates = price.rates(tokens_in)
    dollars = (tokens_in * max(rates.input, rates.cache_write) + out * rates.output) / 1e6
    return math.ceil(dollars * 1e9) / 1e9
