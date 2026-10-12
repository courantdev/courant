"""The models the door may call, per provider (ARC-6, ARC-10, ARC-17).

Free is decided here, by the exact ``:free`` suffix and by being on this list, never by a price
field: some paid ids report a price of "0" (``policy.load_allowlist`` refuses anything else).
A paid id is never listed; it is reached only by a caller pinning it with S's
``COURANT_ALLOW_PAID_LLM=1``.

Each entry:

    id            the provider's model id
    capabilities  "tools" and/or "structured", from the listing's ``supported_parameters``
                  (``tools``; ``structured_outputs`` or ``response_format``)
    zdr           True only where OpenRouter's zero-data-retention endpoint listing serves the
                  model; False where that listing was read and does not. Only True may take
                  customer-derived data (ARC-9); None (unknown) counts as False.
    expires       the listing's ``expiration_date``, or None
    measured      {task: "RM-nn YYYY-MM-DD"}: the pre-registered measurement that cleared the
                  model for a task (ARC-12). Empty until one does, so today no model diagnoses
                  and every call must name the measurement it belongs to.

Order is the candidate order when a caller pins nothing (C7b). The two zero-data-retention
models come first. Qwen3.8 reasons by default at its highest effort (the listing's
``default_effort: xhigh``), which spends a short answer's tokens on thinking, so it is second.

Chosen on 4 Oct 2026 from the seven free models that offered both tool calling and structured
output that day, leaving out the one with an expiration date (dots-3-note-preview) and the 2.6B
one (lfm-2.5). None is measured on any task yet. Re-read with ``scripts/llm_allowlist.py
propose`` and check with ``scripts/llm_allowlist.py refresh``; the listing this was read from
is committed as ``tests/fixtures/llm/openrouter-models-2026-10-04.json``.
"""

from __future__ import annotations

#: When the listing below was read.
READ_ON = "2026-10-04"

#: Where it was read from, keylessly.
SOURCE = ("OpenRouter GET /api/v1/models and GET /api/v1/endpoints/zdr, read without a key on "
          "2026-10-04; committed as tests/fixtures/llm/openrouter-models-2026-10-04.json and "
          "tests/fixtures/llm/openrouter-zdr-endpoints-2026-10-04.json")

ALLOWLIST: dict[str, list[dict]] = {
    "openrouter": [
        # Zero data retention: Novita.
        {"id": "apodex/apodex-1.1-mini:free", "capabilities": ["tools", "structured"],
         "zdr": True, "expires": None, "measured": {}},
        # Zero data retention: ModelRun. Reasons by default.
        {"id": "qwen/qwen3.8-27b:free", "capabilities": ["tools", "structured"],
         "zdr": True, "expires": None, "measured": {}},
        {"id": "nvidia/nemotron-3-super-120b-a12b:free", "capabilities": ["tools", "structured"],
         "zdr": False, "expires": None, "measured": {}},
        {"id": "google/gemma-4-31b-it:free", "capabilities": ["tools", "structured"],
         "zdr": False, "expires": None, "measured": {}},
        {"id": "google/gemma-4-26b-a4b-it:free", "capabilities": ["tools", "structured"],
         "zdr": False, "expires": None, "measured": {}},
    ],
    # Anthropic has no free tier: its ids are paid, its data class is unknown (ARC-9), and
    # paid ids are never on an allowlist. Claude is reached only by a pinned id with S's flag.
    "anthropic": [],
}
