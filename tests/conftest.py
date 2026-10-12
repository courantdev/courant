"""Shared fixtures.

The engine suite needs no services: every test runs against committed fixtures,
replayed solver logs, or temporary directories. The server-side fixtures
(in-memory databases, rate limiters) left with the server code; they are in
Courant-Product at d9ef7d0.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: spends one real request at a model provider; needs OPENROUTER_API_KEY in the "
        "environment and runs only when selected with -m live (T-LLM-FREE)")


# Never read the developer's `.env`. It lives in the repository root, so
# `load_dotenv()` finds it from anywhere in the tree -- which quietly made
# "no provider is configured" tests depend on whether the machine running them
# happened to have keys.
os.environ["COURANT_NO_DOTENV"] = "1"


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path_factory):
    """Restore ``os.environ`` after every test.

    `monkeypatch.setenv` is not enough on its own here. `courant.config`'s
    `.env` loader writes straight into `os.environ`, so a test that points it at
    a temp file leaves those variables set for the rest of the session.
    """
    snapshot = dict(os.environ)
    # The batch verdict cache (GATE-11) never touches the developer's ~/.cache in a test.
    os.environ["COURANT_CACHE_DIR"] = str(tmp_path_factory.mktemp("verdict-cache"))
    os.environ["COURANT_CACHE_KEY_FILE"] = str(tmp_path_factory.mktemp("cache-key") / "cache.key")
    # The suite behaves the same on a laptop and on a CI runner. `CI` turns the batch cache off
    # by default (GATE-11), so a runner's `CI=true` would change what a test sees; a test of that
    # behaviour sets `CI` itself (tests/unit/test_batch_cache.py).
    os.environ.pop("CI", None)
    yield
    os.environ.clear()
    os.environ.update(snapshot)
