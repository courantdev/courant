"""Unit tests for config + .env loading + capability detection."""

import os

import pytest

from courant.config import Settings, load_dotenv


@pytest.fixture
def dotenv_enabled(monkeypatch):
    """Opt back in to `.env` loading.

    conftest disables it suite-wide so no test reads the developer's real
    `.env`. These two tests are about the loader itself, so they turn it on and
    point it at a file they wrote.
    """
    monkeypatch.delenv("COURANT_NO_DOTENV", raising=False)


def test_env_file_loaded_without_override(tmp_path, monkeypatch, dotenv_enabled):
    monkeypatch.delenv("COURANT_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text('COURANT_MODEL="claude-sonnet-5"\n# comment\nFOO=bar\n')
    load_dotenv(str(env))
    assert os.environ["COURANT_MODEL"] == "claude-sonnet-5"
    assert os.environ["FOO"] == "bar"


def test_existing_env_var_not_overridden(tmp_path, monkeypatch, dotenv_enabled):
    monkeypatch.setenv("COURANT_MODEL", "already-set")
    env = tmp_path / ".env"
    env.write_text("COURANT_MODEL=from-file\n")
    load_dotenv(str(env))
    assert os.environ["COURANT_MODEL"] == "already-set"


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    """No provider configured, whatever the developer's shell exports, and the model door's
    state in a temporary directory rather than ~/.courant/llm."""
    for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "COURANT_ALLOW_PAID_LLM",
                 "COURANT_MODEL", "COURANT_OPENROUTER_MODEL", "COURANT_RUNNER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COURANT_LLM_STATE", str(tmp_path / "llm"))


def test_llm_ready_true_with_real_key(no_keys, monkeypatch):
    """`llm_ready` keeps meaning "a key is configured". Whether the LLM diagnoses is the model
    door's answer (T-LLM-FREE), and with only an Anthropic key it is no: Anthropic is paid and
    its data class unknown, so no Evidence Bundle may go there (ARC-6, ARC-9)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-realkey123")
    s = Settings(load_env=False)
    assert s.llm_ready is True
    assert s.capabilities()["llm"] is False


def test_placeholder_key_is_not_ready(no_keys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    s = Settings(load_env=False)
    assert s.llm_ready is False
    assert s.capabilities().get("llm_model") is None
    assert s.capabilities()["llm"] is False


def test_no_key_not_ready(no_keys):
    s = Settings(load_env=False)
    assert s.llm_ready is False
    assert s.providers == []
    assert not hasattr(s, "active_model"), "model choice belongs to the door"


def test_an_openrouter_key_alone_is_reported_truthfully(no_keys, monkeypatch):
    """Was `test_openrouter_alone_is_a_working_deployment`. The point stands: the console and
    `courant doctor` must report what actually answers. That is now the door's decision, and
    with no free model measured on diagnosis (ARC-12) it is the deterministic engine, so
    `llm` is False although the key is configured.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-real")
    s = Settings(load_env=False)
    assert s.llm_ready is True
    assert s.providers == ["openrouter"]
    assert s.capabilities()["llm"] is False
    assert not hasattr(s, "active_model"), "model choice belongs to the door"


def test_providers_follow_the_door_s_registry_order(no_keys, monkeypatch):
    """Was `test_anthropic_is_preferred_when_both_are_set`. Paid-first is what ARC-11 forbids:
    `providers` only reports the configured keys, in the door's registry order."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-real")
    s = Settings(load_env=False)
    assert s.providers == ["openrouter", "anthropic"]
    assert not hasattr(s, "active_model"), "model choice belongs to the door"


def test_the_loader_can_be_switched_off(tmp_path, monkeypatch):
    """`COURANT_NO_DOTENV=1` is what keeps the suite honest — assert it works."""
    monkeypatch.setenv("COURANT_NO_DOTENV", "1")
    monkeypatch.delenv("COURANT_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("COURANT_MODEL=from-file\n")

    assert load_dotenv(str(env)) is None
    assert "COURANT_MODEL" not in os.environ


def test_defaults(no_keys):
    s = Settings(load_env=False)
    for gone in ("model", "openrouter_model", "active_model"):
        assert not hasattr(s, gone), f"Settings.{gone}: model choice belongs to the door"
    assert s.runner_mode == "auto"
    assert "openfoam" in s.openfoam_image
