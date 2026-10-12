"""Local CLI state — ``~/.courant/``.

    config.toml        user settings, hand-editable
    credentials.json   the API token, mode 0600
    queue.db           events pending upload (see queue.py)

Two rules:

1. **Credentials never live in config.toml.** People paste config files into
   issues and commit them to dotfile repos. The token lives in its own
   0600 file that nothing invites you to share.
2. **Everything works with no config at all.** ``courant analyze`` must run on a
   cluster with no account, no network, and no writable home directory. Missing
   config is a normal state, not an error.
"""

from __future__ import annotations

import json
import os
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "https://api.courant.dev"

CONFIG_DIR_ENV = "COURANT_HOME"
API_URL_ENV = "COURANT_API_URL"
TOKEN_ENV = "COURANT_TOKEN"      # for CI, where a device flow has no browser


def config_dir() -> Path:
    """``~/.courant`` unless ``COURANT_HOME`` overrides it (tests, CI, sandboxes)."""
    return Path(os.environ.get(CONFIG_DIR_ENV, Path.home() / ".courant"))


def _ensure_dir() -> Path:
    d = config_dir()
    d.mkdir(parents=True, mode=0o700, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Credentials:
    token: str
    org_id: str
    org_name: str = ""
    plan: str = ""
    api_url: str = DEFAULT_API_URL

    @property
    def prefix(self) -> str:
        """Safe to print or log — identifies the token without exposing it."""
        return self.token[:12] if self.token else ""


def credentials_path() -> Path:
    return config_dir() / "credentials.json"


def load_credentials() -> Credentials | None:
    """Environment first, then disk. ``None`` means "not logged in", not an error.

    ``COURANT_TOKEN`` exists for CI and headless machines where no browser can
    complete a device flow.
    """
    env_token = os.environ.get(TOKEN_ENV)
    if env_token:
        return Credentials(
            token=env_token,
            org_id=os.environ.get("COURANT_ORG_ID", ""),
            api_url=os.environ.get(API_URL_ENV, DEFAULT_API_URL),
        )

    path = credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Credentials(
            token=data["token"],
            org_id=data.get("org_id", ""),
            org_name=data.get("org_name", ""),
            plan=data.get("plan", ""),
            api_url=data.get("api_url", DEFAULT_API_URL),
        )
    except (json.JSONDecodeError, KeyError, OSError):
        # A corrupt credentials file must not brick the CLI — `courant login`
        # has to remain reachable to fix it.
        return None


def save_credentials(creds: Credentials) -> Path:
    """Write 0600. The mode is set before the token is written, so the secret
    is never briefly world-readable on a shared login node."""
    _ensure_dir()
    path = credentials_path()

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(
            {
                "token": creds.token,
                "org_id": creds.org_id,
                "org_name": creds.org_name,
                "plan": creds.plan,
                "api_url": creds.api_url,
            },
            fh,
            indent=2,
        )
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def clear_credentials() -> bool:
    path = credentials_path()
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """``config.toml``, with defaults for everything."""

    api_url: str = DEFAULT_API_URL

    mode: str = "copilot"                 # observe | copilot | autopilot
    max_attempts: int = 3
    notify: list[str] = field(default_factory=lambda: ["desktop"])

    share_case_metadata: bool = True
    # Off by default and asked once, in plain language, after the third
    # successful diagnosis. The corpus is the moat, but taking it silently
    # would betray the trust the product is built on (Product Bible §06).
    contribute_corpus: bool = False

    watch_directories: list[str] = field(default_factory=list)
    poll_ms: int = 500

    fix_allow: list[str] = field(default_factory=lambda: ["numerics", "schemes"])
    fix_deny: list[str] = field(default_factory=lambda: ["mesh"])

    def effective_api_url(self, creds: Credentials | None = None) -> str:
        """Precedence: env > credentials > config > default.

        Credentials outrank config so that logging into staging does not
        silently keep pointing the agent at production.
        """
        if os.environ.get(API_URL_ENV):
            return os.environ[API_URL_ENV].rstrip("/")
        if creds and creds.api_url:
            return creds.api_url.rstrip("/")
        return self.api_url.rstrip("/")


def config_path() -> Path:
    return config_dir() / "config.toml"


def load_settings() -> Settings:
    path = config_path()
    if not path.exists():
        return Settings()
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        # A malformed config falls back to defaults rather than refusing to
        # run. An engineer at 3 a.m. should not be blocked by a stray bracket.
        return Settings()

    api = raw.get("api", {})
    defaults = raw.get("defaults", {})
    privacy = raw.get("privacy", {})
    watch = raw.get("watch", {})
    fix = raw.get("fix", {})

    return Settings(
        api_url=api.get("url", DEFAULT_API_URL),
        mode=defaults.get("mode", "copilot"),
        max_attempts=int(defaults.get("max_attempts", 3)),
        notify=list(defaults.get("notify", ["desktop"])),
        share_case_metadata=bool(privacy.get("share_case_metadata", True)),
        contribute_corpus=bool(privacy.get("contribute_corpus", False)),
        watch_directories=list(watch.get("directories", [])),
        poll_ms=int(watch.get("poll_ms", 500)),
        fix_allow=list(fix.get("allow", ["numerics", "schemes"])),
        fix_deny=list(fix.get("deny", ["mesh"])),
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value))      # correct TOML string escaping


def save_settings(settings: Settings) -> Path:
    """Write config.toml.

    Hand-rolled rather than pulling in a TOML writer: the schema is small and
    fixed, and every dependency is one more thing that has to build for four
    platforms in the packaged binary.
    """
    _ensure_dir()
    path = config_path()

    sections = {
        "api": {"url": settings.api_url},
        "defaults": {"mode": settings.mode, "max_attempts": settings.max_attempts,
                     "notify": settings.notify},
        "privacy": {"share_case_metadata": settings.share_case_metadata,
                    "contribute_corpus": settings.contribute_corpus},
        "watch": {"directories": settings.watch_directories, "poll_ms": settings.poll_ms},
        "fix": {"allow": settings.fix_allow, "deny": settings.fix_deny},
    }

    lines = ["# Courant CLI configuration — https://docs.courant.dev/cli/config",
             "# Credentials are NOT stored here; see credentials.json.", ""]
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(val)}" for key, val in values.items())
        lines.append("")

    path.write_text("\n".join(lines))
    return path
