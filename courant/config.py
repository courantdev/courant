"""Central configuration + capability detection (env-driven, .env-aware).

One place decides what is available. ``real_key`` is the one definition of "a key
is configured", and the model door (``courant/llm/``) reuses it; which model
answers, and whether the LLM diagnoser runs at all, is the door's decision, not
this module's. Start the Docker daemon and the real-OpenFOAM runner auto-engages.
"""

from __future__ import annotations

import os
from functools import lru_cache

# The solver image, pinned by digest (TRD OPS-2). packaging/images.lock is the record: it names
# the tag this was read from (opencfd/openfoam-default:2406), the platform manifests and where
# the digest came from, and tests/unit/test_ops.py fails when the two disagree. The tag moves;
# the digest does not. COURANT_OPENFOAM_IMAGE overrides it for a machine that mirrors the image.
OPENFOAM_IMAGE = "opencfd/openfoam-default@sha256:dd5aa20630a55722663bf83ba0cb74870cba130081303e32e3865007fa2aa35a"

# Values that look like a key but are not one. A copied-but-unedited .env is
# the common case, and treating its placeholder as real makes every request 401
# with an error nobody reads as "you forgot to fill this in".
_PLACEHOLDERS = {"sk-ant-...", "sk-or-...", "sk-...", "your-key-here", ""}


def real_key(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value if value not in _PLACEHOLDERS else None


def _parse_env_file(path: str) -> None:
    """Set KEY=VALUE lines into the environment without overriding existing vars."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_dotenv(explicit: str | None = None) -> str | None:
    """Load the first ``.env`` found (explicit path, repo root, or cwd).

    ``COURANT_NO_DOTENV=1`` disables it entirely. The test suite sets that,
    because `.env` is developer-specific state sitting in the repository root:
    a test asserting "no provider is configured" passed on a machine with no
    keys and failed the moment a real key was added to `.env` — which is the
    worst possible time to discover a test was never checking what it claimed.
    """
    if os.environ.get("COURANT_NO_DOTENV") == "1":
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    for path in [explicit, os.path.join(here, "..", ".env"),
                 os.path.join(os.getcwd(), ".env")]:
        if path and os.path.exists(path):
            _parse_env_file(path)
            return path
    return None


class Settings:
    def __init__(self, load_env: bool = True):
        if load_env:
            load_dotenv()
        # Keys only. Model choice belongs to the model door (courant/llm/): the
        # allowlist, the data class and the measurements decide which model may
        # answer, so nothing here names one.
        self.anthropic_api_key = real_key("ANTHROPIC_API_KEY")
        self.openrouter_api_key = real_key("OPENROUTER_API_KEY")
        self.llm_max_tokens = int(os.environ.get("COURANT_LLM_MAX_TOKENS", "4096"))
        self.openfoam_image = os.environ.get("COURANT_OPENFOAM_IMAGE", OPENFOAM_IMAGE)
        # runner: auto (docker if available else offline) | docker | replay
        self.runner_mode = os.environ.get("COURANT_RUNNER", "auto")

    @property
    def llm_ready(self) -> bool:
        """Whether a model provider's key is configured. Not whether the LLM
        diagnoser runs: that is ``capabilities()["llm"]``, the model door's answer."""
        return bool(self.anthropic_api_key or self.openrouter_api_key)

    @property
    def providers(self) -> list[str]:
        """Providers with a key configured, in the model door's registry order.
        A report only: it is not a fallback order (ARC-11)."""
        from courant.llm import REGISTRY

        keys = {"openrouter": self.openrouter_api_key, "anthropic": self.anthropic_api_key}
        return [name for name in REGISTRY if keys.get(name)]

    def docker_ready(self) -> bool:
        return docker_available()

    @property
    def docker_detail(self) -> str:
        """Why Docker is unusable, or "" when it is fine."""
        return docker_status()[1]

    def capabilities(self) -> dict:
        # Whether the LLM diagnoser runs, and on which model, is the model door's
        # answer, so the console and `courant doctor` report what actually answers.
        from courant.diagnosis.llm import DIAGNOSIS_CALL
        from courant.llm import default_door

        candidates = default_door().eligible(**DIAGNOSIS_CALL)
        return {
            "llm": bool(candidates),
            "llm_model": candidates[0].id if candidates else None,
            "llm_providers": self.providers,
            "docker": self.docker_ready(),
            "docker_detail": self.docker_detail,
            "openfoam_image": self.openfoam_image,
        }


@lru_cache(maxsize=1)
def docker_status() -> tuple[bool, str]:
    """``(available, why_not)``. Cached per process.

    Two failures wear the same face and need opposite fixes, so they are
    reported separately:

      * the ``docker`` **Python SDK** is not installed — `pip install
        'courant-dev[sandbox]'`. Nothing is wrong with the machine.
      * the **daemon** is unreachable — start Docker Desktop, or check
        DOCKER_HOST.

    Collapsing both into "docker: false" sent me looking for a broken daemon on
    a machine where `docker run` worked perfectly and only the Python package
    was missing. A tool whose whole purpose is diagnosis should not misdiagnose
    its own dependencies.
    """
    try:
        import docker  # type: ignore
    except ImportError:
        return False, "the docker Python SDK is not installed (pip install " \
                      "'courant-dev[sandbox]')"

    try:
        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001 — any failure means unusable
        return False, f"the Docker daemon is not reachable ({type(exc).__name__})"

    return True, ""


def docker_available() -> bool:
    """True if a Docker daemon is reachable."""
    return docker_status()[0]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
