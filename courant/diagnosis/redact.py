"""The trust boundary (Tech Bible §02).

Courant's central promise to a customer is one sentence:

    **Your mesh, geometry, and solution fields never leave your machine.**

An Evidence Bundle is the only thing that crosses, and this module is what makes
that promise true rather than merely intended. Every bundle passes through
``sanitize()`` on the way out of the agent, and the API re-checks on the way in —
defence in depth, because a patched or old client is exactly the case where a
leak would be silent.

Three classes of leak are handled:

  * **filesystem provenance** — absolute paths carry usernames, project
    codenames, and cluster layout. ``/scratch/acme-turbine-v3/…`` is a
    commercial disclosure even though it is "just a path".
  * **machine identity** — hostnames and usernames.
  * **bulk numeric payloads** — a long float array is how a mesh or a field
    would actually escape: not as a file, but smuggled inside a config value
    or an event payload.

What deliberately *does* cross: solver name, cell count, residual series
(downsampled), checker findings, mesh quality *metrics*, and config keys drawn
from a fixed allowlist.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "<redacted>"

# Fields each bundle section may carry. Allowlists, not pattern-matching:
# there is no regex for "is this string a username or a hostname", so trying to
# *remove* identity is unwinnable. Naming what may cross is winnable.
#
# ``name`` is here deliberately. A case name is the user's own label for their
# own run, shown only to their own org, and the console is unusable without it.
# It is not a leak — but a *path* containing that same word is.
ALLOWED_CASE_SUMMARY_FIELDS: frozenset[str] = frozenset({
    "name", "solver", "cell_count", "proc_count", "key_dicts_digest",
})

# Mesh *metrics* cross; mesh geometry does not. Every field here is a scalar.
ALLOWED_MESH_FIELDS: frozenset[str] = frozenset({
    "max_skewness", "max_non_ortho", "max_aspect_ratio", "min_volume",
    "negative_volumes", "n_cells", "n_faces", "n_points", "checked", "ok",
})

# A bundle field carrying more numbers than this is not a residual series —
# it is a mesh or a solution field trying to cross.
MAX_NUMERIC_SEQUENCE = 512
MAX_STRING_LENGTH = 2_000
MAX_LOG_WINDOW_CHARS = 4_000
#: Enough to cover the run-up to a failure without becoming a log shipper.
MAX_LOG_LINES = 200

# Absolute POSIX paths (two or more segments) and Windows paths.
_ABS_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[\w.\-@+]+[\\/]){1,}[\w.\-@+]*")
_HOME_PATH = re.compile(r"~[\w.\-]*[\\/][\w.\-\\/]*")
# user@host, common in MPI and scheduler chatter
_USER_AT_HOST = re.compile(r"\b[\w.\-]+@[\w.\-]+\.[A-Za-z]{2,}\b")
# A bare fully-qualified hostname: "gpu-node-47.acme-internal.net". This is a
# blocklist, and blocklists lose — which is why the *structural* defence is the
# solver-line allowlist in `_SOLVER_LINE_PATTERNS`, and this is only the second
# line for the one place that allowlist is deliberately opened (the FATAL error
# block, see `redact_log_window`).
#
# Shaped to miss OpenFOAM's own vocabulary, which is full of dotted tokens:
# two or more dots with a lowercase alphabetic final label, or one dot ending in
# a network TLD. `fvMatrix.C`, `log.simpleFoam`, `0.orig`, and `41.2` all survive;
# `gpu-node-47.acme-internal.net` and `login.hpc.local` do not.
_FQDN = re.compile(
    r"\b(?:[A-Za-z0-9\-]+\.){2,}[a-z]{2,24}\b"
    r"|\b[A-Za-z0-9\-]+\.(?:com|net|org|edu|gov|io|ai|dev|local|internal|lan|cloud)\b"
)
# Anything that looks like a credential, wherever it came from.
_SECRETISH = re.compile(
    r"\b(?:crt_[\w\-]{8,}|sk-[\w\-]{8,}|ghp_[\w]{8,}|AKIA[0-9A-Z]{12,})\b"
)


# Solver-output line shapes. A log window keeps only lines matching one of
# these; everything else — MPI banners, scheduler preamble, shell noise — is
# dropped. Those are precisely where bare hostnames and usernames live, and
# there is no regex for "is this token a username", so the only winnable rule
# is to name what may cross.
_SOLVER_LINE_PATTERNS = (
    re.compile(r"^\s*Time\s*=\s*[\d.eE+\-]+\s*$"),
    re.compile(r"smoothSolver|PCG|PBiCG|GAMG|DICPCG|DILUPBiCG", re.I),
    re.compile(r"Solving for \w+", re.I),
    re.compile(r"Courant Number", re.I),
    re.compile(r"continuity errors", re.I),
    re.compile(r"^\s*bounding \w+", re.I),
    re.compile(r"ExecutionTime|ClockTime"),
    re.compile(r"FOAM FATAL (ERROR|IO ERROR)", re.I),
    re.compile(r"^\s*(-->|#\d+\s)"),                    # FOAM error continuation
    re.compile(r"floating point exception|Segmentation fault", re.I),
    re.compile(r"^\s*(SIMPLE|PIMPLE|PISO) solution converged", re.I),
    re.compile(r"Final residual|Initial residual", re.I),
    re.compile(r"^\s*End\s*$"),
)

_FATAL_MARKER = re.compile(r"FOAM FATAL (ERROR|IO ERROR)", re.I)
# Bounded: the reason and its provenance, not an unbounded tail of whatever
# followed the crash.
FATAL_BLOCK_LINES = 6


def redact_log_lines(text: str, *, max_lines: int = 200) -> list[dict]:
    """Keep only recognisable solver output, **with its byte offsets**.

    The offsets are the reason this returns structures rather than a string.
    A deterministic Finding records the ``log_offset`` it was established from,
    and the console's promise is that clicking an evidence chip shows *the line
    that produced it*. Filtering the log to a flat string throws away the only
    thing that can connect the two, and "trust me, a line like this existed"
    is not evidence.

    What crosses is still only what the allowlist admits. The offset is a
    number; it discloses nothing on its own.
    """
    if not text:
        return []

    kept: list[dict] = []
    error_block_remaining = 0
    offset = 0

    for line in text.splitlines(keepends=True):
        start = offset
        offset += len(line.encode())
        stripped = line.rstrip("\n")

        def keep(content: str, at: int = start) -> None:
            kept.append({"offset": at, "text": redact_text(content)})

        # A FOAM FATAL ERROR is followed by its reason on the next few indented
        # lines ("Maximum number of iterations exceeded"). That is the single
        # most diagnostically valuable text in the whole log and matches no
        # solver pattern, so capture a bounded block after the marker and put
        # it through the ordinary path/identity redaction.
        if _FATAL_MARKER.search(stripped):
            keep(stripped)
            error_block_remaining = FATAL_BLOCK_LINES
            continue

        if error_block_remaining > 0:
            error_block_remaining -= 1
            if stripped.strip():
                keep(stripped)
            continue

        if any(p.search(stripped) for p in _SOLVER_LINE_PATTERNS):
            keep(stripped)

    # Tail, not head: the lines nearest the failure are the useful ones.
    return kept[-max_lines:]


def redact_log_window(text: str, *, max_chars: int = 4_000) -> str:
    """The same filtered log as flat text — what the model is shown.

    The parser has already turned the diagnostically meaningful lines into
    structured Events; what survives here is the human-readable context an
    engineer would want to see quoted in a report — chiefly the FOAM FATAL
    ERROR block, which no Event captures in full.
    """
    window = "\n".join(line["text"] for line in redact_log_lines(text))
    return window[-max_chars:] if len(window) > max_chars else window


def redact_text(value: str) -> str:
    """Strip filesystem, identity, and credential material from free text.

    Paths collapse to their basename rather than vanishing: ``log.simpleFoam``
    is diagnostically useful, ``/scratch/acme-turbine-v3/`` is a disclosure.
    """
    if not value:
        return value

    text = _SECRETISH.sub(REDACTED, value)
    text = _USER_AT_HOST.sub(REDACTED, text)
    text = _HOME_PATH.sub(lambda m: m.group(0).rsplit("/", 1)[-1] or REDACTED, text)
    text = _ABS_PATH.sub(lambda m: m.group(0).replace("\\", "/").rsplit("/", 1)[-1]
                         or REDACTED, text)
    # Last, so it sees the basenames the path rules just produced rather than
    # whole paths.
    text = _FQDN.sub(REDACTED, text)

    return text[:MAX_STRING_LENGTH]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sanitize_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively sanitise one bundle value."""
    if isinstance(value, str):
        return redact_text(value)

    if isinstance(value, dict):
        return {k: sanitize_value(v, key=k) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        # A long run of numbers is a field or a mesh, not a diagnostic.
        # Truncate rather than drop, so a residual series stays legible.
        if len(value) > MAX_NUMERIC_SEQUENCE and all(_is_number(v) for v in value):
            return list(value[:MAX_NUMERIC_SEQUENCE])
        return [sanitize_value(v) for v in value[:MAX_NUMERIC_SEQUENCE]]

    return value


def _allowlist(section: dict, allowed: frozenset[str]) -> dict:
    """Keep only named fields, sanitising the values that remain."""
    return {
        key: sanitize_value(value, key=key)
        for key, value in (section or {}).items()
        if key in allowed
    }


#: Fields of `CaseConfig` that may cross, and how each is treated.
#:
#: **Scalars and OpenFOAM-vocabulary maps.** Keys here are either fixed
#: (`delta_t`) or drawn from OpenFOAM's own closed grammar — `div(phi,U)` is a
#: term name, `U`/`p`/`".*"` are field names, `nu` is a property name. None of
#: them is customer-authored text.
CONFIG_FIELDS_VERBATIM: frozenset[str] = frozenset({
    "solver", "delta_t", "end_time", "adjust_timestep", "max_co",
    "write_control", "write_interval",
    "ddt_scheme", "div_schemes",
    "relaxation_factors", "relaxation_scopes", "linear_solvers",
    "n_non_orthogonal_correctors", "n_correctors", "n_outer_correctors",
    "residual_control",                 # field name or pattern -> tolerance, like relaxation
    "turbulence_type", "turbulence_model", "turbulence_switch", "transport",
})

#: `boundary` maps **patch name** -> {field: bcType}. Patch names *are*
#: customer-authored — `turbine_blade_root_v3` names a product — so they are
#: replaced with stable pseudonyms while the contents cross intact.
#:
#: Nothing is lost by that. The only questions the server asks of `boundary` are
#: shape questions: does any patch fix both `p` and `U`, is there an inlet with
#: no turbulence specification. `CONFIG_PREDICATES["overconstrained_pu"]` is
#: literally `any(... for f in c.boundary.values())` — it never reads a name.
#: The agent keeps the real names locally, so the finding the engineer reads
#: still says `inlet`.
CONFIG_FIELD_PSEUDONYMISED: str = "boundary"

#: `unreadable` maps a case-relative path (`system/controlDict` — fixed names,
#: not customer text) to a parser message. The message is the risk: a parser is
#: entitled to quote the line it choked on. Truncated hard and run through
#: `redact_text` like any other free text.
CONFIG_FIELD_REASONS: str = "unreadable"

#: `regions` are region names, customer-authored in a multi-region case.
CONFIG_FIELD_REGIONS: str = "regions"


def sanitize_config_view(config: dict) -> dict:
    """Enforce the trust boundary over the case configuration.

    **This is written in `CaseConfig`'s vocabulary, and that is the whole
    point.** It used to be written in OpenFOAM's camelCase *dictionary* keys —
    `maxCo`, `deltaT`, `divSchemes` — while `config_view` is
    `CaseConfig.model_dump()`, whose keys are the model's snake_case *field*
    names. The intersection of the two was exactly ``{"solver"}``, and only by
    coincidence: "solver" was on the list for `fvSolution`.

    So every Evidence Bundle that crossed the boundary arrived carrying no case
    configuration whatsoever. Not a leak — the opposite, and silent. All four KB
    config predicates were unreachable server-side, and the fix proposer read a
    solver only because `case_summary` carries it separately.

    Nothing caught it because the security suite's fixture writes `config_view`
    in camelCase — a shape the product never produces. The gate certified a
    fixture rather than the artefact. That is the same failure the real-solver
    certification exists to catch, arriving here in the layer we trust most,
    which is why `TestTheBoundaryIsTestedAgainstTheRealArtefact` now runs an
    actual `CaseConfig` through this function.

    `CaseConfig` is itself a closed, curated model — it cannot hold arbitrary
    OpenFOAM content — so it *is* the allowlist for the top level. What still
    needs naming is the content of its free-form maps, whose keys come from the
    customer's case. See the four constants above.
    """
    out: dict = {}
    for key, value in (config or {}).items():
        if key in CONFIG_FIELDS_VERBATIM:
            out[key] = sanitize_value(value, key=key)

        elif key == CONFIG_FIELD_PSEUDONYMISED and isinstance(value, dict):
            out[key] = {
                f"patch{i}": sanitize_value(fields, key=key)
                for i, (_name, fields) in enumerate(sorted(value.items()), 1)
            }

        elif key == CONFIG_FIELD_REASONS and isinstance(value, dict):
            out[key] = {
                path: redact_text(str(reason))[:MAX_REASON_LENGTH]
                for path, reason in value.items()
                if _is_case_relative(path)
            }

        elif key == CONFIG_FIELD_REGIONS and isinstance(value, list):
            # How many regions there are is the diagnostic ("this is a
            # multi-region case"); what they are called is not. Pseudonyms
            # rather than a bare count, so the value keeps `CaseConfig`'s shape
            # and the server can rebuild the model from what it received.
            out[key] = [f"region{i}" for i in range(1, len(value) + 1)]

    return out


#: A parser message may quote the line it choked on. Short enough to carry the
#: location and the exception type, short enough not to carry a dictionary.
MAX_REASON_LENGTH = 120


def _is_case_relative(path: Any) -> bool:
    """Whether this is one of the case-relative names the reader produces.

    `system/controlDict`, `constant/transportProperties`. An absolute path or a
    traversal is not something the reader emits, so its presence means something
    else wrote this field and it does not cross.
    """
    return (isinstance(path, str) and path
            and not path.startswith(("/", "\\", "~"))
            and ".." not in path and ":" not in path)


def sanitize_bundle(bundle: dict) -> dict:
    """Enforce the trust boundary over a serialised Evidence Bundle.

    Takes and returns a plain dict so it can run on the agent (before send) and
    in the API (on receive) without either side importing the other's models.
    """
    clean = dict(bundle)

    clean["config_view"] = sanitize_config_view(bundle.get("config_view") or {})

    # Section allowlists rather than pattern scrubbing: a bare hostname
    # ("gpu-node-47.acme-internal.net") or a bare username ("shivanshu") in an
    # unexpected field cannot be recognised by regex, so the only safe rule is
    # to name what may cross and drop everything else.
    clean["case_summary"] = _allowlist(
        bundle.get("case_summary") or {}, ALLOWED_CASE_SUMMARY_FIELDS
    )
    if bundle.get("mesh_quality") is not None:
        clean["mesh_quality"] = _allowlist(
            bundle["mesh_quality"], ALLOWED_MESH_FIELDS
        )

    # Events are structured, but payloads can carry a raw log line.
    window = clean.get("event_window") or []
    clean["event_window"] = [sanitize_value(e) for e in window[:200]]

    # Residual series: downsample rather than drop — the shape is the evidence.
    series = clean.get("residual_series") or {}
    clean["residual_series"] = {
        str(field): [float(x) for x in values[:MAX_NUMERIC_SEQUENCE]]
        for field, values in series.items()
        if isinstance(values, (list, tuple))
    }
    clean["residual_iterations"] = [
        float(x) for x in (clean.get("residual_iterations") or [])[:MAX_NUMERIC_SEQUENCE]
    ]

    clean["det_findings"] = [sanitize_value(f) for f in (clean.get("det_findings") or [])]
    clean["prior_attempts"] = [
        sanitize_value(a) for a in (clean.get("prior_attempts") or [])
    ]

    if isinstance(clean.get("log_window"), str):
        clean["log_window"] = redact_log_window(
            clean["log_window"], max_chars=MAX_LOG_WINDOW_CHARS
        )

    # Already-sanitised lines pass through the same allowlist again rather than
    # being trusted: this function runs on the agent *and* on the API, and the
    # second pass is the one that has to assume the first never happened.
    if isinstance(clean.get("log_lines"), list):
        clean["log_lines"] = [
            {"offset": int(line.get("offset", 0)),
             "text": redact_text(str(line.get("text", "")))}
            for line in clean["log_lines"][:MAX_LOG_LINES]
            if isinstance(line, dict)
        ]

    return clean


# ---------------------------------------------------------------------------
# Verification — used by the release-gate test and by the API on receive
# ---------------------------------------------------------------------------


def find_leaks(payload: Any, canaries: list[str]) -> list[str]:
    """Return the canary strings that appear anywhere in ``payload``.

    Case-insensitive: a Windows path or a shouted hostname is the same leak.
    """
    haystack = _flatten(payload).lower()
    return [c for c in canaries if c and c.lower() in haystack]


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def assert_no_absolute_paths(payload: Any) -> list[str]:
    """Absolute paths that survived sanitisation. Empty means clean."""
    text = _flatten(payload)
    return [m.group(0) for m in _ABS_PATH.finditer(text)
            if m.group(0).count("/") >= 2 or "\\" in m.group(0)]
