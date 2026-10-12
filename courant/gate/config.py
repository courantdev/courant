"""`courant.toml`: what a repository tells the gate (TRD GATE-8).

    fail-on = "not-accept"          # or "reject": an insufficient run then passes (GATE-7)
    pack-store = "evidence"         # a path inside the repository; the packs are copied there
    courant-version = "0.4.0"       # optional: the gate refuses to run under another version
    rules = []                      # reserved for an organisation's rules files (OVR-3, Phase 3)

    [[run]]
    paths = ["runs/*"]              # case globs, relative to the repository root
    outputs = "solver-output"       # optional: where each run's log and results are
    spec = "specs/task.json"        # optional: the task specification the run was asked to meet
    spec-sha256 = "9f2c..."         # required with `spec`: the hash it was locked at

An unknown key is an error that names it and says what is allowed: never a silent default (a
misspelt `fail-on` that quietly became `not-accept` would be a check that does not say what the
repository thinks it says). The same settings come in as the Action's inputs and the `courant
gate` flags, as one more `[[run]]` group and as overrides of the top-level keys.

`outputs` is where the verifier reads. A run named by `paths` is looked up there by its name
(`outputs/<name>/`, `outputs/<name>.log` or `outputs/<name>.log.gz`, a sidecar `<name>.case.json`
beside it), because repositories rarely commit solver output: it is an artefact of an earlier
job, an LFS path or a mounted volume. A run with no output there fails the check whatever
`fail-on` says.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from courant.verify.outcome import FAIL_ON

TOP_KEYS = ("fail-on", "pack-store", "courant-version", "rules", "run")
RUN_KEYS = ("paths", "outputs", "spec", "spec-sha256")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """`courant.toml` (or an Action input) cannot be used. The gate ends with exit 4."""


@dataclass(frozen=True)
class RunGroup:
    paths: tuple[str, ...]
    outputs: tuple[str, ...] = ()
    spec: str | None = None
    spec_sha256: str | None = None


@dataclass(frozen=True)
class GateConfig:
    runs: tuple[RunGroup, ...] = ()
    fail_on: str = "not-accept"
    pack_store: str | None = None
    courant_version: str | None = None
    rules: tuple[str, ...] = ()
    source: str = "(inputs)"


def _strs(value, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{where} must be a string or a list of strings")
    return tuple(value)


def _unknown(table: dict, allowed: tuple[str, ...], where: str, source: str) -> None:
    bad = sorted(set(table) - set(allowed))
    if bad:
        raise ConfigError(f"{source}: unknown key {', '.join(repr(k) for k in bad)} in {where} "
                          f"(allowed: {', '.join(allowed)})")


def _group(table: dict, where: str, source: str) -> RunGroup:
    _unknown(table, RUN_KEYS, where, source)
    if "paths" not in table:
        raise ConfigError(f"{source}: {where} has no 'paths'")
    spec = table.get("spec")
    sha = table.get("spec-sha256")
    if spec is not None and not isinstance(spec, str):
        raise ConfigError(f"{source}: {where}: spec must be a path")
    if sha is not None and not isinstance(sha, str):
        raise ConfigError(f"{source}: {where}: spec-sha256 must be a string")
    return make_group(_strs(table["paths"], f"{source}: {where}.paths"),
                      _strs(table["outputs"], f"{source}: {where}.outputs") if "outputs" in table else (),
                      spec, sha, where=f"{source}: {where}")


def make_group(paths, outputs=(), spec: str | None = None, spec_sha256: str | None = None,
               *, where: str = "inputs") -> RunGroup:
    if spec and not spec_sha256:
        raise ConfigError(f"{where}: 'spec' needs 'spec-sha256', the hash the specification was "
                          "locked at before the run; without it nothing stops the specification "
                          "from being edited until the run fits it")
    if spec_sha256 and not spec:
        raise ConfigError(f"{where}: 'spec-sha256' without 'spec'")
    if spec_sha256 and not _HEX64.match(spec_sha256.strip().lower()):
        raise ConfigError(f"{where}: spec-sha256 must be 64 hexadecimal digits")
    return RunGroup(tuple(paths), tuple(outputs), spec,
                    spec_sha256.strip().lower() if spec_sha256 else None)


def check_fail_on(value: str, source: str = "fail-on") -> str:
    if value not in FAIL_ON:
        raise ConfigError(f"{source}: fail-on must be one of {', '.join(FAIL_ON)}, not {value!r}")
    return value


def load_config(path: str | Path) -> GateConfig:
    """Read `courant.toml`. A missing file is an error; ask `exists` first when it is optional."""
    path = Path(path)
    source = path.name
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source} is not valid TOML: {exc}") from exc
    _unknown(table, TOP_KEYS, "the top level", source)
    runs = table.get("run", [])
    if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs):
        raise ConfigError(f"{source}: 'run' must be an array of tables, [[run]]")
    fail_on = table.get("fail-on", "not-accept")
    if not isinstance(fail_on, str):
        raise ConfigError(f"{source}: fail-on must be a string")
    check_fail_on(fail_on, f"{source}: fail-on")
    store = table.get("pack-store")
    if store is not None and not isinstance(store, str):
        raise ConfigError(f"{source}: pack-store must be a path")
    version = table.get("courant-version")
    if version is not None and not isinstance(version, str):
        raise ConfigError(f"{source}: courant-version must be a string")
    rules = _strs(table["rules"], f"{source}: rules") if "rules" in table else ()
    return GateConfig(
        runs=tuple(_group(r, f"[[run]] #{i + 1}", source) for i, r in enumerate(runs)),
        fail_on=fail_on, pack_store=store, courant_version=version, rules=rules, source=source)


def merged(base: GateConfig, *, group: RunGroup | None = None, fail_on: str | None = None,
           pack_store: str | None = None, courant_version: str | None = None) -> GateConfig:
    """`base` with the Action's inputs applied: a group added, the top-level keys overridden."""
    return GateConfig(
        runs=(*base.runs, *([group] if group else [])),
        fail_on=check_fail_on(fail_on) if fail_on else base.fail_on,
        pack_store=pack_store or base.pack_store,
        courant_version=courant_version or base.courant_version,
        rules=base.rules, source=base.source)
