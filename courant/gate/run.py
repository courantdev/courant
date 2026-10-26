"""The CI gate: verify the runs a repository names, write the summary and the packs, and decide
whether the check passes (TRD GATE-7, GATE-8, GATE-11).

This is the logic behind `courant gate`, which the composite Action (`.github/actions/
courant-gate/`) calls and nothing else: the Action has no logic of its own. The verdicts come
from `courant.verify` (ARC-13); what this module adds is which runs, where their output is, and
what a verdict does to a check (`courant.verify.outcome`).

Read-only on the repository except two places the caller names: the packs directory, and the
`pack-store` path, which must lie inside the repository root. No network is opened.

Exit codes of the gate:

- 0 the check passes;
- 1 at least one run was judged and fails the check (GATE-7: anything but `accept`, or
  anything but `reject` when `fail-on` says so; always a void or invalid specification);
- 4 something could not be judged: a configuration error, a glob that matches no run, a run
  with no output where `outputs` named one, or an unreadable input (GATE-10). It outranks 1.
"""

from __future__ import annotations

import glob
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from courant.gate.config import ConfigError, GateConfig, RunGroup
from courant.verify.outcome import check_fails, first_reason, spec_integrity

EXIT_PASS, EXIT_FAIL, EXIT_UNJUDGED = 0, 1, 4
_LOG_SUFFIXES = (".log", ".log.gz")


@dataclass
class Row:
    """One run's line in the summary."""

    run: str
    kind: str                      # "verdict" | "no-output" | "no-match" | "unreadable"
    verdict: str | None = None
    basis: str | None = None
    rule: str | None = None
    grade: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    failed: bool = False
    why: str = ""                  # why it fails the check, or why it passes though not an accept
    error: str | None = None
    pack: str | None = None


@dataclass
class GateResult:
    rows: list[Row] = field(default_factory=list)
    exit_code: int = EXIT_PASS
    fail_on: str = "not-accept"
    version: str = ""
    pack_store: str | None = None
    packs_dir: str | None = None
    cache_hits: int = 0


def _is_log(p: Path) -> bool:
    return p.is_file() and p.name.endswith(_LOG_SUFFIXES) and not p.name.startswith("log.")


def _run_name(p: Path) -> str:
    n = p.name
    for suffix in (".gz", ".log"):
        n = n.removesuffix(suffix)
    return n


def _sidecar(log: Path) -> Path | None:
    from courant.verify.ingest import sidecar_of

    side = sidecar_of(log)
    return side if side.is_file() else None


def _expand(root: Path, pattern: str) -> list[Path]:
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ConfigError(f"paths: {pattern!r} leaves the repository; use a path under {root.name}/")
    hits = sorted(glob.glob(pattern, root_dir=root, recursive=True))
    return [root / h for h in hits if not any(part.startswith(".") for part in Path(h).parts)]


def _output_for(root: Path, outputs: tuple[str, ...], name: str) -> tuple[Path, Path | None] | None:
    for out in outputs:
        base = root / out
        for cand in (base / name, base / f"{name}.log", base / f"{name}.log.gz"):
            if cand.is_dir() and any(c.name.startswith("log.") for c in cand.iterdir()):
                return cand, None
            if _is_log(cand):
                return cand, _sidecar(cand)
    return None


def _rel(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def _targets(root: Path, group: RunGroup) -> tuple[list[tuple[str, Path, Path | None]], list[Row]]:
    """(run label, target, sidecar) for each run the group names, and the rows for the ones
    that cannot be judged."""
    runs, problems = [], []
    for pattern in group.paths:
        matches = [m for m in _expand(root, pattern) if m.is_dir() or _is_log(m)]
        if not matches:
            problems.append(Row(run=pattern, kind="no-match",
                                error=f"the pattern {pattern!r} matches no case directory or log"))
        for m in matches:
            label = _rel(root, m)
            if group.outputs:
                found = _output_for(root, group.outputs, _run_name(m))
                if found is None:
                    where = ", ".join(group.outputs)
                    problems.append(Row(run=label, kind="no-output", error=(
                        f"no output for {_run_name(m)!r} under {where}: expected a directory "
                        f"with a log, {_run_name(m)}.log or {_run_name(m)}.log.gz")))
                    continue
                runs.append((label, found[0], found[1]))
            else:
                runs.append((label, m, _sidecar(m) if m.is_file() else None))
    return runs, problems


def _pack_name(label: str) -> str:
    return label.replace("/", "__").removesuffix(".gz").removesuffix(".log") or "run"


@dataclass
class Judged:
    """One named run and what the gate made of it: a verdict, or why there is none."""

    label: str
    target: Path
    sidecar: Path | None
    group: RunGroup
    verdict: object | None = None
    error: str | None = None


def check_config(root: Path, config: GateConfig) -> list[Path]:
    """The refusals that come before any run is read (a pinned version, no runs, a pack store
    outside the repository, a rules file that is not there). Returns the rules files."""
    from courant import __version__

    if config.courant_version and config.courant_version not in ("latest", __version__):
        raise ConfigError(f"{config.source} pins Courant {config.courant_version}; this is "
                          f"Courant {__version__}")
    if not config.runs:
        raise ConfigError("no runs named: give `paths` (courant.toml [[run]] or the Action's "
                          "`paths` input)")
    if config.pack_store:
        if config.pack_store.startswith("store:"):
            raise ConfigError("named pack stores are not available; give a path in the repository")
        if not (root / config.pack_store).resolve().is_relative_to(root):
            raise ConfigError(f"pack-store {config.pack_store!r} is outside the repository")
    rules = [root / r for r in config.rules]
    for r in rules:
        if not r.is_file():
            raise ConfigError(f"rules file not found: {r.relative_to(root)}")
    return rules


def judge_groups(root: Path, config: GateConfig, *, jobs: int = 1, cache=None
                 ) -> tuple[list[Judged], list[Row], int]:
    """Verify every run the groups name, in order, each run once. (the judged runs, the rows for
    what could not be judged, the cache hits.) Shared by `courant gate` and by
    `courant verify --batch` in a directory that has a courant.toml (GATE-8)."""
    from courant.verify.batch import verify_batch

    rules = check_config(root, config)
    judged: list[Judged] = []
    problems_all: list[Row] = []
    hits = 0
    seen: set[Path] = set()
    for group in config.runs:
        runs, problems = _targets(root, group)
        runs = [(label, t, c) for label, t, c in runs if t not in seen]
        seen.update(t for _, t, _ in runs)
        for p in problems:
            p.failed = True
            p.why = p.error or ""
        problems_all += problems
        if not runs:
            continue
        spec = root / group.spec if group.spec else None
        batch = verify_batch(root, jobs=jobs, cache=cache, spec=spec,
                             spec_locked_sha256=group.spec_sha256, rules=rules,
                             runs=[(t, c) for _, t, c in runs])
        hits += batch.hits
        for (label, target, sidecar), (_, verdict, err) in zip(runs, batch.results, strict=True):
            judged.append(Judged(label, target, sidecar, group, verdict, err))
    return judged, problems_all, hits


def run_gate(root: str | Path, config: GateConfig, *, packs_dir: str | Path | None = None,
             jobs: int = 1, cache=None) -> GateResult:
    """Verify every run `config` names under `root`."""
    from courant import __version__
    from courant.verify.pack import write_pack

    root = Path(root).resolve()
    result = GateResult(fail_on=config.fail_on, version=__version__,
                        pack_store=config.pack_store,
                        packs_dir=None if packs_dir is None else str(packs_dir))
    judged, problems, result.cache_hits = judge_groups(root, config, jobs=jobs, cache=cache)
    store = (root / config.pack_store).resolve() if config.pack_store else None
    result.rows += problems
    for j in judged:
        if j.error is not None:
            result.rows.append(Row(run=j.label, kind="unreadable", error=j.error, failed=True,
                                   why=j.error))
            continue
        verdict = j.verdict
        reason = first_reason(verdict)
        fails = check_fails(verdict, config.fail_on)
        row = Row(run=j.label, kind="verdict", verdict=verdict.verdict, basis=verdict.basis,
                  rule=verdict.rule, grade=verdict.grade,
                  reason_code=reason[0] if reason else None,
                  reason=reason[1] if reason else None, failed=fails)
        if spec_integrity(verdict):
            row.why = f"the specification's integrity: {verdict.rule}"
        elif fails:
            row.why = f"{verdict.verdict} ({verdict.rule})"
        elif verdict.verdict == "insufficient":
            row.why = "passes: fail-on is reject"
        if packs_dir is not None:
            name = _pack_name(j.label)
            write_pack(verdict, Path(packs_dir) / name, j.target.name,
                       j.sidecar.name if j.sidecar else None,
                       spec_path=root / j.group.spec if j.group.spec else None,
                       spec_locked_sha256=j.group.spec_sha256)
            row.pack = name
            if store is not None:
                shutil.copytree(Path(packs_dir) / name, store / name, dirs_exist_ok=True)
        result.rows.append(row)

    if any(r.kind != "verdict" for r in result.rows):
        result.exit_code = EXIT_UNJUDGED
    elif any(r.failed for r in result.rows):
        result.exit_code = EXIT_FAIL
    return result


def _cell(text: str | None, limit: int = 180) -> str:
    text = (text or "").replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_summary(result: GateResult) -> str:
    """The job summary (Markdown): one row per run, the verdict, its basis and its first
    reason; then why the check failed, and where the packs are."""
    rows = result.rows
    judged = [r for r in rows if r.kind == "verdict"]
    accepted = sum(1 for r in judged if r.verdict == "accept")
    out = [f"## Courant gate: {accepted} of {len(rows)} runs accepted", "",
           f"`fail-on: {result.fail_on}` · Courant {result.version} · {len(rows)} runs", "",
           "| Run | Verdict | Basis | First reason |", "|---|---|---|---|"]
    for r in rows:
        if r.kind != "verdict":
            out.append(f"| `{_cell(r.run)}` | **{r.kind.replace('-', ' ')}** | – | {_cell(r.error)} |")
            continue
        verdict = f"{r.verdict} ({r.rule})" + (f", grade {r.grade}" if r.grade else "")
        if r.verdict == "insufficient" and not r.failed:
            verdict += " – passes: fail-on is reject"
        first = f"`{r.reason_code}`: {_cell(r.reason)}" if r.reason_code else "none"
        out.append(f"| `{_cell(r.run)}` | {'**' + verdict + '**' if r.failed else verdict} "
                   f"| {r.basis} | {first} |")
    failing = [r for r in rows if r.failed]
    out.append("")
    if result.exit_code == EXIT_PASS:
        out.append(f"**The check passes:** {accepted} of {len(rows)} runs accepted"
                   + (f", {len(rows) - accepted} let through by `fail-on: reject`."
                      if accepted < len(rows) else "."))
    else:
        out.append(f"**The check fails:** {len(failing)} of {len(rows)} runs "
                   + ("could not be judged or fail it." if result.exit_code == EXIT_UNJUDGED
                      else "fail it."))
        for r in failing:
            cite = f", first reason `{r.reason_code}`" if r.reason_code else ""
            out.append(f"- `{r.run}`: {r.why or r.error}{cite}")
    if result.packs_dir:
        out.append("")
        if result.pack_store:
            out.append(f"Evidence packs are kept in `{result.pack_store}/` and attached to the "
                       "workflow run.")
        else:
            out.append("> **Warning:** the evidence packs exist only as workflow artefacts, which "
                       "expire. Set `pack-store` to a path in the repository to keep them.")
    return "\n".join(out) + "\n"
