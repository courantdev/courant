"""Verify every run under a directory (TRD VER-12, GATE-11), in sorted order: one at a time, or
`--jobs N` at once with identical results, and optionally through the verdict cache."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from courant.case import reads
from courant.case.mesh import is_read_by_its_case
from courant.verify.ingest import InputError, sidecar_of


def run_at(p: Path) -> tuple[Path, Path | None] | None:
    """(target, case_json) when `p` is a run, else None: the one rule for what a run is. A
    directory holding a `log.*` file is a case directory; a `*.log` or `*.log.gz` file that does
    not start with `log.` is a log, paired with its sidecar (`sidecar_of`) when it is there.

    A directory whose entries cannot be read (`reads.unreadable_dir`: mode 000, or listed and
    not searched, mode 0644) may hold a run, so it is one, and reading it says why it cannot be
    read: never a traceback, never a dropped row (GATE-14). A path whose own kind cannot be told
    (`reads.kind` is None: it lies in such a directory) is not a run; that directory is the row.
    Every test of a path goes through `reads.kind`, so this never raises."""
    p = Path(p)
    is_dir = reads.kind(p)
    if is_dir is None:
        return None
    if is_dir:
        if reads.unreadable_dir(p) is not None:
            return (p, None)
        try:
            if any(c.name.startswith("log.") and reads.kind(c, stat.S_ISREG)
                   for c in p.iterdir()):
                return (p, None)
        except OSError:
            return (p, None)
        return None
    if reads.kind(p, stat.S_ISREG) and (p.name.endswith(".log") or p.name.endswith(".log.gz")) \
            and not p.name.startswith("log."):
        sidecar = sidecar_of(p)
        return (p, sidecar if reads.is_file(sidecar) else None)
    return None


def discover(root: str | Path) -> list[tuple[Path, Path | None]]:
    """(target, case_json) for each run: case directories with a log.*, and *.log / *.log.gz files.

    A directory that cannot be read inside a case directory already found, at any depth below a
    directory the case's readers read (`mesh.is_read_by_its_case` of the first name under the
    nearest case: `constant/polyMesh`, `0/<region>`, `processors4`), is not a run of its own: the
    case's own row and verdict cover it (GATE-14, T-PRECHECK-2 C5). Any other unreadable
    directory (a sweep's `run3` under a top-level `log.blockMesh`), and every readable nested
    run, keeps its row.

    `root` itself may be a case (a readable directory with a `log.*` at its top). Then a
    directory that cannot be read inside it, and in no case found under it, is held back: if
    nothing else under `root` is a run, `root` is one case (`runs_under`) and those directories
    are its own, so none is a run and this returns no run; if something else is, `root` is a
    directory of runs and each is a row. The same rule as `runs_under`, so neither a case named
    as DIR nor `verify --batch` on it turns a directory inside the case into the only run."""
    root = Path(root)
    root_is_case = bool(reads.kind(root)) and reads.unreadable_dir(root) is None \
        and run_at(root) is not None
    runs: list[tuple[Path, Path | None]] = []
    cases: set[Path] = set()                 # sorted rglob yields a parent before its entries
    held: set[Path] = set()                  # unreadable, and inside `root` alone
    for p in sorted(root.rglob("*")):
        # processor<N>/ holds a decomposed run's pieces; only that name is skipped (SEC070-13).
        if any(part.startswith(".") or re.fullmatch(r"processor\d+", part)
               for part in p.relative_to(root).parts):
            continue
        run = run_at(p)
        if run is None:
            continue
        if reads.kind(p):                     # a case directory (run_at told its kind)
            if reads.unreadable_dir(p) is not None:
                case = next((d for d in p.parents if d in cases), None)
                if case is not None and is_read_by_its_case(p.relative_to(case).parts[0]):
                    continue
                if root_is_case:
                    held.add(p)
            cases.add(p)
        runs.append(run)
    if root_is_case and all(t in held for t, _ in runs):
        return []
    return runs


def runs_under(root: str | Path) -> list[tuple[Path, Path | None]]:
    """The runs `root` names (GATE-14): every run under it (`discover`, the one answer), and
    `root` alone only when `discover` finds none there and `root` is itself a run (`run_at`), so
    a directory of runs with a stray `log.*` at its top is still a directory of runs.

    Raises `InputError` when `root` is a directory whose entries cannot be read
    (`reads.unreadable_dir`), before anything is judged: a DIR that cannot be read is refused,
    never judged as one run (T-PRECHECK-2 C3)."""
    root = Path(root)
    if reads.kind(root):
        why = reads.unreadable_dir(root)
        if why is not None:
            raise InputError(f"{root.name} {why}")
    found = discover(root)
    if found:
        return found
    own = run_at(root)
    return [own] if own is not None else []


@dataclass
class BatchResult:
    """The outcome of one batch: a result per run, in sorted-path order, and what the cache did."""

    results: list = field(default_factory=list)      # (target, verdict or None, error or None)
    hits: int = 0
    misses: int = 0


def _verify_one(job: tuple):
    """One run, in a worker or in this process: (verdict JSON or None, error or None).

    A verdict crosses the process boundary as its canonical JSON, so a parallel run and a serial
    run hand back the same bytes by construction."""
    from courant.verify import verify_path

    target, sidecar, spec, locked = job
    try:
        v = verify_path(target, sidecar, spec=spec, spec_locked_sha256=locked)
    except InputError as exc:
        return None, str(exc)
    return v.to_json(), None


def verify_batch(root: str | Path, *, jobs: int = 1, cache=None, spec: str | Path | None = None,
                 spec_locked_sha256: str | None = None, rules=(), runs=None) -> BatchResult:
    """Verify every run under `root` (GATE-11).

    `jobs` runs in parallel, in separate processes; the results are identical to `jobs=1` and in
    the same order. `cache` is a `courant.verify.cache.VerdictCache` or None (no cache). A
    `spec` (and the hash it was locked at) applies to every run. `runs`, a list of
    `(target, sidecar)`, names the runs instead of `discover(root)` finding them (the GitHub
    Action names its own); each is keyed by its path under `root`."""
    from courant.verify import cache as vcache
    from courant.verify.verdict2 import load_verdict

    root = Path(root)
    runs = discover(root) if runs is None else [(Path(t), None if c is None else Path(c))
                                                for t, c in runs]
    out = BatchResult(results=[None] * len(runs))
    ctx = vcache.context(spec=spec, spec_locked_sha256=spec_locked_sha256, rules=rules) \
        if cache is not None else None
    keys: dict[int, str] = {}
    snaps: dict[int, list[str]] = {}
    pending: list[int] = []
    for i, (target, sidecar) in enumerate(runs):
        if cache is not None:
            name = target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target)
            keys[i] = vcache.run_key(cast(dict, ctx), name)     # ctx is set when cache is
            snaps[i] = vcache.snapshot(target, sidecar)        # before it is read, not after
            hit = cache.get(keys[i], target, sidecar, snaps[i])
            if hit is not None:
                out.results[i] = (target, hit, None)
                out.hits += 1
                continue
            out.misses += 1
        pending.append(i)

    work = [(str(runs[i][0]), None if runs[i][1] is None else str(runs[i][1]),
             None if spec is None else str(spec), spec_locked_sha256) for i in pending]
    jobs = max(1, min(int(jobs), len(work) or 1))
    if jobs == 1:
        done = [_verify_one(w) for w in work]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            done = list(pool.map(_verify_one, work))
    for i, (text, err) in zip(pending, done, strict=True):
        target, sidecar = runs[i]
        if err is not None:
            out.results[i] = (target, None, err)
            continue
        verdict = load_verdict(cast(str, text))              # no error: _verify_one gave JSON
        out.results[i] = (target, verdict, None)
        if cache is not None:
            cache.put(keys[i], target, sidecar, verdict, snaps[i])
    return out


def verify_all(root: str | Path, *, jobs: int = 1, cache=None, spec: str | Path | None = None,
               spec_locked_sha256: str | None = None):
    """(target, verdict or None, error or None) per run; verdicts are `courant.verdict/2`."""
    return verify_batch(root, jobs=jobs, cache=cache, spec=spec,
                        spec_locked_sha256=spec_locked_sha256).results


def summarise(results, schema: int | None = None) -> tuple[str, int]:
    """(summary line, exit code). The line is in the verdicts' own words: `accept`, `reject`
    and `insufficient` for `courant.verdict/2`, the version 1 words for `--schema 1`. With no
    verdict to go by (every run unreadable), `schema` decides; it defaults to 2."""
    from courant.verify.verdict2 import FROM_V1, VerdictV2, as_v1

    counts = {"TRUSTWORTHY": 0, "NOT_TRUSTWORTHY": 0, "NOT_ASSESSABLE": 0}
    errors = 0
    for _, v, err in results:
        if err:
            errors += 1
        else:
            counts[as_v1(v).verdict] += 1
            if schema is None:
                schema = 2 if isinstance(v, VerdictV2) else 1
    total = len(results)
    if schema == 1:
        words = {"TRUSTWORTHY": "trustworthy", "NOT_TRUSTWORTHY": "not trustworthy",
                 "NOT_ASSESSABLE": "not assessable"}
    else:
        words = FROM_V1
    line = (f"{total} runs: " + ", ".join(f"{counts[k]} of {total} {words[k]}" for k in counts)
            + (f", {errors} of {total} unreadable" if errors else ""))
    code = 2 if counts["NOT_TRUSTWORTHY"] else 3 if counts["NOT_ASSESSABLE"] else 4 if errors else 0
    return line, code
