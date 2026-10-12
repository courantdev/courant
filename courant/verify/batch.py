"""Verify every run under a directory (TRD VER-12, GATE-11), in sorted order: one at a time, or
`--jobs N` at once with identical results, and optionally through the verdict cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from courant.verify.ingest import InputError


def discover(root: str | Path) -> list[tuple[Path, Path | None]]:
    """(target, case_json) for each run: case directories with a log.*, and *.log / *.log.gz files."""
    root = Path(root)
    runs: list[tuple[Path, Path | None]] = []
    for p in sorted(root.rglob("*")):
        if any(part.startswith(("processor", ".")) for part in p.relative_to(root).parts):
            continue
        if p.is_dir() and any(c.name.startswith("log.") and c.is_file() for c in p.iterdir()):
            runs.append((p, None))
        elif p.is_file() and (p.name.endswith(".log") or p.name.endswith(".log.gz")) \
                and not p.name.startswith("log."):
            stem = p.name.removesuffix(".gz").removesuffix(".log")
            sidecar = p.with_name(stem + ".case.json")
            runs.append((p, sidecar if sidecar.is_file() else None))
    return runs


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
            keys[i] = vcache.run_key(ctx, name)
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
        verdict = load_verdict(text)
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
