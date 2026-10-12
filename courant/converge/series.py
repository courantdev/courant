"""Reading what a run wrote (TRD CONV-3 to CONV-5). The only module in this package with I/O.

Monitors come from function-object output under `postProcessing/`. Residuals and the clock
come from the solver log's events. Everything here turns files into the plain sequences
`assess()` takes, and notes every file it read (`courant.case.reads`), so a pack can re-hash
them (REP-5, VER-14).

What it reads:
- whitespace tables, `*.dat`: forceCoeffs `coefficient.dat`, forces `force.dat` and
  `moment.dat`, surfaceFieldValue, volFieldValue;
- the probes format, one file per field, `# Probe i (x y z)`;
- vectors in parentheses, expanded to `_x`, `_y`, `_z`;
- the legacy forces header `forces(pressure viscous porous)`, expanded to `forces_pressure` …;
- restarts across start-time directories (M1-S-5).

What it does not read: `fieldMinMax`, whose rows interleave fields. Those files are listed as
ignored, never guessed at.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from courant.case.reads import note
from courant.converge.assess import (
    Clock,
    Convergence,
    Finding,
    MonitorSeries,
    Options,
    RunMeta,
    assess,
)

POST = "postProcessing"
MAX_FILES = 256
MAX_BYTES = 256 * 1024 * 1024
BAD_ROW_FRACTION = 0.01
RESTART_MAJORITY = 0.5

_IGNORED = re.compile(r"minmax", re.I)
_ENTRY = re.compile(r"\(([^()]*)\)|(\S+)")
_GROUP = re.compile(r"(\w+)\(([^()]*\s[^()]*)\)")
_COMPONENT = re.compile(r"^(.+)_([xyz])$")
_NPROCS = re.compile(r"^nProcs\s*:\s*(\d+)", re.M)
_EXEC = re.compile(r"^Exec\s*:\s*(\S+)", re.M)


def _is_number(name: str) -> bool:
    try:
        return math.isfinite(float(name))
    except ValueError:
        return False


# ------------------------------------------------------------------ tables
@dataclass
class Table:
    columns: list[str]
    vectors: dict[str, str]                 # expanded column -> its vector's name
    rows: list[tuple[float, list[float]]]
    bad_rows: int
    total_rows: int


def _entries(line: str) -> list[tuple[str, list[str]]]:
    """('s', [tok]) for a scalar, ('v', [x, y, z]) for a parenthesised vector."""
    line = line.replace("((", "(").replace("))", ")")
    out = []
    for m in _ENTRY.finditer(line):
        if m.group(1) is not None:
            out.append(("v", m.group(1).split()))
        else:
            out.append(("s", [m.group(2)]))
    return out


def _header_names(comments: list[str], n_entries: int) -> list[str] | None:
    """Column names for the entries after time, from the comment block, or None."""
    for line in reversed(comments[-2:]):
        text = _GROUP.sub(lambda m: " ".join(f"{m.group(1)}_{s}" for s in m.group(2).split()),
                          line.lstrip("#").strip())
        toks = text.split()
        if not toks:
            continue
        if toks[0].lower() == "time" and len(toks) - 1 == n_entries:
            return toks[1:]
        if toks[0].lower() == "probe" and len(toks) - 1 == n_entries:
            return [f"probe{t}" for t in toks[1:]]
    return None


def parse_table(text: str) -> Table | None:
    comments: list[str] = []
    shape: list[tuple[str, int]] | None = None
    names: list[str] = []
    vectors: dict[str, str] = {}
    rows: list[tuple[float, list[float]]] = []
    bad = total = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if shape is None:
                comments.append(line)
            continue
        ents = _entries(line)
        if shape is None:
            if not ents or ents[0][0] != "s":
                return None
            shape = [(k, len(v)) for k, v in ents]
            base = _header_names(comments, len(ents) - 1) or [f"c{i}" for i in range(1, len(ents))]
            for (kind, width), name in zip(shape[1:], base):
                if kind == "v" and width == 3:
                    for axis in "xyz":
                        names.append(f"{name}_{axis}")
                        vectors[f"{name}_{axis}"] = name
                elif kind == "v":
                    names.extend(f"{name}_{i}" for i in range(width))
                else:
                    names.append(name)
        total = total + 1
        if [(k, len(v)) for k, v in ents] != shape:
            bad = bad + 1
            continue
        try:
            flat = [float(tok) for _, toks in ents for tok in toks]
        except ValueError:
            bad = bad + 1
            continue
        rows.append((flat[0], flat[1:]))
    if shape is None:
        return None
    # v2406 writes force components as separate scalar columns (total_x total_y total_z), not
    # as parenthesised vectors. A full x, y, z triple under one base name is a vector.
    for name in names:
        m = _COMPONENT.match(name)
        if m and name not in vectors and all(f"{m.group(1)}_{a}" in names for a in "xyz"):
            vectors[name] = m.group(1)
    return Table(names, vectors, rows, bad, total)


def _merge_restarts(rows):
    """M1-S-5: the last strictly increasing run wins; earlier runs keep only what precedes it."""
    runs, cur = [], []
    for r in rows:
        if cur and r[0] <= cur[-1][0]:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    if not runs:
        return [], 0, None
    merged = list(runs[-1])
    discarded, span = 0, None
    for run in reversed(runs[:-1]):
        t0 = merged[0][0]
        keep = [r for r in run if r[0] < t0]
        lost = [r for r in run if r[0] >= t0]
        if lost:
            discarded = discarded + len(lost)
            lo, hi = lost[0][0], lost[-1][0]
            span = (min(lo, span[0]), max(hi, span[1])) if span else (lo, hi)
        merged = keep + merged
    return merged, discarded, span


# ------------------------------------------------------------------ discovery
@dataclass
class Discovery:
    monitors: list[MonitorSeries]
    ignored: list[str]
    searched: list[str]
    files: list[Path]


def _candidate_files(start_dir: Path):
    for f in sorted(start_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix == ".dat":
            yield f
        elif f.suffix == "":
            try:
                with f.open("r", errors="replace") as fh:
                    head = fh.readline()
            except OSError:
                continue
            note(f)                     # VER-14: its first line decided whether it is a monitor
            if head.lstrip().startswith("# Probe"):
                yield f


def discover(case_dir: Path) -> Discovery:
    post = case_dir / POST
    searched = [POST]
    if not post.is_dir():
        return Discovery([], [], searched, [])
    groups: dict[tuple[str, str], list[Path]] = {}
    ignored: list[str] = []
    for obj in sorted(p for p in post.iterdir() if p.is_dir()):
        if _IGNORED.search(obj.name):
            ignored.append(f"{POST}/{obj.name}")
            continue
        for start in sorted((d for d in obj.iterdir() if d.is_dir() and _is_number(d.name)),
                            key=lambda d: float(d.name)):
            for f in _candidate_files(start):
                if _IGNORED.search(f.name):
                    ignored.append(str(f.relative_to(case_dir)))
                    continue
                groups.setdefault((obj.name, f.name), []).append(f)
    monitors: list[MonitorSeries] = []
    files: list[Path] = []
    budget = MAX_BYTES
    for (obj, fname), paths in sorted(groups.items()):
        if len(files) + len(paths) > MAX_FILES:
            ignored.extend(str(p.relative_to(case_dir)) for p in paths)
            continue
        try:
            size = sum_ints(p.stat().st_size for p in paths)
        except OSError:                 # vanished since it was listed: not read, not listed
            ignored.extend(str(p.relative_to(case_dir)) for p in paths)
            continue
        if size > budget:
            ignored.extend(str(p.relative_to(case_dir)) for p in paths)
            continue
        budget = budget - size
        tables = []
        for p in paths:
            try:
                text = p.read_text(errors="replace")
            except OSError:             # unreadable: not read, so not listed (VER-14)
                ignored.append(str(p.relative_to(case_dir)))
                continue
            note(p)                     # VER-14: read, so listed, whether or not it parses
            t = parse_table(text)
            if t is not None:
                tables.append((p, t))
        if not tables:
            ignored.extend(str(p.relative_to(case_dir)) for p in paths)
            continue
        files.extend(p for p, _ in tables)
        columns = tables[-1][1].columns
        usable = [(p, t) for p, t in tables if t.columns == columns]
        rows = [r for _, t in usable for r in t.rows]
        bad = sum_ints(t.bad_rows for _, t in usable)
        total = sum_ints(t.total_rows for _, t in usable)
        merged, discarded, span = _merge_restarts(rows)
        stem = fname[:-4] if fname.endswith(".dat") else fname
        source = ";".join(str(p.relative_to(case_dir)) for p, _ in usable)
        vectors = tables[-1][1].vectors
        for j, col in enumerate(columns):
            mid = f"{obj}/{stem}:{col}"
            findings: list[Finding] = []
            reason = None
            if discarded:
                findings.append(Finding("CONV.RESTART_OVERLAP", "WARN", mid, "M1-S-5",
                                        {"fraction": RESTART_MAJORITY},
                                        {"monitor": mid, "discarded_samples": discarded,
                                         "discarded_time_range": list(span)}))
                if discarded > RESTART_MAJORITY * (len(merged) + discarded):
                    reason = "RESTART_OVERLAP_MAJORITY"
            if total and bad > BAD_ROW_FRACTION * total:
                findings.append(Finding("CONV.MONITOR_UNREADABLE", "WARN", mid, "table reader",
                                        {"fraction": BAD_ROW_FRACTION},
                                        {"monitor": mid, "bad_rows": bad, "rows": total}))
                reason = "UNREADABLE_ROWS"
            monitors.append(MonitorSeries(
                id=mid, column=col, source=source, times=[r[0] for r in merged],
                values=[r[1][j] for r in merged],
                vector=f"{obj}/{stem}:{vectors[col]}" if col in vectors else None,
                reader_findings=tuple(findings), not_assessable=reason))
    return Discovery(monitors, sorted(set(ignored)), searched, files)


def sum_ints(values) -> int:
    total = 0
    for v in values:
        total = total + int(v)
    return total


# ------------------------------------------------------------------ the log
def from_events(events) -> tuple[dict[str, list[float]], Clock, int]:
    from courant.events.core import EventKind

    residuals: dict[str, list[float]] = {}
    times, clock_s, exec_s = [], [], []
    for e in events:
        if e.kind is not EventKind.RESIDUAL:
            continue
        p = e.payload
        for name, value in p.initial_residuals.items():
            residuals.setdefault(name, []).append(value)
        times.append(p.time if p.time is not None else float(len(times) + 1))
        clock_s.append(p.clock_time)
        exec_s.append(p.exec_time)
    return residuals, Clock(times, clock_s, exec_s), len(times)


def ranks_from_log(log_text: str) -> tuple[int, str]:
    m = _NPROCS.search(log_text[:20000])
    if m:
        return int(m.group(1)), "log_nprocs"
    return 1, "serial"


def solver_from_log(log_text: str) -> str | None:
    m = _EXEC.search(log_text[:20000])
    return m.group(1) if m else None


def regime_of(config, solver: str | None, clock: Clock) -> tuple[str | None, str, dict | None]:
    """config first, then the solver table, then the time column (M1-R-1 to R-4)."""
    from courant.case import solvers

    by_config = None
    if config is not None and getattr(config, "ddt_scheme", None):
        by_config = "steady" if config.is_steady() else "transient"
    by_solver = None
    if solver and solvers.is_known(solver):
        by_solver = "steady" if solvers.is_steady(solver) else "transient"
    conflict = None
    if by_config and by_solver and by_config != by_solver:
        conflict = {"config": by_config, "solver": by_solver}
    if by_config:
        return by_config, "config", conflict
    if by_solver:
        return by_solver, "solver", None
    t = list(clock.times)
    if len(t) >= 2:
        integral = all(float(x).is_integer() for x in t) and all(t[i + 1] - t[i] == 1.0
                                                                 for i in range(len(t) - 1))
        return ("steady" if integral else "transient"), "inferred", None
    return None, "unknown", None


def ended_by(events) -> tuple[str | None, int | None]:
    from courant.events.core import EventKind

    steps = len([e for e in events if e.kind is EventKind.RESIDUAL])
    for e in events:
        if e.kind is EventKind.CRASH:
            return ("signal" if e.payload.signal else "crash"), steps
    return None, steps


# ------------------------------------------------------------------ which monitors are the answer
#: CONV-4a. By default the answer is the totals a function object writes. A front/rear or
#: pressure/viscous split is a diagnostic of a total, and judging it as well lets the least
#: relevant column decide the run. Splits are excluded only where their total is present in
#: the same file. `--monitor` patterns replace the default entirely.
_FORCE_SPLIT = re.compile(r"^(pressure|viscous|porous)_[xyz]$")
_COEFF_SPLIT = re.compile(r"\((f|r)\)$")


def select(monitors, patterns) -> tuple[list, list, str]:
    """(selected, not selected, basis)."""
    import fnmatch

    if patterns:
        chosen = [m for m in monitors
                  if any(fnmatch.fnmatchcase(m.id, p) or fnmatch.fnmatchcase(m.column, p)
                         for p in patterns)]
        return chosen, [m.id for m in monitors if m not in chosen], "user_patterns"
    by_file: dict[str, set[str]] = {}
    for m in monitors:
        by_file.setdefault(m.id.rsplit(":", 1)[0], set()).add(m.column)
    chosen, dropped = [], []
    for m in monitors:
        cols = by_file[m.id.rsplit(":", 1)[0]]
        split = ((_FORCE_SPLIT.match(m.column) and "total_x" in cols) or
                 (_COEFF_SPLIT.search(m.column) and m.column.split("(")[0] in cols))
        (dropped.append(m.id) if split else chosen.append(m))
    return chosen, dropped, "default_totals"


# ------------------------------------------------------------------ the entry point
def assess_case(case_dir: str | Path | None, *, log_text: str, events, config=None,
                solver: str | None = None, options: Options = Options(),
                read: tuple = ()) -> Convergence:
    """Read the case's monitors and the log's residuals and clock, then `assess()`.

    `Convergence.inputs` is the verdict's manifest function applied to this run (VER-14):
    `read`, the manifest of what the caller already read for it (the ingest's log, deck and
    checkMesh log, where the regime and the clock came from), merged with every case file this
    function read, parsed or not, as its readers noted it.
    """
    from courant.case.reads import recording
    from courant.verify.schema import manifest, merge

    residuals, clock, steps = from_events(events)
    solver = solver or solver_from_log(log_text)
    regime, basis, conflict = regime_of(config, solver, clock)
    ranks, ranks_basis = ranks_from_log(log_text)
    how, last = ended_by(events)
    if case_dir:
        with recording(case_dir) as reads:
            found = discover(Path(case_dir))
        inputs = merge(read, manifest(case_dir, reads.paths()))
    else:
        found, inputs = Discovery([], [], [], []), tuple(read)
    chosen, unselected, basis_sel = select(found.monitors, options.monitors)
    meta = RunMeta(regime=regime, regime_basis=basis, regime_conflict=conflict, ranks=ranks,
                   ranks_basis=ranks_basis, ended_by=how, last_complete_step=last,
                   paths_searched=tuple(found.searched), monitors_ignored=tuple(found.ignored),
                   monitors_unselected=tuple(unselected), selection=basis_sel)
    return assess(chosen, residuals, clock, meta, options, inputs)
