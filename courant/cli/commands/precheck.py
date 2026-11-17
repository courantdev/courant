"""`courant precheck DIR`: the pre-check intake (TRD GATE-14).

It verifies every run under DIR (or DIR itself when it is one run) on this machine and writes
one archive to send to Courant: an evidence pack per run, `summary.json`, `summary.md` and
`MANIFEST.sha256`. No log, deck, mesh or field file enters it unless `--include-logs` adds each
run's solver log. It sends nothing: the archive stays here until you send it.

`courant precheck --check ARCHIVE` checks an archive (Courant's side).

Exit codes:
- 0 the archive was written (whatever the verdicts), or `--check` found it intact
- 4 no run was found, an option is refused, or the archive is not a readable zip
- 5 `--check` found a difference
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from courant.cli.guard import guarded

CONTACT = "hello@courant.dev"


def _refuse(message: str) -> typer.Exit:
    print(f"courant precheck: {message}", file=sys.stderr)
    return typer.Exit(4)


def _inside_a_run(out: Path, runs) -> Path | None:
    """The run `out` falls inside (or is), if any: the archive is never written into a run."""
    where = out.parent.resolve() / out.name
    for target, sidecar in runs:
        t = Path(target).resolve()
        if where == t or (t.is_dir() and where.is_relative_to(t)):
            return Path(target)
        if sidecar is not None and where == Path(sidecar).resolve():
            return Path(target)
    return None


def _refuse_inside(path: Path, runs, what: str) -> None:
    """Exit 4 when `path` (the archive, its partial file or the label map) would land in a run."""
    for p in (path, path.with_name(f".{path.name}.partial")):
        inside = _inside_a_run(p, runs)
        if inside is not None:
            shown = inside if inside.name not in ("", "..") else inside.resolve()
            raise _refuse(f"{what} is inside the run {shown}, and a run is never written to; "
                          "name the archive with -o outside every run")


@guarded("precheck")
def precheck(
    dir: str = typer.Argument(None, help="A directory of runs, or one run (a case directory or a solver log)."),
    out: str = typer.Option(None, "-o", "--output", help="Write the archive here (default: courant-precheck-<UTC date>-<digest>.zip in the working directory)."),
    include_logs: bool = typer.Option(False, "--include-logs", help="Also add each run's solver log to the archive (not with --redact-paths)."),
    redact_paths: bool = typer.Option(False, "--redact-paths", help="Replace each run's place and name with a label (run-01, ...); the label map is written beside the archive and stays on this machine."),
    jobs: int = typer.Option(1, "--jobs", min=1, help="Verify this many runs at once (default 1). The archive is identical to a serial run."),
    check: str = typer.Option(None, "--check", help="Check a pre-check archive: its manifest, its verdicts and its summary. Exit 5 on any difference."),
    debug: bool = typer.Option(False, "--debug", help="Print the traceback of an internal error instead of one line."),
) -> None:
    """Verify every run under DIR and write one archive to send to Courant. Sends nothing."""
    from courant.verify import precheck as pc

    if check:
        if dir or out or include_logs or redact_paths:
            raise _refuse("--check reads an archive; give it alone")
        from courant.verify.render import _one_line, _shown

        try:
            ok, message = pc.check_archive(check)
        except pc.ArchiveError as exc:
            raise _refuse(_one_line(_shown(str(exc)))) from exc
        # The message can quote the archive's member names and fields: the sender's text, shown
        # inert on one line (SEC070-6).
        print(("OK       " if ok else "MISMATCH ") + _one_line(_shown(message)))
        raise typer.Exit(0 if ok else 5)
    if include_logs and redact_paths:
        raise _refuse("--include-logs cannot go with --redact-paths: a solver log carries the "
                      "case's absolute path, and Courant does not edit a run's bytes")
    if not dir:
        raise _refuse("give a directory of runs, or one run (see --help)")
    import os

    root = Path(os.path.abspath(dir))                # one archive whatever DIR's spelling
    if not root.exists():
        raise _refuse(f"{dir} does not exist: nothing was judged")

    from courant.verify.batch import runs_under
    from courant.verify.ingest import InputError
    from courant.verify.render import _shown

    try:
        runs = runs_under(root)
    except InputError as exc:                        # a DIR that cannot be read (T-PRECHECK-2 C3)
        raise _refuse(f"{_shown(str(exc))}: nothing was judged, and no archive was written") \
            from exc
    if not runs:                                     # SEC070-12: no run is not a pass
        raise _refuse(f"no run found under {dir}: nothing was judged, and no archive was written")
    if out is not None:
        target = Path(out)
        if target.is_dir():
            raise _refuse(f"-o {out} is a directory; name the archive file")
        if not target.parent.is_dir():
            raise _refuse(f"-o {out}: its directory does not exist")
        _refuse_inside(target, runs, f"-o {out}")
    else:                                            # the working directory, checked the same way
        _refuse_inside(Path.cwd() / pc.default_name({pc.MANIFEST: b""}, pc.utc_today()), runs,
                       "the working directory")

    members, labels, rows = pc.build(root, jobs=jobs, include_logs=include_logs,
                                     redact=redact_paths, runs=runs)

    print(f"courant precheck: {len(rows)} runs found under {_shown(str(dir))}")
    import json

    for i, row in enumerate(rows, 1):
        place = _shown(row.place)
        if row.pack is None:
            print(f"  {i:>3}  {place}: could not be read: {_shown(row.unreadable or '')}")
            continue
        verdict = json.loads(members[f"{row.pack}/verdict.json"])
        print(f"  {i:>3}  {place}: {row.verdict}; {row.pack} names "
              + ", ".join(_shown(f["path"]) for f in verdict["run"]["inputs"]))
        if row.log is not None:
            print(f"       log added: {_shown(row.log)}")
        elif row.log_note is not None:
            print(f"       log left out: {_shown(row.log_note)}")
    if any(row.log is not None for row in rows):
        print("The solver logs are added as they are: an OpenFOAM log's banner names this "
              "machine's host name (Host :) and the case's absolute path (Case :), and both go "
              "with the archive.")
    print("The archive holds the packs (verdict.json, report.md, inputs.manifest), summary.json, "
          "summary.md and MANIFEST.sha256"
          + (", and the solver logs listed above" if include_logs else
             "; no log, deck, mesh or field file") + ".")
    print("This command sends nothing and opens no network connection: the archive stays on "
          "this machine until you send it.")

    if out is None:
        path = Path.cwd() / pc.default_name(members, pc.utc_today())
    else:
        path = Path(out)
    label_file = path.with_name(f"{path.stem}.labels.json")
    _refuse_inside(path, runs, f"the archive {path}")      # TST-4: a run is never written to
    if labels:
        _refuse_inside(label_file, runs, f"the label map {label_file}")
        if label_file.is_symlink() or (label_file.exists() and not label_file.is_file()):
            raise _refuse(f"the label map {label_file} is a link or not a file; it is never "
                          "written through one")
    data = pc.write_zip(members, path)
    import hashlib

    print(f"written: {path}")
    print(f"size: {len(data)} bytes")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    if labels:
        from courant.verify.pack import _write
        from courant.verify.schema import canonical_json

        # pack._write: O_NOFOLLOW, and a link is refused, never followed (SEC070-5)
        _write(label_file, canonical_json({"archive": path.name, "labels": labels}).encode())
        print(f"labels: {label_file} (which label is which run; it stays on this machine and is "
              "not in the archive)")
    print(f"Send the archive to {CONTACT}.")
