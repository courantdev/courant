"""`courant verify`: a verdict on a completed run, with its evidence (TRD VER-10).

It writes `courant.verdict/2` (accept, reject or insufficient, with its basis; TRD GATE-1..3),
and `courant.verdict/1` behind `--schema 1` for one minor release (ARC-16).

Exit codes:
- 0 accept (TRUSTWORTHY in version 1)
- 2 reject (NOT_TRUSTWORTHY)
- 3 insufficient (NOT_ASSESSABLE)
- 4 the inputs could not be read
- 5 --check-pack or --check-record found a difference

stdout carries the report. stderr carries diagnostics. Read-only: nothing in
the case is written, no solver is run, no socket is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from courant.cli.guard import guarded

#: Either major's answer to its exit code (GATE-1: the same three codes).
EXIT = {"TRUSTWORTHY": 0, "NOT_TRUSTWORTHY": 2, "NOT_ASSESSABLE": 3,
        "accept": 0, "reject": 2, "insufficient": 3}


def certification_summary() -> str:
    """The rates published for this version, read only from the packaged results (GATE-12,
    CERT-3, PKG-5). One implementation: `courant.verify.about`."""
    from courant.verify import about

    return about.render_text()


@guarded("verify")
def verify(
    path: str = typer.Argument(None, help="A case directory, or a solver log (.log or .log.gz)."),
    case_json: str = typer.Option(None, "--case-json", help="Captured configuration sidecar (*.case.json)."),
    as_json: bool = typer.Option(False, "--json", help="Print the canonical verdict JSON."),
    report: str = typer.Option(None, "--report", help="'md' prints a Markdown report; 'html' prints a self-contained HTML report."),
    html: str = typer.Option(None, "--html", help="Write the self-contained HTML report to this file."),
    junit: str = typer.Option(None, "--junit", help="With --batch: write JUnit XML here, one test per run."),
    about: bool = typer.Option(False, "--about", help="Print this version's published error rates (false accepts, false alarms, abstentions, each with its denominator and confidence interval); with --json or --report md, as data or Markdown."),
    pack: str = typer.Option(None, "--pack", help="Write an evidence pack (verdict.json, report.md, inputs.manifest) here."),
    check_pack: str = typer.Option(None, "--check-pack", help="Re-verify a pack from its inputs; exit 5 if anything differs."),
    record: str = typer.Option(None, "--record", help="Write a verified record (courant.record/1: the specification, the verdict, provenance and the full manifest) here."),
    check_record: str = typer.Option(None, "--check-record", help="Re-derive a record from its inputs; exit 5 if anything differs."),
    producer: str = typer.Option(None, "--producer", help="With --record: who or what produced the run (an agent and its version)."),
    requester: str = typer.Option(None, "--requester", help="With --record: who asked for the run."),
    root: str = typer.Option(None, "--root", help="Where a pack's or a record's inputs live (for --check-pack and --check-record; default: cwd)."),
    batch: str = typer.Option(None, "--batch", help="Verify every run under this directory."),
    spec: str = typer.Option(None, "--spec", help="The task specification (courant.spec/1) the run was asked to satisfy; every value it names is checked against the case."),
    spec_sha256: str = typer.Option(None, "--spec-sha256", help="The hash the specification was locked at before the run (courant spec hash). A specification that no longer matches it makes the verdict insufficient."),
    jobs: int = typer.Option(1, "--jobs", min=1, help="With --batch: verify this many runs at once (default 1). The results are identical to a serial run."),
    no_cache: bool = typer.Option(False, "--no-cache", help="With --batch: do not read or write the verdict cache ($COURANT_CACHE_DIR, else ~/.cache/courant/verdicts)."),
    cache_on: bool = typer.Option(False, "--cache", help="With --batch: use the verdict cache even where CI is set (it is off there by default)."),
    checks: str = typer.Option("default", "--checks", help="'next' also runs the checks that ship behind this flag for one minor release before they are on by default (GATE-13); the verdict records it. One run at a time: not with --batch, --pack or --record."),
    schema: int = typer.Option(2, "--schema", help="The verdict schema to write: 2 (courant.verdict/2, the default) or 1 (courant.verdict/1, the previous major, kept for one minor release)."),
    debug: bool = typer.Option(False, "--debug", help="Print the traceback of an internal error instead of one line."),
) -> None:
    """Verify a completed simulation: accept, reject or insufficient, with its basis."""
    from courant.verify import InputError, render_json, render_markdown, render_text
    from courant.verify.verdict2 import SCHEMAS, in_schema

    if schema not in SCHEMAS:
        print(f"courant verify: --schema must be 1 or 2, not {schema}", file=sys.stderr)
        raise typer.Exit(4)

    from courant.verify.checks import MODES

    if checks not in MODES:
        print(f"courant verify: --checks must be one of {', '.join(MODES)}, not {checks}",
              file=sys.stderr)
        raise typer.Exit(4)
    if checks == "next" and (batch or pack or record):
        print("courant verify: --checks next verifies one run; packs, records and batches use "
              "the default checks", file=sys.stderr)
        raise typer.Exit(4)

    if about:
        from courant.verify import about as _about

        if as_json:
            from courant.verify.schema import canonical_json

            sys.stdout.write(canonical_json(_about.error_rates()))
        elif report == "md":
            sys.stdout.write(_about.render_markdown())
        else:
            sys.stdout.write(certification_summary())
        raise typer.Exit(0)
    if check_pack:
        from courant.verify.pack import check_pack as _check

        ok, message = _check(check_pack, root or ".")
        print(("OK       " if ok else "MISMATCH ") + message)
        raise typer.Exit(0 if ok else 5)
    if check_record:
        from courant.record import check_record as _check_record

        ok, message = _check_record(check_record, root or ".")
        print(("OK       " if ok else "MISMATCH ") + message)
        raise typer.Exit(0 if ok else 5)
    if (producer or requester) and not record:
        print("courant verify: --producer and --requester need --record", file=sys.stderr)
        raise typer.Exit(4)
    if record and batch:
        print("courant verify: --record writes one run's record; it cannot go with --batch",
              file=sys.stderr)
        raise typer.Exit(4)
    if batch:
        from courant.verify.batch import summarise, verify_batch
        from courant.verify.cache import VerdictCache

        if cache_on and no_cache:
            print("courant verify: --cache and --no-cache contradict each other", file=sys.stderr)
            raise typer.Exit(4)
        import os

        no_cache = no_cache or (not cache_on and bool(os.environ.get("CI")))
        if spec_sha256 and not spec:
            print("courant verify: --spec-sha256 needs --spec", file=sys.stderr)
            raise typer.Exit(4)
        settings = Path(batch) / "courant.toml"
        if settings.is_file():
            # GATE-8: a repository's courant.toml names the runs, their outputs and their
            # specifications; an unknown key in it is an error, never a silent default.
            from courant.gate.config import ConfigError, load_config
            from courant.gate.run import judge_groups

            if spec or spec_sha256:
                print("courant verify: --spec and --spec-sha256 do not combine with the "
                      "courant.toml in this directory", file=sys.stderr)
                raise typer.Exit(4)
            try:
                judged, problems, hits = judge_groups(
                    Path(batch).resolve(), load_config(settings), jobs=jobs,
                    cache=None if no_cache else VerdictCache())
            except ConfigError as exc:
                print(f"courant verify: {exc}", file=sys.stderr)
                raise typer.Exit(4) from exc
            unjudged = [(Path(batch) / p.run, None, p.error) for p in problems]
            results = [(j.target, None if j.verdict is None else in_schema(j.verdict, schema),
                        j.error) for j in judged] + unjudged
            if not no_cache:
                print(f"cache: {hits} of {len(judged)} runs served from the cache", file=sys.stderr)
        else:
            done = verify_batch(batch, jobs=jobs, cache=None if no_cache else VerdictCache(),
                                spec=spec, spec_locked_sha256=spec_sha256)
            results = [(t, None if v is None else in_schema(v, schema), e)
                       for t, v, e in done.results]
            if not no_cache:
                print(f"cache: {done.hits} of {done.hits + done.misses} runs served from the "
                      "cache", file=sys.stderr)
        base = Path(batch).resolve() if settings.is_file() else Path(batch)
        for target, v, err in results:
            rel = target.relative_to(base) if target.is_relative_to(base) else target
            print(f"{'ERROR' if err else v.verdict:16s} {'' if err else v.rule:8s} "
                  f"{'-' if err or not v.grade else v.grade}  {rel}" + (f"  ({err})" if err else ""))
        line, code = summarise(results, schema)
        print(line)
        if junit:
            from courant.verify.junit import render_junit

            Path(junit).write_text(render_junit(results, base))
            print(f"junit written: {junit}", file=sys.stderr)
        raise typer.Exit(code)
    if not path:
        print("give a case directory or a log file (see --help)", file=sys.stderr)
        raise typer.Exit(4)
    from courant.verify import from_path, verify_inputs, with_spec

    if spec_sha256 and not spec:
        print("courant verify: --spec-sha256 needs --spec", file=sys.stderr)
        raise typer.Exit(4)
    try:
        inputs = from_path(path, case_json)
        if spec:
            inputs = with_spec(inputs, spec, spec_sha256)
        if checks == "next":
            inputs = inputs.model_copy(update={"checks": "next"})
        verdict = in_schema(verify_inputs(inputs), schema)
    except InputError as exc:
        print(f"courant verify: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    if html:
        from courant.verify.html import render_html

        Path(html).write_text(render_html(verdict, inputs))
        print(f"report written: {html}", file=sys.stderr)

    if pack:
        from pathlib import Path as _P

        from courant.verify.pack import write_pack

        out = write_pack(verdict, pack, _P(path).name, _P(case_json).name if case_json else None,
                         spec_path=spec, spec_locked_sha256=spec_sha256)
        print(f"pack written: {out}/ (verdict.json, report.md, inputs.manifest)", file=sys.stderr)
    if record:
        from courant.record import write_record

        out = write_record(verdict, record, target_name=Path(path).name,
                           case_json_name=Path(case_json).name if case_json else None,
                           spec_path=spec, spec_locked_sha256=spec_sha256,
                           producer=producer, requester=requester)
        print(f"record written: {out}", file=sys.stderr)
    if as_json:
        sys.stdout.write(render_json(verdict))
    elif report == "md":
        sys.stdout.write(render_markdown(verdict))
    elif report == "html":
        from courant.verify.html import render_html

        sys.stdout.write(render_html(verdict, inputs))
    else:
        sys.stdout.write(render_text(verdict))
    raise typer.Exit(EXIT[verdict.verdict])
