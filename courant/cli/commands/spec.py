"""`courant spec`: write, lock and check a task specification (PRD 2.0 P2-SPEC).

    courant spec init ./reference_case -o spec.json --task "pitzDaily, inlet 10 m/s"
    courant spec hash spec.json                      # lock it before the run
    courant spec check ./agent_case --spec spec.json # conformance only, before or after a run
    courant verify ./agent_case --spec spec.json --spec-sha256 <hash>

`check` needs no log: conformance is decided by the case as set up, so it can run before the
solver spends anything. Exit codes: 0 every named value matches; 2 a value does not; 3 a
value could not be read, or the specification is impossible in itself (SPEC-9); 4 the inputs
could not be read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(help="Write, lock and check a task specification (courant.spec/1).",
                  no_args_is_help=True, add_completion=False)


@app.command("init")
def init(
    case: str = typer.Argument(..., help="The reference case the task is written against."),
    out: str = typer.Option(None, "-o", "--out", help="Write the specification here (default: stdout)."),
    task: str = typer.Option(None, "--task", help="One line saying what the task asks for."),
    no_mesh: bool = typer.Option(False, "--no-mesh", help="Leave the mesh cell counts out (they are a modelling choice)."),
    checks: str = typer.Option("default", "--checks", help="'default', or 'next': leave out "
                               "blockMeshDict's cells or scale when the case's own mesh does "
                               "not give them, as `--checks next` compares them with the mesh "
                               "the solver read (TRD SPEC-13)."),
) -> None:
    """Draft a specification from a reference case. Review it, then lock it with `courant spec hash`."""
    from courant.spec import draft_spec
    from courant.verify.checks import MODES

    if checks not in MODES:
        print(f"courant spec init: --checks is one of {', '.join(MODES)}, not {checks!r}",
              file=sys.stderr)
        raise typer.Exit(4)
    if not Path(case).is_dir():
        print(f"courant spec init: no such case directory: {case}", file=sys.stderr)
        raise typer.Exit(4)
    try:
        spec = draft_spec(case, task=task, include_mesh=not no_mesh, checks=checks)
    except ValueError as exc:
        print(f"courant spec init: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    text = spec.to_json()
    if out:
        Path(out).write_text(text)
        from courant.spec import spec_sha256

        print(f"specification written: {out} ({len(spec.items)} values) · sha256 "
              f"{spec_sha256(out)}", file=sys.stderr)
    else:
        sys.stdout.write(text)


@app.command("hash")
def hash_(spec: str = typer.Argument(..., help="A courant.spec/1 file.")) -> None:
    """Print the SHA-256 that locks this specification. Record it before the run."""
    from courant.spec import SpecError, load_spec

    try:
        _, sha = load_spec(spec)
    except SpecError as exc:
        print(f"courant spec hash: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    print(sha)


@app.command("check")
def check(
    case: str = typer.Argument(..., help="The case to check, before or after its run."),
    spec: str = typer.Option(..., "--spec", help="The task specification (courant.spec/1)."),
    spec_sha256: str = typer.Option(None, "--spec-sha256", help="The hash it was locked at."),
    as_json: bool = typer.Option(False, "--json", help="Print the conformance block as JSON."),
    checks: str = typer.Option("default", "--checks", help="'default', or 'next': also compare "
                               "the mesh items with the mesh the solver read, read stopAt and "
                               "the turbulence switch, and read 0/ values at the time the run "
                               "starts from (TRD SPEC-13). No log is read here."),
) -> None:
    """Is this case the problem that was asked? Every named value, compared to the case."""
    from courant.spec import SpecError, check_conformance, load_spec
    from courant.verify.checks import MODES
    from courant.verify.schema import canonical_json

    if checks not in MODES:
        print(f"courant spec check: --checks is one of {', '.join(MODES)}, not {checks!r}",
              file=sys.stderr)
        raise typer.Exit(4)
    if not Path(case).is_dir():
        print(f"courant spec check: no such case directory: {case}", file=sys.stderr)
        raise typer.Exit(4)
    try:
        spec_model, sha = load_spec(spec)
    except SpecError as exc:
        print(f"courant spec check: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    locked = spec_sha256.strip().lower() if spec_sha256 else None
    block = check_conformance(spec_model, case, spec_sha256=sha, locked_sha256=locked,
                              checks=checks)
    if locked and locked != sha:
        print(f"SPECIFICATION CHANGED  {spec} is {sha[:12]}, locked at {locked[:12]}")
        raise typer.Exit(3)
    if block.invalid:                                   # SPEC-9: nothing was compared
        if as_json:
            sys.stdout.write(canonical_json(block.model_dump(mode="json")))
        else:
            print(f"SPECIFICATION IMPOSSIBLE  {spec} cannot be met by any case; nothing was "
                  "compared")
            for why in block.invalid:
                print(f"  ✗ {why}")
        raise typer.Exit(3)
    if as_json:
        sys.stdout.write(canonical_json(block.model_dump(mode="json")))
    else:
        ok = sum(1 for i in block.items if i.conforms)
        head = ("CONFORMS" if not block.nonconformant and not block.unreadable
                else "DOES NOT CONFORM" if block.nonconformant else "NOT FULLY CHECKED")
        print(f"{head}  {ok} of {len(block.items)} specified values match · spec {sha[:12]}")
        width = max(len(i.name) for i in block.items)
        for i in block.items:
            mark = "✓" if i.conforms else "✗" if i.conforms is False else "?"
            detail = (f"{i.found}" if i.conforms else
                      (i.note or f"{i.found} in the case, {i.specified} specified")
                      if i.conforms is False else f"not checked: {i.note}")
            print(f"  {mark} {i.name:<{width}}  {detail}")
    raise typer.Exit(2 if block.nonconformant else 3 if block.unreadable else 0)


@app.command("extract")
def extract(
    task_file: str = typer.Argument(..., help="The task, in words (a text file)."),
    out: str = typer.Option(None, "-o", "--out", help="Write the specification here (default: stdout only)."),
    model: str = typer.Option(None, "--model", help="Pin a model id; the door still decides whether it may take the call."),
    data: str = typer.Option("customer", "--data", help="'customer' (the default: only zero-data-retention endpoints) or 'public' for a task from public tutorials."),
) -> None:
    """Draft a specification from a task in words through the model door (SPEC-7).

    The draft is shown, hashed and marked extracted. Until a person confirms it
    (`courant spec confirm --by NAME`) it is not binding: a mismatch can only make a verdict
    insufficient, never reject. If no model may extract it, the exit code is 4 (SPEC-8: only a model
    measured by RRD RM-23).
    """
    from courant.spec import spec_sha256
    from courant.spec.extract import ExtractionError, extract_spec

    path = Path(task_file)
    if not path.is_file():
        print(f"courant spec extract: no such task file: {task_file}", file=sys.stderr)
        raise typer.Exit(4)
    if data not in ("customer", "public"):
        print("courant spec extract: --data is 'customer' or 'public'", file=sys.stderr)
        raise typer.Exit(4)
    from courant.llm import LLMError, contact_notice, default_door

    door = default_door()
    # Asked before anything touches the network (docs/NETWORK.md): a call the door refuses
    # contacts nothing; one it lets through names its host first.
    chosen = door.eligible(task="spec-extract", data=data, model=model)
    if not chosen:
        why = door.why_not(task="spec-extract", data=data, model=model)
        print(f"courant spec extract: no model may extract this specification: {why}. Nothing "
              "was contacted. Write it by hand, or draft it from a reference case (courant spec "
              "init).", file=sys.stderr)
        raise typer.Exit(4)
    print(f"courant spec extract: {contact_notice(chosen[0].provider)}; the task's text is sent",
          file=sys.stderr)
    try:
        result = extract_spec(path.read_text(errors="replace"), door=door, data=data,
                              model=model)
    except LLMError as exc:
        print(f"courant spec extract: no model may extract this specification: {exc.message} "
              "Write it by hand, or draft it from a reference case (courant spec init).",
              file=sys.stderr)
        raise typer.Exit(4) from exc
    except ExtractionError as exc:
        print(f"courant spec extract: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    text = result.spec.to_json()
    sys.stdout.write(text)                              # shown, before anything relies on it
    for why in result.dropped:
        print(f"  left out: {why}", file=sys.stderr)
    where = ""
    if out:
        Path(out).write_text(text)
        where = f"{out} · sha256 {spec_sha256(out)} · "
    print(f"specification extracted by {result.model}: {where}{len(result.spec.items)} values, "
          "UNCONFIRMED. It can only lower a verdict to insufficient until a person reviews it: "
          "courant spec confirm SPEC --by NAME", file=sys.stderr)


@app.command("confirm")
def confirm_(
    spec: str = typer.Argument(..., help="A courant.spec/1 file, typically an extracted one."),
    by: str = typer.Option(..., "--by", help="Who reviewed it."),
    date: str = typer.Option(None, "--date", help="The review date, YYYY-MM-DD (default: today, UTC)."),
    out: str = typer.Option(None, "-o", "--out", help="Write the confirmed specification here (default: in place)."),
) -> None:
    """Record a person's review in the specification and print its new hash (SPEC-7).

    A confirmed specification is binding: its values can reject. Lock the new hash.
    """
    from datetime import UTC, datetime

    from courant.spec import SpecError, load_spec, spec_sha256
    from courant.spec.extract import confirm

    try:
        spec_model, _ = load_spec(spec)
        confirmed = confirm(spec_model, by=by,
                            date=date or datetime.now(UTC).date().isoformat())
    except (SpecError, ValueError) as exc:
        print(f"courant spec confirm: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    target = out or spec
    Path(target).write_text(confirmed.to_json())
    print(spec_sha256(target))
    print(f"confirmed by {confirmed.provenance.confirmed_by} on "
          f"{confirmed.provenance.confirmed_date}: {target} is binding; lock this hash",
          file=sys.stderr)


@app.command("lock")
def lock(spec: str = typer.Argument(..., help="A courant.spec/1 file.")) -> None:
    """Print the SHA-256 that locks this specification, and say whether it binds (SPEC-7).

    Record the hash before the run and pass it as --spec-sha256. An extracted specification
    nobody confirmed locks like any other, and still only lowers a verdict to insufficient.
    """
    from courant.spec import SpecError, load_spec
    from courant.spec.extract import binding

    try:
        spec_model, sha = load_spec(spec)
    except SpecError as exc:
        print(f"courant spec lock: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    print(sha)
    if binding(spec_model):
        print(f"{spec}: binding; a value the case does not match rejects the run", file=sys.stderr)
    else:
        print(f"{spec}: extracted by {spec_model.provenance.model or 'a model'} and UNCONFIRMED; "
              "a mismatch only makes the verdict insufficient. To make it binding: courant spec "
              "confirm --by NAME, then lock the new hash", file=sys.stderr)
