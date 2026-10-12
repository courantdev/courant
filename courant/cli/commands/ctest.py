"""`courant test`: the Courant Test (TRD CT-8)."""

from __future__ import annotations

import sys

import typer

app = typer.Typer(help="The Courant Test: can AI do engineering simulation, and tell when it is wrong?",
                  no_args_is_help=True)


@app.command("build")
def build() -> None:
    """Build judge-v0 from certification/corpus (tasks and hash-locked labels)."""
    from courant.ctest.build import LABELS, TASKS, build_judge_v0, write

    tasks, labels = build_judge_v0()
    write(tasks, labels)
    print(f"{len(tasks)} tasks -> {TASKS.name}; labels -> {LABELS.name}")


@app.command("run")
def run(judge: str = typer.Option("courant", help="courant | constant:<VERDICT> | llm:<provider>:<model>. "
                                   "An llm judge sends each task's text to that provider with your "
                                   "key, only once the model door lets that model judge, and names "
                                   "the host first; otherwise it contacts nothing. The others are "
                                   "offline."),
        split: str = typer.Option("all", help="all | held-in | held-out"),
        out: str = typer.Option(None, help="write the full result JSON here")) -> None:
    """Score a judge. Every rate prints with its denominator."""
    from courant.ctest.run import run as run_ctest
    from courant.verify.schema import canonical_json

    door = None
    if judge.startswith("llm:"):
        from courant.llm import contact_notice, default_door

        _, provider, model = (judge.split(":", 2) + ["", ""])[:3]
        door = default_door()
        # Asked before anything touches the network: a call the door refuses contacts nothing
        # (docs/NETWORK.md); one it lets through names its host first.
        reason = door.why_not(task="judge", data="public", model=model or None,
                              provider=provider or None)
        if reason is not None:
            print(f"stopped, nothing scored or written: {reason}. Nothing was contacted.",
                  file=sys.stderr)
            raise typer.Exit(3)
        print(f"courant test run: {contact_notice(provider)}, through the model door; each "
              "task's text is sent", file=sys.stderr)
    try:
        result = run_ctest(judge, split, door=door)
    except Exception as exc:
        from courant.llm import stops_batch

        if not stops_batch(exc):
            raise
        # A stop is not the judge's answer: nothing is scored or written (ARC-19).
        until = getattr(exc, "until", None)
        print(f"stopped, nothing scored or written: {exc}"
              + (f" (resumes after {until:%Y-%m-%d %H:%M} UTC)" if until is not None else ""),
              file=sys.stderr)
        raise typer.Exit(3) from None
    if out:
        with open(out, "w") as fh:
            fh.write(canonical_json(result))
    for name in ("all", "held_in", "held_out"):
        c = result["scorecard"][name]
        if not c["tasks"]:
            continue
        print(f"{name:9s} tasks {c['tasks']:3d} · accuracy {c['accuracy']['n']}/{c['accuracy']['of']}"
              f" · false accepts {c['far']['n']}/{c['far']['of']}"
              f" · false alarms {c['frr']['n']}/{c['frr']['of']}"
              f" · abstain {c['abstain']['n']}/{c['abstain']['of']}"
              f" · top-2 {c['top2']['n']}/{c['top2']['of']}")
    sys.stdout.flush()
