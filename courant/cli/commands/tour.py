"""`courant demo`: ten seconds that show what Courant does, on runs it ships (TRD PKG-6).

Offline, one core, read-only. Nothing is written outside a temporary folder unless
`--keep DIR` is given. The runs are real OpenFOAM v2406 output, packaged in `courant/data/`:
a healthy cavity, the same cavity with a timestep ten times too large, and pitzDaily run to a
fixed 2,000-iteration budget with monitors.

It calls the same functions `courant verify` and `courant convergence` call. A demo with its
own code path would be a demo of nothing.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import typer


def demo(
    keep: str = typer.Option(None, "--keep", help="Write the report and evidence pack here instead of a temporary folder."),
) -> None:
    """See Courant verify four real runs, offline, in seconds."""
    from courant.converge import from_run_inputs
    from courant.converge.render import five_lines
    from courant.data import DEMO, certification, extract_demo_case
    from courant.verify import from_path, verify_inputs
    from courant.verify.html import render_html
    from courant.verify.pack import write_pack

    t0 = time.perf_counter()
    out = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="courant-demo-"))
    out.mkdir(parents=True, exist_ok=True)

    def say(line: str = "") -> None:
        print(line, flush=True)

    say("Courant demo · offline · read-only · four real OpenFOAM v2406 runs")
    say()

    runs = [("1", "A healthy run", "icoFoam cavity, as OpenFOAM ships it", "icoFoam_cavity__healthy"),
            ("2", "A broken run", "the same cavity with a timestep ten times too large",
             "icoFoam_cavity__cfl_blowup")]
    for n, title, what, name in runs:
        work = out / name
        work.mkdir(exist_ok=True)
        for ext in (".log.gz", ".case.json"):
            shutil.copy(DEMO / f"{name}{ext}", work / f"{name}{ext}")
        inputs = from_path(work / f"{name}.log.gz", work / f"{name}.case.json")
        v = verify_inputs(inputs)
        say(f"{n}  {title}: {what}")
        say(f"     {v.verdict}  (rule {v.rule}, grade {v.grade or '-'})")
        for r in v.reasons[:2] if v.verdict != "accept" else []:
            say(f"     why     {r.code}: {r.summary}")
        if v.verdict != "accept" and v.probable_causes:
            say(f"     cause   {v.probable_causes[0].code} ({v.probable_causes[0].name})")
        if v.next_action:
            say(f"     next    {v.next_action}")
        say()

    # Preflight: a setup fault caught before the solver would have run at all.
    from courant.verify.preflight import preflight_path

    name = "icoFoam_cavity__transport_units"
    work = out / name
    work.mkdir(exist_ok=True)
    for ext in (".log.gz", ".case.json"):
        shutil.copy(DEMO / f"{name}{ext}", work / f"{name}{ext}")
    pre = preflight_path("unused", work / f"{name}.case.json")
    say("3  Before the run: the cavity with its viscosity in the wrong units (courant preflight)")
    say(f"     {pre.verdict}  before the first timestep")
    for r in pre.reasons[:1]:
        say(f"     why     {r.code}: {r.summary}")
    if pre.next_action:
        say(f"     next    {pre.next_action}")
    say()

    case = extract_demo_case(out)
    inputs = from_path(case)
    v = verify_inputs(inputs)
    conv = from_run_inputs(inputs)
    say("4  A steady run with a fixed budget: pitzDaily, 2,000 iterations, monitors on the walls "
        "and the inlet")
    say(f"     {v.verdict}  (rule {v.rule}, grade {v.grade or '-'}); convergence {conv.status}")
    for ln in five_lines(conv):
        say(f"     {ln}")
    say()
    report = out / "report.html"
    report.write_text(render_html(v, inputs))
    pack = write_pack(v, out / "pack", case.name, None)
    from courant.verify.audit import audit

    doc = audit(out)
    t = doc["totals"]
    settled = [r for r in doc["runs"] if r.get("wasted_fraction") is not None]
    say(f"5  The compute audit over these {t['runs']} runs (courant audit)")
    say(f"     {t['trustworthy']} of {t['readable']} trustworthy · {t['runs_setup_fault']} of "
        f"{t['readable']} with a setup fault preflight finds before the start · on "
        f"{t['runs_settled']} of {t['runs_with_monitors']} monitored runs the answer settled"
        + (f", with {100 * settled[0]['wasted_fraction']:.1f}% of its iterations still to run"
           if len(settled) == 1 else ""))
    say("     On your runs it totals the core-hours, and the cost at your rate.")
    say()
    say("6  Evidence")
    say(f"     report   {report}")
    say(f"     pack     {pack}/  (re-derive it: courant verify --check-pack {pack} --root {case})")
    say()
    cert = certification() or {}
    card = (cert.get("scorecard") or {}).get("all") or {}
    if card:
        far, frr = card.get("far", {}), card.get("frr", {})
        say(f"What Courant measured about itself, on {card.get('n', 31)} real runs: false accepts "
            f"{far.get('n')} of {far.get('of')}, false alarms {frr.get('n')} of {frr.get('of')} "
            f"(courant verify --about).")
    say(f"Done in {time.perf_counter() - t0:.1f} s. Nothing outside {out} was written, no solver "
        f"was run, and no network was used.")
    if not keep:
        say("The folder is temporary; pass --keep DIR to keep the report.")
    sys.stdout.flush()
