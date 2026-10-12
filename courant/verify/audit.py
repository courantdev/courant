"""The compute audit: what a folder of runs cost, and how much of it bought nothing (TRD AUDIT-1 to 4).

    courant audit runs/ --cost-per-core-hour 0.05 --html audit.html

For every run under a directory (`batch.discover`), it reports the verdict, the convergence
block where the run has monitors, and the preflight result. Then it totals the compute,
from the logs alone:
- **core-hours per run** are `ranks × final ClockTime / 3600`. `nProcs` comes from the log
  banner, and a serial run counts as 1. ClockTime is the solver's wall clock, MPI waits
  included;
- **on results that cannot be trusted:** the core-hours of runs that ended NOT_TRUSTWORTHY;
- **after the answer stopped changing:** the sum of each run's `wasted_core_hours`, on runs
  whose monitors settled;
- **on setups broken before they started:** the core-hours of runs whose preflight found a
  setup fault. All of that could have been saved by checking first.

Each total carries its denominator. A run with no clock is counted, excluded from the hours,
and named. Money appears only with the user's own rate, and is labelled as such. Read-only and
deterministic, like everything else here.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_CLOCK = re.compile(r"ExecutionTime = \S+ s\s+ClockTime = (\S+) s")
_NPROCS = re.compile(r"^nProcs\s*:\s*(\d+)", re.M)
SCHEMA = "courant.audit/1"


def core_hours(log_text: str) -> tuple[float | None, int]:
    """(ranks × final ClockTime / 3600, ranks), or (None, ranks) when the log has no clock."""
    clocks = _CLOCK.findall(log_text[-20000:]) or _CLOCK.findall(log_text)
    m = _NPROCS.search(log_text[:20000])
    ranks = int(m.group(1)) if m else 1
    if not clocks:
        return None, ranks
    try:
        return ranks * float(clocks[-1]) / 3600.0, ranks
    except ValueError:
        return None, ranks


def audit(root: str | Path, cost_per_core_hour: float | None = None) -> dict:
    from courant import __version__
    from courant.verify import from_path, verify_inputs
    from courant.verify.batch import discover
    from courant.verify.ingest import InputError
    from courant.verify.preflight import preflight_path

    root = Path(root)
    runs = []
    for target, sidecar in discover(root):
        rel = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
        row: dict = {"run": rel}
        try:
            inputs = from_path(target, sidecar)
            v = verify_inputs(inputs).to_v1()             # courant.audit/1 keeps its words
        except InputError as exc:
            row.update({"error": str(exc)})
            runs.append(row)
            continue
        hours, ranks = core_hours(inputs.log_text)
        pre = None
        try:
            if target.is_dir():
                pre = preflight_path(target).verdict
            elif sidecar is not None:
                pre = preflight_path("unused", sidecar).verdict
        except Exception:                                   # noqa: BLE001 - the audit goes on
            pre = None
        c = v.convergence
        row.update({
            "verdict": v.verdict, "rule": v.rule, "grade": v.grade,
            "cause": v.probable_causes[0].code if v.probable_causes and v.verdict != "TRUSTWORTHY" else None,
            "core_hours": hours, "ranks": ranks, "preflight": pre,
            "convergence": c.status if c is not None and c.monitors_found else None,
            "wasted_core_hours": float(c.wasted_core_hours) if c is not None and c.wasted_core_hours else None,
            "wasted_fraction": float(c.wasted_fraction) if c is not None and c.wasted_fraction else None,
        })
        runs.append(row)

    ok = [r for r in runs if "error" not in r]
    clocked = [r for r in ok if r["core_hours"] is not None]

    def hours(rows) -> float:
        import math

        return math.fsum(r["core_hours"] for r in rows if r["core_hours"] is not None)

    untrusted = [r for r in clocked if r["verdict"] == "NOT_TRUSTWORTHY"]
    setup = [r for r in clocked if r["preflight"] == "SETUP_FAULT"]
    settled = [r for r in ok if r["wasted_core_hours"] is not None]
    import math

    totals = {
        "runs": len(runs), "readable": len(ok), "unreadable": len(runs) - len(ok),
        "trustworthy": sum(1 for r in ok if r["verdict"] == "TRUSTWORTHY"),
        "not_trustworthy": sum(1 for r in ok if r["verdict"] == "NOT_TRUSTWORTHY"),
        "not_assessable": sum(1 for r in ok if r["verdict"] == "NOT_ASSESSABLE"),
        "runs_with_clock": len(clocked), "runs_without_clock": [r["run"] for r in ok if r["core_hours"] is None],
        "core_hours": hours(clocked),
        "core_hours_not_trustworthy": hours(untrusted), "runs_not_trustworthy_clocked": len(untrusted),
        "core_hours_setup_fault": hours(setup), "runs_setup_fault": sum(1 for r in ok if r["preflight"] == "SETUP_FAULT"),
        "core_hours_after_convergence": math.fsum(r["wasted_core_hours"] for r in settled),
        "runs_settled": len(settled),
        "runs_with_monitors": sum(1 for r in ok if r["convergence"] is not None),
    }
    money = None
    if cost_per_core_hour is not None:
        rate = float(cost_per_core_hour)
        money = {"rate_per_core_hour": rate, "basis": "the user's own rate",
                 **{k.replace("core_hours", "cost"): totals[k] * rate
                    for k in ("core_hours", "core_hours_not_trustworthy", "core_hours_setup_fault",
                              "core_hours_after_convergence")}}
    causes = Counter(r["cause"] for r in ok if r.get("cause"))
    return {"schema": SCHEMA, "tool": {"name": "courant", "version": __version__},
            "root": root.name, "runs": runs, "totals": totals, "money": money,
            "causes": dict(sorted(causes.items(), key=lambda kv: (-kv[1], kv[0])))}


def _h(x: float) -> str:
    return f"{x:,.4g}"


def _runs(n: int) -> str:
    return f"{n} run" + ("" if n == 1 else "s")


def render_text(doc: dict) -> str:
    t, m = doc["totals"], doc.get("money")
    n = t["readable"]
    out = [f"Courant audit · {doc['root']} · {t['runs']} runs"
           + (f" ({t['unreadable']} unreadable)" if t["unreadable"] else ""), ""]
    out.append(f"  Trust        {t['trustworthy']} of {n} trustworthy · {t['not_trustworthy']} of {n} "
               f"not trustworthy · {t['not_assessable']} of {n} not assessable")
    out.append(f"  Compute      {_h(t['core_hours'])} core-hours across {t['runs_with_clock']} of {n} runs "
               "with a solver clock")
    out.append(f"  Untrusted    {_h(t['core_hours_not_trustworthy'])} core-hours on "
               f"{_runs(t['runs_not_trustworthy_clocked'])} that ended NOT_TRUSTWORTHY")
    out.append(f"  Avoidable    {_h(t['core_hours_setup_fault'])} core-hours on "
               f"{_runs(t['runs_setup_fault'])} whose setup fault preflight finds before the first "
               "timestep")
    out.append(f"  Settled      {_h(t['core_hours_after_convergence'])} core-hours after the answer "
               f"stopped changing, on {t['runs_settled']} of {_runs(t['runs_with_monitors'])} with "
               "monitors")
    if m:
        out += ["", f"  At your rate of {m['rate_per_core_hour']:g} per core-hour: "
                f"{_h(m['cost'])} in total, {_h(m['cost_not_trustworthy'])} on untrusted results, "
                f"{_h(m['cost_setup_fault'])} avoidable before the start, "
                f"{_h(m['cost_after_convergence'])} after convergence."]
    if t["runs_without_clock"]:
        out += ["", f"  No solver clock (counted, not in the hours): {', '.join(t['runs_without_clock'][:8])}"
                + (" …" if len(t["runs_without_clock"]) > 8 else "")]
    if doc["causes"]:
        out += ["", "  Leading causes of NOT_TRUSTWORTHY runs:"]
        out += [f"    {k:28s} {v}" for k, v in doc["causes"].items()]
    out += ["", "  Per run: courant audit --json. Every figure is computed from the run's own log and "
                "files; nothing left this machine."]
    return "\n".join(out) + "\n"


def render_html(doc: dict) -> str:
    from html import escape

    from courant.verify.html import CSS, _table

    t = doc["totals"]
    n = t["readable"]
    rows = [
        f"<tr><td class='mono'>{escape(r['run'])}</td><td>{escape(r.get('verdict') or 'unreadable')}</td>"
        f"<td>{escape(r.get('cause') or '—')}</td><td>{escape(r.get('preflight') or '—')}</td>"
        f"<td>{escape(r.get('convergence') or '—')}</td>"
        f"<td class='num'>{'—' if r.get('core_hours') is None else _h(r['core_hours'])}</td>"
        f"<td class='num'>{'—' if r.get('wasted_core_hours') is None else _h(r['wasted_core_hours'])}</td></tr>"
        for r in doc["runs"]]
    cards = [
        ("Runs that can be trusted", f"{t['trustworthy']} of {n}"),
        ("Core-hours on untrusted results", f"{_h(t['core_hours_not_trustworthy'])}"),
        ("Core-hours avoidable before the start", f"{_h(t['core_hours_setup_fault'])}"),
        ("Core-hours after the answer settled", f"{_h(t['core_hours_after_convergence'])}"),
    ]
    card_html = "".join(f"<div class='card'><div class='meta'>{escape(k)}</div>"
                        f"<div style='font-size:24px;font-weight:600'>{escape(v)}</div></div>"
                        for k, v in cards)
    money = doc.get("money")
    money_html = "" if not money else (
        f"<p>At your rate of {money['rate_per_core_hour']:g} per core-hour: {_h(money['cost'])} in "
        f"total; {_h(money['cost_not_trustworthy'])} on untrusted results; "
        f"{_h(money['cost_setup_fault'])} avoidable before the start; "
        f"{_h(money['cost_after_convergence'])} after convergence.</p>")
    return ("<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Courant audit · {escape(doc['root'])}</title><style>{CSS}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}"
            "</style></head><body><main>"
            f"<h1>Compute audit: {escape(doc['root'])}</h1>"
            f"<div class='meta'>{t['runs']} runs · {t['runs_with_clock']} with a solver clock · "
            f"courant {escape(doc['tool']['version'])}</div>"
            f"<div class='grid'>{card_html}</div>{money_html}"
            "<h2>Every run</h2>"
            + _table("Every run in the folder, with its verdict and core-hours",
                     ("Run", "Verdict", "Leading cause", "Preflight", "Convergence", "Core-hours",
                      "After settling"), rows)
            + "</main><footer>Computed from each run's own log and files: core-hours are ranks × "
            "the solver's final ClockTime. Nothing left this machine. Courant did not edit any case "
            "or run any solver.</footer></body></html>\n")
