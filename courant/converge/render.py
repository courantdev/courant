"""How a convergence assessment is shown (TRD CONV-13, M1-A-13).

Five lines, in this order, whenever a stopping point exists:
1. the answer at the stopping point and at the end;
2. how much the answer changed after the stopping point;
3. iterations past convergence;
4. wall clock past convergence;
5. core-hours past convergence, or the fixed sentence saying why they were not computed.

Every number printed is taken from the assessment's data, and every share carries its
denominator.
"""

from __future__ import annotations

from courant.converge.assess import FAIL, NOT_ASSESSABLE, PASS, WARN, Convergence

FOOTER = ("Courant reports where this run could have stopped. It does not say where a future "
          "run will.")
MAX_ROWS = 24


def _g(x) -> str:
    return "n/a" if x is None else f"{x:.6g}"


def _int(n) -> str:
    return "n/a" if n is None else f"{n:,}"


def _pct(x) -> str:
    if x is None:
        return "n/a"
    if 0 < abs(x) < 0.001:
        return f"{100.0 * x:.2g}%"            # 0.021%, not 0.0%
    return f"{100.0 * x:.1f}%"


def _change(row: dict) -> str:
    if row.get("delta") is None:
        return "-"
    if row.get("delta_counts") is not None:
        return f"{row['delta_counts']:.2f} counts"
    if row.get("delta_fraction") is not None:
        return f"{_pct(row['delta_fraction'])} of scale"
    return _g(row["delta"])


def _limit(row: dict) -> str:
    tol = row.get("tolerance") or {}
    if tol.get("branch") == "absolute":
        if tol.get("value") == 1e-4:
            return "1 count"
        return f"{_g(tol.get('value'))} absolute"
    return f"{_pct(tol.get('value'))} of scale"


def summary(conv: Convergence) -> dict:
    """The compact block a verdict and the MCP tool carry (CONV-12)."""
    d = conv.data
    w = d.get("waste") or {}
    b = d.get("binding_monitor")
    row = (d.get("monitors") or {}).get(b) or {}
    return {
        "status": conv.status, "regime": conv.regime, "enters_verdict": conv.counted,
        "converged_at_step": d.get("converged_at_step"), "binding_monitor": b,
        "change_after_stop": row.get("delta"), "change_counts": row.get("delta_counts"),
        "change_fraction": row.get("delta_fraction"),
        "wasted_steps": w.get("wasted_steps"), "steps_total": w.get("steps_total"),
        "wasted_fraction": w.get("wasted_fraction"),
        "wasted_wall_clock_s": w.get("wasted_wall_clock_s"),
        "wasted_core_hours": w.get("wasted_core_hours"),
        "monitors_assessed": len([m for m in (d.get("monitors") or {}).values()
                                  if m.get("counts") and m.get("verdict") != NOT_ASSESSABLE]),
        "monitors_found": len(d.get("monitors") or {}),
    }


def five_lines(conv: Convergence) -> list[str]:
    d = conv.data
    w = d.get("waste")
    b = d.get("binding_monitor")
    if not w or not b:
        return []
    row = d["monitors"][b]
    unit = "iterations" if conv.regime == "steady" else "time steps"
    noun = "value" if conv.regime == "steady" else "time-average"
    lines = [
        f"1  Answer       {b}: {noun} {_g(row['value_at_stop'])} at step {_int(w['converged_at_step'])}, "
        f"{_g(row['value_at_end'])} at the end (step {_int(w['final_step'])})",
        f"2  Change       {_change(row)} after the stopping point (limit {_limit(row)})",
        f"3  {unit.capitalize():<12} {_int(w['wasted_steps'])} of {_int(w['steps_total'])} past convergence "
        f"({_pct(w['wasted_fraction'])})",
    ]
    if w.get("wasted_wall_clock_s") is not None:
        total = w.get("wall_clock_total_s")
        of = f" of {_g(total)} s" if total is not None else ""
        cpu = (f"; rank-0 CPU {_g(w['wasted_cpu_s'])} s" if w.get("wasted_cpu_s") is not None else "")
        lines.append(f"4  Wall clock   {_g(w['wasted_wall_clock_s'])} s{of} past convergence "
                     f"(solver ClockTime{cpu})")
    else:
        lines.append("4  Wall clock   not available: the log carried no clock at these steps")
    if w.get("wasted_core_hours") is not None:
        cost = ""
        if w.get("wasted_cost") is not None:
            cost = (f"; {_g(w['wasted_cost'])} at your rate of {_g(w['cost_per_core_hour'])} "
                    f"per core-hour")
        basis = {"log_nprocs": "nProcs in the log", "serial": "a serial run",
                 "user": "as given"}.get(w["ranks_basis"], w["ranks_basis"])
        lines.append(f"5  Core-hours   {_g(w['wasted_core_hours'])} past convergence "
                     f"({w['ranks']} rank{'s' if w['ranks'] != 1 else ''}, {basis}, x wall clock)"
                     f"{cost}")
    else:
        note = next((f.message for f in conv.findings if f.code == "CONV.CORE_HOURS_NOT_COMPUTED"), "")
        lines.append(f"5  Core-hours   {note}")
    return lines


def render_text(conv: Convergence, name: str = "", solver: str | None = None) -> str:
    d = conv.data
    head = " · ".join(x for x in (name, solver, conv.regime or "regime unknown") if x)
    out = [f"Courant convergence · {head}"]
    stop = d.get("converged_at_step")
    lead = {
        PASS: f"  PASS            the answer stopped changing at step {_int(stop)}",
        WARN: "  WARN            converged where it could be assessed; see below",
        FAIL: "  FAIL            the answer did not settle",
        NOT_ASSESSABLE: "  NOT ASSESSABLE  convergence of the answer could not be judged",
    }[conv.status]
    if conv.status == WARN and stop is not None:
        lead = f"  WARN            the answer stopped changing at step {_int(stop)}; see below"
    out += [lead, ""]
    lines = five_lines(conv)
    if lines:
        out += ["  " + ln for ln in lines] + [""]
    for f in conv.findings:
        if f.severity in (FAIL, WARN) and f.message:
            out.append(f"  {f.severity:<4}  {f.message}")
    for f in conv.findings:
        if f.code in ("CONV.TRANSIENT_NO_INTENT",) and f.message:
            out.append(f"  NOTE  {f.message}")
    mons = d.get("monitors") or {}
    if mons:
        if out[-1] != "":
            out.append("")
        out += [f"  Monitors ({len(mons)})"]
        width = min(60, max(len(k) for k in mons))
        for k in sorted(mons)[:MAX_ROWS]:
            r = mons[k]
            stop_s = _int(r.get("stop_step")) if r.get("stop_step") is not None else "-"
            flag = "" if r.get("counts") else f"  [{(r.get('advisory_reason') or '').lower()}]"
            out.append(f"    {k:<{width}}  {r.get('verdict', '-'):<29} stop {stop_s:>8}  {_change(r)}{flag}")
        if len(mons) > MAX_ROWS:
            out.append(f"    … and {len(mons) - MAX_ROWS} more in --json")
    res = d.get("residuals") or {}
    orders = {k: v.get("orders_dropped_floor") for k, v in res.items()
              if v.get("orders_dropped_floor") is not None}
    if orders:
        shown = ", ".join(f"{k} {v}" for k, v in sorted(orders.items()))
        out += ["", f"  Residual orders dropped (Celik et al. 2008 ask for 3, prefer 4): {shown}"]
    out += ["", f"  {FOOTER}"]
    return "\n".join(out) + "\n"


def render_markdown(conv: Convergence) -> str:
    """The convergence section of an evidence report (REP-1, REP-6)."""
    d = conv.data
    out = ["## Convergence", "", f"**Status:** {conv.status} · regime: {conv.regime or 'unknown'}"]
    lines = five_lines(conv)
    if lines:
        out += [""] + [f"- {ln[3:].strip()}" for ln in lines]
    msgs = [f for f in conv.findings if f.severity in (FAIL, WARN) and f.message]
    if msgs:
        out += [""] + [f"- **{f.severity}** `{f.code}`: {f.message}" for f in msgs]
    mons = d.get("monitors") or {}
    if mons:
        out += ["", "| Monitor | Verdict | Stop step | Change after stop |", "|---|---|---|---|"]
        for k in sorted(mons):
            r = mons[k]
            out.append(f"| `{k}` | {r.get('verdict')} | {_int(r.get('stop_step')) if r.get('stop_step') is not None else '-'} | {_change(r)} |")
    out += ["", FOOTER, ""]
    return "\n".join(out)
