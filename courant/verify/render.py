"""Evidence renderers (TRD REP-1 to REP-3, VER-10).

Every renderer is a pure function of the verdict. Nothing appears in a report
that is not in the verdict JSON (REP-2), and nothing is machine- or
time-specific (REP-3).

Each takes either major (GATE-3, ARC-16) and writes in that verdict's own words: a
`courant.verdict/2` says `accept`, `reject` or `insufficient` and states its basis; a
`courant.verdict/1` (`--schema 1`) renders byte for byte as version 1 did.
"""

from __future__ import annotations

from courant.verify.verdict2 import AnyVerdict, VerdictV2

_WHY_HEAD = {"TRUSTWORTHY": "Notes", "NOT_TRUSTWORTHY": "Why", "NOT_ASSESSABLE": "Why",
             "accept": "Notes", "reject": "Why", "insufficient": "Why"}


def _accepted(v: AnyVerdict) -> bool:
    return v.verdict in ("TRUSTWORTHY", "accept")


def _basis(v: AnyVerdict) -> str | None:
    return v.basis if isinstance(v, VerdictV2) else None


def render_json(v: AnyVerdict) -> str:
    return v.to_json()


def _reason_lines(r, indent: str) -> list[str]:
    lines = [f"{r.code} — {r.summary}"]
    lines.append(f"{indent}read    {r.input}")
    lines.append(f"{indent}method  {r.method}")
    if r.value is not None or r.threshold is not None:
        lines.append(f"{indent}value   {r.value if r.value is not None else '—'}"
                     f"   threshold {r.threshold if r.threshold is not None else '—'}")
    return lines


def render_text(v: AnyVerdict) -> str:
    out = [f"VERDICT   {v.verdict}   (rule {v.rule})"]
    if _basis(v):
        out.append(f"BASIS     {_basis(v)}")
    out.append(f"GRADE     {v.grade or '—'}")
    out.append(f"RUN       {v.run.id}" + (f" · {v.run.solver}" if v.run.solver else ""))
    if v.reasons:
        out += ["", _WHY_HEAD[v.verdict].upper()]
        for i, r in enumerate(v.reasons, 1):
            first, *rest = _reason_lines(r, "     ")
            out.append(f"  {i:<2} {first}")
            out += rest
    if v.probable_causes and not _accepted(v):
        out += ["", "PROBABLE CAUSES"]
        out += [f"  {c.rank}  {c.code} — {c.name}" for c in v.probable_causes]
    out += _convergence_text(v)
    out += _conformance_text(v)
    if v.not_assessed:
        out += ["", "NOT CHECKED"]
        out += [f"  - {s}" for s in v.not_assessed]
    if v.next_action:
        out += ["", f"NEXT  {v.next_action}"]
    return "\n".join(out) + "\n"


def _conformance_text(v: AnyVerdict) -> list[str]:
    c = v.conformance
    if c is None:
        return []
    conforming = sum(1 for i in c.items if i.conforms)
    out = ["", f"CONFORMANCE  {conforming} of {len(c.items)} specified values match the case"
               f" · spec {c.spec_sha256[:12]}"
               + (f" (locked {c.locked_sha256[:12]})" if c.locked_sha256 else "")]
    width = max((len(i.name) for i in c.items), default=0)
    for i in c.items:
        if i.conforms:
            continue
        if i.conforms is False:
            what = i.note or f"{i.found} in the case, {i.specified} specified"
            out.append(f"  ✗ {i.name:<{width}}  {what}  [{i.file}; {i.tolerance}]")
        else:
            out.append(f"  ? {i.name:<{width}}  not checked: {i.note}")
    return out


def _share(c) -> str:
    if c.wasted_steps is None:
        return ""
    pct = f" ({100 * float(c.wasted_fraction):.1f}%)" if c.wasted_fraction else ""
    wall = f"; {c.wasted_wall_clock_s} s wall clock" if c.wasted_wall_clock_s else ""
    ch = f"; {c.wasted_core_hours} core-hours" if c.wasted_core_hours else ""
    return f"{c.wasted_steps:,} of {c.steps_total:,} steps{pct} came after it{wall}{ch}"


def _convergence_text(v: AnyVerdict) -> list[str]:
    c = v.convergence
    if c is None or not c.monitors_found:
        return []
    head = f"CONVERGENCE  {c.status} · {c.regime or 'regime unknown'}"
    if c.converged_at_step is not None:
        head = head + f" · the answer stopped changing at step {c.converged_at_step:,}"
    out = ["", head]
    if c.wasted_steps is not None:
        out.append(f"  {_share(c)}")
    out += [f"  {m}" for m in c.messages]
    out.append("  Reported beside the verdict, not in it (VER-5.4b is off). "
               "Details: courant convergence <case>.")
    return out


def render_markdown(v: AnyVerdict) -> str:
    out = [f"# Verification: {v.run.id}", ""]
    out.append(f"**Verdict:** `{v.verdict}` (rule {v.rule})  ")
    if _basis(v):
        out.append(f"**Basis:** `{_basis(v)}`  ")
    out.append(f"**Grade:** {v.grade or '—'}  ")
    out.append(f"**Solver:** {v.run.solver or '—'}  ")
    out.append(f"**Verifier:** {v.tool.name} {v.tool.version}, schema `{v.schema_}`")
    out += ["", f"## {_WHY_HEAD[v.verdict]}", ""]
    if not v.reasons:
        out.append("No reason was raised.")
    else:
        out.append("| # | Code | Severity | Finding | Read | Method | Value | Threshold |")
        out.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(v.reasons, 1):
            out.append(f"| {i} | `{r.code}` | {r.severity} | {_cell(r.summary)} | {_cell(r.input)} "
                       f"| {_cell(r.method)} | {r.value or '—'} | {r.threshold or '—'} |")
    if v.probable_causes:
        out += ["", "## Probable causes", ""]
        out += [f"{c.rank}. `{c.code}` — {c.name}" for c in v.probable_causes]
    c = v.convergence
    if c is not None and c.monitors_found:
        out += ["", "## Convergence", "", f"**Status:** {c.status} · regime: {c.regime or '—'}  "]
        if c.converged_at_step is not None:
            out.append(f"**Stopping point:** step {c.converged_at_step:,}, binding monitor "
                       f"`{c.binding_monitor}`  ")
        if c.wasted_steps is not None:
            out.append(f"**After it:** {_share(c)}  ")
        out += [f"- {m}" for m in c.messages]
        out.append("")
        out.append("Reported beside the verdict, not in it: policy rule VER-5.4b is off.")
    k = v.conformance
    if k is not None:
        out += ["", "## Conformance to the task specification", "",
                f"**Specification:** `{k.spec_schema}`, SHA-256 `{k.spec_sha256}`"
                + (f", locked at `{k.locked_sha256}`" if k.locked_sha256 else "") + "  "]
        if k.task:
            out.append(f"**Task:** {_cell(k.task)}  ")
        out += ["", "| Value | Specified | In the case | Tolerance | Read | Result |",
                "|---|---|---|---|---|---|"]
        for i in k.items:
            result = ("matches" if i.conforms else "**does not match**" if i.conforms is False
                      else f"not checked: {_cell(i.note or '')}")
            if i.conforms is False and i.note:
                result += f" ({_cell(i.note)})"
            name = i.name + (" (modelling choice)" if i.modelling_choice else "")
            out.append(f"| {_cell(name)} | {_cell(i.specified or '—')} | {_cell(i.found or '—')} "
                       f"| {i.tolerance} | `{_cell(i.file)}` {_cell(i.location)}"
                       + (f" ({_cell(i.basis)})" if i.basis else "") + f" | {result} |")
    out += ["", "## Not checked", ""]
    out += [f"- {s}" for s in v.not_assessed] or ["- Nothing."]
    if v.next_action:
        out += ["", "## Next action", "", v.next_action]
    out += ["", "## Inputs", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += [f"| `{f.path}` | {f.bytes} | `{f.sha256}` |" for f in v.run.inputs]
    s = v.stats
    out += ["", f"Parsed {s.events} events, {s.residual_steps} residual steps, "
                f"{s.crashes} crash events, {s.divergences} divergence events."]
    return "\n".join(out) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
