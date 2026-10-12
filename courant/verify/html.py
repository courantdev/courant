"""The HTML evidence report: one self-contained file a manager can open (TRD REP-6).

The page loads nothing from the network, has inline CSS and SVG, runs no JavaScript, and its
bytes depend only on the verdict, the inputs and the packaged error rates. It has no timestamp
and no absolute path. Every number printed as text is in the verdict JSON (REP-2), except the
error rates in the footer, which are read from the packaged results (GATE-12). The charts are drawn from the
log and the monitor files the verdict lists among its inputs, and are captioned as such.
"""

from __future__ import annotations

import math
from html import escape

from courant.verify import about
from courant.verify.verdict2 import AnyVerdict, VerdictV2

W, H = 760, 220
PAD_L, PAD_R, PAD_T, PAD_B = 84, 14, 14, 30
MAX_POINTS = 800
PALETTE = ("var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)")
#: SVG dash patterns, one per series beside its colour (WCAG 1.4.1: not colour alone). The
#: first is solid. The legend swatch draws the same pattern.
DASHES = ((), (7, 3), (1.5, 3), (7, 3, 1.5, 3), (3, 3), (11, 4))

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f21;--muted:#5f6368;--line:#e3e3e0;--card:#ffffff;
--ok:#1f7a4d;--bad:#b42318;--warn:#a15c07;--s1:#2f6fb3;--s2:#c4472f;--s3:#2f8f5b;
--s4:#8a5cb5;--s5:#a87a0a;--s6:#4a4a4a;--shade:rgba(47,111,179,.08);--on-fill:#ffffff}
@media (prefers-color-scheme:dark){:root{--bg:#141516;--fg:#e8e8e6;--muted:#a3a5a8;
--line:#2c2e30;--card:#1b1c1e;--ok:#4cc38a;--bad:#ff6b5c;--warn:#f0a43a;--s1:#6ea8fe;
--s2:#ff8a73;--s3:#5fd39a;--s4:#c39bf0;--s5:#e8c25a;--s6:#bdbdbd;--shade:rgba(110,168,254,.10);
--on-fill:#141516}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px;letter-spacing:.02em}
.meta{color:var(--muted);font-size:13px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.badge{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:600;font-size:14px;
color:var(--on-fill);margin-right:8px}.TRUSTWORTHY{background:var(--ok)}.NOT_TRUSTWORTHY{background:var(--bad)}
.NOT_ASSESSABLE{background:var(--warn)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px 16px;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.scroll{overflow-x:auto}svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
.axis{stroke:var(--muted)}.cap{color:var(--muted);font-size:12px;margin-top:4px}ul{margin:6px 0;padding-left:20px}
.legend span{display:inline-block;margin-right:12px;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:24px;height:3px;margin-right:6px;vertical-align:middle}
footer{max-width:860px;margin:0 auto;padding:0 16px 40px;color:var(--muted);font-size:12px}
.sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;
clip:rect(0 0 0 0);white-space:nowrap;border:0}
:focus-visible{outline:3px solid var(--s1);outline-offset:2px}
"""


def _f(x: float) -> str:
    return f"{x:.1f}"


def _table(label: str, headers: tuple[str, ...], rows: list[str]) -> str:
    """A data table a keyboard and a screen reader can use (WCAG 1.3.1, 2.1.1, 4.1.2).

    The scrolling region is focusable and named, the table has a caption, and every column has
    a header cell with its scope.
    """
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    return (f'<div class="scroll" role="region" tabindex="0" aria-label="{escape(label)}">'
            f'<table><caption class="sr">{escape(label)}</caption><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _swatch(index: int) -> str:
    """The legend's sample of a series: its colour, drawn with its dash pattern."""
    colour = PALETTE[index % len(PALETTE)]
    dash = DASHES[index % len(DASHES)]
    if not dash:
        return f'<i aria-hidden="true" style="background:{colour}"></i>'
    pattern = list(dash) if len(dash) % 2 == 0 else list(dash) * 2
    stops, pos = [], 0.0
    for k, seg in enumerate(pattern):
        c = colour if k % 2 == 0 else "transparent"
        stops.append(f"{c} {pos:g}px {pos + seg:g}px")
        pos += seg
    return (f'<i aria-hidden="true" style="background:repeating-linear-gradient(90deg,'
            f'{",".join(stops)})"></i>')


def _decimate(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= MAX_POINTS:
        return points
    stride = -(-len(points) // MAX_POINTS)
    out = points[::stride]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _chart(series: dict[str, list[tuple[float, float]]], *, log_y: bool, title: str, cid: str,
           marker: float | None = None, marker_label: str = "", skip_start: float = 0.0) -> str:
    """One SVG line chart. Coordinates are rounded to 0.1 px, so the bytes are stable.

    `title` names the chart and `cid` makes its ids unique on the page. The chart has a title and
    a description a screen reader announces, and each series has its own dash pattern as well as
    its colour (WCAG 1.1.1, 1.4.1).
    """
    clean: dict[str, list[tuple[float, float]]] = {}
    for name, pts in series.items():
        keep = [(x, (math.log10(y) if log_y else y)) for x, y in pts
                if math.isfinite(y) and (y > 0 or not log_y)]
        if keep:
            clean[name] = _decimate(keep)
    if not clean:
        return ""
    xs = [p[0] for pts in clean.values() for p in pts]
    x0, x1 = min(xs), max(xs)
    # The y-range ignores the opening `skip_start` of the run (a first-iteration value can be
    # orders away from the answer and flatten everything after it). Points outside it are
    # clipped to the frame rather than dropped.
    cut = x0 + (x1 - x0) * skip_start
    ys = [p[1] for pts in clean.values() for p in pts if p[0] >= cut] or \
         [p[1] for pts in clean.values() for p in pts]
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        y0, y1 = y0 - 1.0, y1 + 1.0
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(x: float) -> float:
        return PAD_L + (x - x0) / (x1 - x0) * pw

    def sy(y: float) -> float:
        return PAD_T + (1.0 - (min(max(y, y0), y1) - y0) / (y1 - y0)) * ph

    names = sorted(clean)
    marked = marker is not None and x0 <= marker <= x1
    desc = (f"Line chart of {len(names)} series ({', '.join(names)}) against step, from step "
            f"{round(x0):,} to step {round(x1):,}"
            + ("; the vertical axis is the base-10 logarithm of the value" if log_y else "")
            + (f"; the answer stopped changing at step {round(marker):,}" if marked else "") + ".")
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="{cid}-t {cid}-d">',
             f'<title id="{cid}-t">{escape(title)}</title>',
             f'<desc id="{cid}-d">{escape(desc)}</desc>']
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T + ph}" x2="{PAD_L + pw}" y2="{PAD_T + ph}"/>')
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + ph}"/>')
    if marker is not None and x0 <= marker <= x1:
        mx = sx(marker)
        parts.append(f'<rect x="{_f(mx)}" y="{PAD_T}" width="{_f(PAD_L + pw - mx)}" height="{ph}" '
                     f'fill="var(--shade)"/>')
        parts.append(f'<line x1="{_f(mx)}" y1="{PAD_T}" x2="{_f(mx)}" y2="{PAD_T + ph}" '
                     f'stroke="var(--s1)" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{_f(mx + 4)}" y="{PAD_T + 12}">{escape(marker_label)}</text>')
    for k in range(5):
        yv = y0 + (y1 - y0) * k / 4
        label = f"1e{yv:.1f}" if log_y else f"{yv:.3g}"
        parts.append(f'<text x="{PAD_L - 6}" y="{_f(sy(yv) + 4)}" text-anchor="end">{label}</text>')
    for k in range(5):
        xv = x0 + (x1 - x0) * k / 4
        xl = f"{round(xv):,}" if x1 - x0 >= 20 else f"{xv:.3g}"     # steps are whole numbers
        parts.append(f'<text x="{_f(sx(xv))}" y="{H - 10}" text-anchor="middle">{xl}</text>')
    for i, (name, pts) in enumerate(sorted(clean.items())):
        d = " ".join(f"{_f(sx(x))},{_f(sy(y))}" for x, y in pts)
        dash = DASHES[i % len(DASHES)]
        dash_attr = f' stroke-dasharray="{" ".join(f"{x:g}" for x in dash)}"' if dash else ""
        parts.append(f'<polyline fill="none" stroke="{PALETTE[i % len(PALETTE)]}" stroke-width="1.6"'
                     f'{dash_attr} points="{d}"/>')
    parts.append("</svg>")
    legend = "".join(f"<span>{_swatch(i)}{escape(n)}</span>" for i, n in enumerate(names))
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def _series_from_inputs(inputs) -> tuple[dict, dict, dict, int | None, str | None]:
    """Residuals, Courant number and the binding monitor's series, from the run's own files."""
    from courant.events.core import EventKind
    from courant.watcher.exitcode import infer_exit_code
    from courant.watcher.parser import LogParser

    code = inputs.exit_code if inputs.exit_code is not None else infer_exit_code(inputs.log_text)
    events = LogParser(run_id=inputs.run_id).parse_text(inputs.log_text, return_code=code)
    res: dict[str, list] = {}
    co: dict[str, list] = {}
    step = 0
    for e in events:
        if e.kind is not EventKind.RESIDUAL:
            continue
        step = step + 1
        for name, v in e.payload.initial_residuals.items():
            res.setdefault(name, []).append((float(step), v))
        if e.payload.courant_max is not None:
            co.setdefault("Courant max", []).append((float(step), e.payload.courant_max))
    mon: dict[str, list] = {}
    stop, binding = None, None
    if inputs.case_dir:
        from courant.converge.series import assess_case

        conv = assess_case(inputs.case_dir, log_text=inputs.log_text, events=events)
        binding = conv.data.get("binding_monitor")
        stop = conv.data.get("converged_at_step")
        rows = conv.data.get("monitors") or {}
        from courant.converge.series import discover
        from pathlib import Path

        found = {m.id: m for m in discover(Path(inputs.case_dir)).monitors}
        chosen = [binding] if binding else [k for k, v in sorted(rows.items())
                                            if v.get("counts")][:1]
        for mid in chosen:
            m = found.get(mid)
            if m is not None:
                steps = rows[mid].get("first_step")
                mon[mid] = [(float(i + (steps or 1)), v) for i, v in enumerate(m.values)]
    return res, co, mon, stop, binding


#: The version 2 badges (GATE-3). Appended, so a `--schema 1` page keeps version 1's bytes.
CSS_V2 = (CSS.rstrip("\n") + "\n.accept{background:var(--ok)}.reject{background:var(--bad)}"
          ".insufficient{background:var(--warn)}\n")


def render_html(v: AnyVerdict, inputs=None) -> str:
    v2 = isinstance(v, VerdictV2)
    title = f"Courant · {v.run.id}"
    out = ["<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{escape(title)}</title>", f"<style>{CSS_V2 if v2 else CSS}</style>", "</head>",
           "<body>", "<main>"]
    out.append(f"<h1>Verification: {escape(v.run.id)}</h1>")
    out.append(f'<div class="meta">{escape(v.run.solver or "solver unknown")} · '
               f'{escape(v.tool.name)} {escape(v.tool.version)} · schema '
               f'<span class="mono">{escape(v.schema_)}</span></div>')
    out.append(f'<div class="card"><span class="badge {v.verdict}">{v.verdict.replace("_", " ")}</span>'
               f'rule <span class="mono">{escape(v.rule)}</span> · grade '
               f'<b>{escape(v.grade or "—")}</b>'
               + (f' · basis <span class="mono">{escape(v.basis)}</span>' if v2 else "")
               + "</div>")
    accepted = v.verdict in ("TRUSTWORTHY", "accept")
    head = "Notes" if accepted else "Why"
    out.append(f"<h2>{head}</h2>")
    if v.reasons:
        rows = [f'<tr><td class="mono">{escape(r.code)}</td><td>{escape(r.severity)}</td>'
                f"<td>{escape(r.summary)}</td><td class=\"mono\">{escape(r.input)}</td>"
                f"<td>{escape(r.method)}</td><td class=\"num\">{escape(r.value or '—')}</td>"
                f"<td class=\"num\">{escape(r.threshold or '—')}</td></tr>" for r in v.reasons]
        out.append(_table(f"{head}: the reasons the checks raised",
                          ("Code", "Severity", "Finding", "Read", "Method", "Value", "Threshold"),
                          rows))
    else:
        out.append("<p>No reason was raised.</p>")
    if v.probable_causes and not accepted:
        out.append("<h2>Probable causes</h2><ol>")
        out += [f'<li><span class="mono">{escape(c.code)}</span> — {escape(c.name)}</li>'
                for c in v.probable_causes]
        out.append("</ol>")
    if v.next_action:
        out.append(f"<h2>Next action</h2><p>{escape(v.next_action)}</p>")

    res, co, mon, stop, binding = ({}, {}, {}, None, None)
    if inputs is not None:
        res, co, mon, stop, binding = _series_from_inputs(inputs)
    c = v.convergence
    if c is not None and c.monitors_found:
        out.append("<h2>Convergence</h2>")
        line = f"<b>{escape(c.status)}</b> · {escape(c.regime or 'regime unknown')}"
        if c.converged_at_step is not None:
            line = line + (f" · the answer stopped changing at step {c.converged_at_step:,} "
                           f"(<span class=\"mono\">{escape(c.binding_monitor or '')}</span>)")
        out.append(f'<div class="card">{line}')
        if c.wasted_steps is not None:
            pct = f" ({100 * float(c.wasted_fraction):.1f}%)" if c.wasted_fraction else ""
            out.append(f"<br>{c.wasted_steps:,} of {c.steps_total:,} steps{pct} came after it"
                       + (f"; {escape(c.wasted_wall_clock_s)} s wall clock" if c.wasted_wall_clock_s else "")
                       + (f"; {escape(c.wasted_core_hours)} core-hours" if c.wasted_core_hours else ""))
        if c.messages:
            out.append("<ul>" + "".join(f"<li>{escape(m)}</li>" for m in c.messages) + "</ul>")
        out.append('<div class="cap">Reported beside the verdict, not in it: policy rule VER-5.4b '
                   "is off.</div></div>")
        if mon:
            out.append(_chart(mon, log_y=False, title="The binding monitor against step",
                              cid="chart-monitor", marker=float(stop) if stop else None,
                              marker_label="stopped changing" if stop else "", skip_start=0.02))
            out.append('<div class="cap">The binding monitor, drawn from its postProcessing file. '
                       "Shaded: steps after the answer stopped changing. The vertical range "
                       "leaves out the first 2% of the run.</div>")
    k = v.conformance
    if k is not None:
        conforming = sum(1 for i in k.items if i.conforms)
        out.append("<h2>Conformance to the task specification</h2>")
        out.append(f'<div class="card">{conforming} of {len(k.items)} specified values match the '
                   f'case · spec <span class="mono">{escape(k.spec_sha256[:12])}</span>'
                   + (f' (locked <span class="mono">{escape(k.locked_sha256[:12])}</span>)'
                      if k.locked_sha256 else "")
                   + (f"<br>{escape(k.task)}" if k.task else "") + "</div>")
        rows = []
        for i in k.items:
            result = ("matches" if i.conforms else "<b>does not match</b>" if i.conforms is False
                      else f"not checked: {escape(i.note or '')}")
            rows.append(f"<tr><td>{escape(i.name)}</td><td class=\"num\">{escape(i.specified or '—')}</td>"
                        f"<td class=\"num\">{escape(i.found or '—')}</td><td>{escape(i.tolerance)}</td>"
                        f"<td class=\"mono\">{escape(i.file)}</td><td>{result}</td></tr>")
        out.append(_table("Specified values against the case",
                          ("Value", "Specified", "In the case", "Tolerance", "Read", "Result"), rows))
    if res:
        out.append("<h2>Residuals</h2>")
        out.append(_chart(res, log_y=True, title="Initial residuals against step",
                          cid="chart-residuals"))
        out.append('<div class="cap">Initial residual per field and step, drawn from the solver log '
                   "(log scale).</div>")
    if co:
        out.append("<h2>Courant number</h2>")
        out.append(_chart(co, log_y=False, title="Maximum Courant number against step",
                          cid="chart-courant"))
        out.append('<div class="cap">Maximum Courant number per step, drawn from the solver log.</div>')
    if v.not_assessed:
        out.append("<h2>Not checked</h2><ul>")
        out += [f"<li>{escape(s)}</li>" for s in v.not_assessed]
        out.append("</ul>")
    out.append("<h2>Inputs</h2>")
    out.append(_table("Input files with their sizes and SHA-256", ("File", "Bytes", "SHA-256"),
                      [f'<tr><td class="mono">{escape(f.path)}</td><td class="num">{f.bytes:,}</td>'
                       f'<td class="mono">{escape(f.sha256)}</td></tr>' for f in v.run.inputs]))
    s = v.stats
    out.append("</main>")
    out.append(f"<footer>Parsed {s.events:,} events and {s.residual_steps:,} residual steps. "
               "Every number in the text above is in this run's verdict JSON, apart from the error "
               "rates at the end of this line; the charts are drawn "
               "from the input files listed, whose SHA-256 is shown. Courant is read-only: it did "
               "not edit the case or run a solver. Generated by "
               f"{escape(v.tool.name)} {escape(v.tool.version)}. "
               f"{escape(about.footer())}</footer>")
    out += ["</body>", "</html>", ""]
    return "\n".join(out)
