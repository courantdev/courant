"""HTML reports meet WCAG 2.2 AA for contrast and structure, on generated output (TRD REP-8).

The checks read the pages Courant writes, not a model of them: the colour pairs come from the CSS
inside the page, in both themes; the structure comes from parsing the markup. A checker that
cannot fail is worth nothing, so the second half feeds it pages with known faults.

What is checked, and the success criterion each stands for:
  * contrast of at least 4.5:1 for every text and background pair, light and dark (1.4.3), and
    3:1 for the graphics that carry data and for the focus ring (1.4.11, 2.4.13);
  * `lang` set (3.1.1), a title (2.4.2), a viewport that allows zoom (1.4.4);
  * one `main` and a `footer` outside it, and no text outside a landmark (1.3.1, 2.4.1);
  * one h1 and heading levels that never skip down (1.3.1, 2.4.6);
  * a caption and column headers with scope on every table, rows as wide as the header (1.3.1);
  * a named, focusable scrolling region around every table (2.1.1, 4.1.2);
  * a title and a description on every chart, and a dash pattern beside each series' colour
    (1.1.1, 1.4.1);
  * no colour set inline, no script, no handlers, no positive tabindex, unique ids.
"""

from __future__ import annotations

import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

from courant.verify import from_path, verify_inputs, verify_path
from courant.verify import html as report_html

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
MON = ROOT / "certification" / "monitored"
CASE = MON / "simpleFoam_pitzDaily__fixed_budget"

AA_TEXT = 4.5
AA_GRAPHIC = 3.0


# --------------------------------------------------------------------------- contrast
def _hex(c: str) -> tuple[float, float, float]:
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))      # type: ignore[return-value]


def _rgba(c: str) -> tuple[float, float, float, float]:
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)",
                     c.strip())
    if m:
        r, g, b = (float(m.group(i)) / 255 for i in (1, 2, 3))
        return r, g, b, float(m.group(4) or 1)
    return (*_hex(c), 1.0)


def luminance(rgb: tuple[float, float, float]) -> float:
    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def over(top: tuple[float, float, float, float], under: tuple[float, float, float]):
    r, g, b, a = top
    return tuple(a * t + (1 - a) * u for t, u in zip((r, g, b), under, strict=True))


# --------------------------------------------------------------------------- the CSS
def _blocks(css: str, media: str | None = None) -> list[tuple[str | None, str, dict[str, str]]]:
    out, i = [], 0
    while True:
        j = css.find("{", i)
        if j < 0:
            return out
        sel, depth, k = css[i:j].strip(), 1, j + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        body = css[j + 1:k - 1]
        if sel.startswith("@media"):
            out += _blocks(body, sel)
        else:
            decls = {}
            for part in body.split(";"):
                if ":" in part:
                    prop, _, val = part.partition(":")
                    decls[prop.strip()] = val.strip()
            for one in sel.split(","):
                out.append((media, one.strip(), decls))
        i = k


def page_css(page: str) -> str:
    return "".join(re.findall(r"<style>(.*?)</style>", page, flags=re.DOTALL))


def themes(css: str) -> dict[str, dict[str, str]]:
    rules = _blocks(css)
    light = {k: v for m, s, d in rules if s == ":root" and m is None for k, v in d.items()}
    dark = dict(light)
    for m, s, d in rules:
        if s == ":root" and m and "prefers-color-scheme:dark" in m.replace(" ", ""):
            dark.update(d)
    return {"light": light, "dark": dark}


def resolve(value: str, tokens: dict[str, str]) -> str:
    while "var(" in value:
        value = re.sub(r"var\((--[\w-]+)\)", lambda m: tokens[m.group(1)], value)
    return value


#: Where each rule's text sits. Every text colour in the CSS must be listed, so a new colour
#: cannot slip in without its surfaces being named and measured.
TEXT_SURFACES = {
    "body": ("--bg", "--card"),
    ".meta": ("--bg", "--card"),
    ".cap": ("--bg", "--card"),
    "th": ("--bg", "--card"),
    ".legend span": ("--bg",),
    "footer": ("--bg",),
    "svg text": ("--bg", "shaded"),
    ".badge": ("--ok", "--bad", "--warn"),
}
#: Every rule that paints a background, and the token it paints.
BACKGROUNDS = {"body": "--bg", ".card": "--card", ".TRUSTWORTHY": "--ok",
               ".NOT_TRUSTWORTHY": "--bad", ".NOT_ASSESSABLE": "--warn",
               # courant.verdict/2's badges paint the same tokens as v1's (verify/html.py CSS_V2)
               ".accept": "--ok", ".reject": "--bad", ".insufficient": "--warn"}


def surface(name: str, tokens: dict[str, str]) -> tuple[float, float, float]:
    base = _hex(resolve("var(--bg)", tokens))
    if name == "shaded":                       # the chart's shaded region, over the page
        return over(_rgba(resolve("var(--shade)", tokens)), base)
    return _hex(resolve(f"var({name})", tokens))


def text_colour_failures(css: str) -> list[str]:
    bad = []
    rules = _blocks(css)
    declared = {}
    for media, sel, d in rules:
        if media:
            continue
        if "color" in d:
            declared[sel] = d["color"]
        if "fill" in d and sel.startswith("svg"):
            declared[sel] = d["fill"]
    for sel in declared:
        if sel not in TEXT_SURFACES:
            bad.append(f"text colour on '{sel}' has no named surface in the test")
    painted = {sel for media, sel, d in rules if not media and ("background" in d
                                                               or "background-color" in d)}
    for sel in painted - set(BACKGROUNDS) - {".legend i"}:
        bad.append(f"background on '{sel}' is not in the test's table")
    for theme, tokens in themes(css).items():
        for sel, colour in declared.items():
            fg = _hex(resolve(colour, tokens))
            for s in TEXT_SURFACES.get(sel, ()):
                ratio = contrast(fg, surface(s, tokens))
                if ratio < AA_TEXT:
                    bad.append(f"{theme}: '{sel}' text on {s} is {ratio:.2f}:1, needs {AA_TEXT}:1")
    return bad


def graphic_failures(css: str) -> list[str]:
    bad = []
    for theme, tokens in themes(css).items():
        for k in range(1, 7):
            for s in ("--bg", "--card", "shaded"):
                ratio = contrast(_hex(tokens[f"--s{k}"]), surface(s, tokens))
                if ratio < AA_GRAPHIC:
                    bad.append(f"{theme}: series --s{k} on {s} is {ratio:.2f}:1, needs {AA_GRAPHIC}:1")
        for what, tok in (("axis", "--muted"), ("focus ring", "--s1")):
            for s in ("--bg", "--card"):
                ratio = contrast(_hex(tokens[tok]), surface(s, tokens))
                if ratio < AA_GRAPHIC:
                    bad.append(f"{theme}: {what} on {s} is {ratio:.2f}:1, needs {AA_GRAPHIC}:1")
    rules = _blocks(css)
    axis = [d for m, s, d in rules if s == ".axis" and m is None]
    if not axis or axis[0].get("stroke") != "var(--muted)":
        bad.append("the chart axes must be drawn in --muted, not a hairline colour")
    if not any(s == ":focus-visible" and "outline" in d for m, s, d in rules):
        bad.append("no :focus-visible style (2.4.7)")
    return bad


# --------------------------------------------------------------------------- the markup
class Node:
    def __init__(self, tag: str, attrs: dict, parent: Node | None):
        self.tag, self.attrs, self.parent = tag, attrs, parent
        self.children: list[Node | str] = []

    def walk(self):
        yield self
        for c in self.children:
            if isinstance(c, Node):
                yield from c.walk()

    def text(self) -> str:
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)

    def ancestors(self):
        n = self.parent
        while n is not None:
            yield n
            n = n.parent


VOID = {"meta", "link", "br", "img", "input", "hr", "col"}


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {}, None)
        self.cur = self.root
        self.doctype = ""

    def handle_decl(self, decl):
        self.doctype = decl

    def handle_starttag(self, tag, attrs):
        n = Node(tag, dict(attrs), self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, dict(attrs), self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def parse(page: str) -> tuple[Node, str]:
    b = _Builder()
    b.feed(page)
    return b.root, b.doctype


def structure_failures(page: str) -> list[str]:
    bad: list[str] = []
    root, doctype = parse(page)
    nodes = list(root.walk())
    def by(tag: str) -> list[Node]:
        return [n for n in nodes if n.tag == tag]

    if doctype.lower() != "doctype html":
        bad.append("no <!doctype html>: the page would render in quirks mode")
    html = by("html")
    if not html or not html[0].attrs.get("lang"):
        bad.append("<html> has no lang (3.1.1)")
    titles = by("title")
    if not titles or not titles[0].text().strip():
        bad.append("no <title> (2.4.2)")
    for m in by("meta"):
        if m.attrs.get("name") == "viewport":
            c = m.attrs.get("content", "").replace(" ", "")
            if "user-scalable=no" in c or "maximum-scale=1" in c:
                bad.append("the viewport forbids zoom (1.4.4)")
    # landmarks
    mains, foots = by("main"), by("footer")
    if len(mains) != 1:
        bad.append(f"{len(mains)} <main> elements, needs exactly 1")
    if len(foots) != 1:
        bad.append(f"{len(foots)} <footer> elements, needs exactly 1")
    for f in foots:
        if any(a.tag == "main" for a in f.ancestors()):
            bad.append("<footer> is inside <main>, so it is not a contentinfo landmark")
    for body in by("body"):
        for n in body.walk():
            for c in n.children:
                if isinstance(c, str) and c.strip():
                    chain = {a.tag for a in n.ancestors()} | {n.tag}
                    if not chain & {"main", "footer"}:
                        bad.append(f"text outside any landmark: {c.strip()[:40]!r}")
    # headings
    heads = [n for n in nodes if re.fullmatch(r"h[1-6]", n.tag)]
    if [h.tag for h in heads].count("h1") != 1:
        bad.append("needs exactly one <h1>")
    if heads and heads[0].tag != "h1":
        bad.append("the first heading is not the <h1>")
    prev = 0
    for h in heads:
        level = int(h.tag[1])
        if level > prev + 1:
            bad.append(f"heading level jumps from h{prev} to h{level}")
        if not h.text().strip():
            bad.append(f"empty <{h.tag}>")
        prev = level
    # tables
    for t in by("table"):
        cap = [c for c in t.children if isinstance(c, Node) and c.tag == "caption"]
        if not cap or not cap[0].text().strip():
            bad.append("a table has no caption")
        ths = [n for n in t.walk() if n.tag == "th"]
        if not ths:
            bad.append("a table has no header cells")
        for th in ths:
            if th.attrs.get("scope") not in ("col", "row", "colgroup", "rowgroup"):
                bad.append(f"header cell {th.text()!r} has no scope")
            if not th.text().strip():
                bad.append("an empty header cell")
        head_rows = [n for n in t.walk() if n.tag == "tr" and any(
            isinstance(c, Node) and c.tag == "th" for c in n.children)]
        body_rows = [n for n in t.walk() if n.tag == "tr" and n not in head_rows]
        width = len(ths)
        for r in body_rows:
            cells = [c for c in r.children if isinstance(c, Node) and c.tag in ("td", "th")]
            if len(cells) != width:
                bad.append(f"a table row has {len(cells)} cells under {width} headers")
        region = next((a for a in t.ancestors() if a.attrs.get("role") == "region"), None)
        if region is None:
            bad.append("a table is not inside a named scrolling region")
        else:
            if region.attrs.get("tabindex") != "0":
                bad.append("a scrolling region cannot be reached by keyboard (2.1.1)")
            if not region.attrs.get("aria-label", "").strip():
                bad.append("a scrolling region has no accessible name (4.1.2)")
    # charts and images
    ids = {n.attrs["id"]: n for n in nodes if "id" in n.attrs}
    for svg in by("svg"):
        if svg.attrs.get("role") != "img":
            bad.append("an <svg> chart is not role=img")
        refs = svg.attrs.get("aria-labelledby", "").split()
        if len(refs) != 2:
            bad.append("an <svg> chart needs aria-labelledby with a title and a description")
        for r in refs:
            if r not in ids or not ids[r].text().strip():
                bad.append(f"chart label {r!r} is missing or empty")
        tags = {ids[r].tag for r in refs if r in ids}
        if refs and tags != {"title", "desc"}:
            bad.append("a chart is labelled by something other than its <title> and <desc>")
        lines = [n for n in svg.walk() if n.tag == "polyline"]
        dashes = [n.attrs.get("stroke-dasharray", "") for n in lines]
        if len(set(dashes)) < min(len(lines), len(report_html.DASHES)):
            bad.append("chart series are told apart by colour alone (1.4.1)")
    for img in by("img"):
        if "alt" not in img.attrs:
            bad.append("an <img> has no alt (1.1.1)")
    # the rest
    seen: set[str] = set()
    for n in nodes:
        if "id" in n.attrs:
            if n.attrs["id"] in seen:
                bad.append(f"duplicate id {n.attrs['id']!r}")
            seen.add(n.attrs["id"])
        if n.attrs.get("tabindex", "0").lstrip("-").isdigit() and int(n.attrs.get("tabindex", "0")) > 0:
            bad.append("a positive tabindex")
        if any(k.startswith("on") for k in n.attrs):
            bad.append(f"an event handler on <{n.tag}>")
        style = n.attrs.get("style", "")
        if re.search(r"(^|;)\s*(color|fill|stroke)\s*:", style):
            bad.append("a colour set inline escapes the contrast check")
        if n.tag == "i" and n.attrs.get("aria-hidden") != "true":
            bad.append("a decorative swatch is not aria-hidden")
    if by("script"):
        bad.append("a <script> in a report")
    badges = [n for n in nodes if "badge" in n.attrs.get("class", "").split()]
    for b in badges:
        if not b.text().strip():
            bad.append("the verdict badge has no text: colour alone would carry the verdict")
    return bad


# --------------------------------------------------------------------------- generated pages
@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    from courant.data import extract_demo_case
    from courant.spec import draft_spec
    from courant.verify.audit import audit
    from courant.verify.audit import render_html as render_audit

    out = {}
    inputs = from_path(CASE)
    out["monitored run, charts and convergence"] = report_html.render_html(
        verify_inputs(inputs), inputs)
    log = CORPUS / "icoFoam_cavity__cfl_blowup.log.gz"
    inputs = from_path(log)
    out["log-only run that fails"] = report_html.render_html(verify_inputs(inputs), inputs)
    healthy = CORPUS / "icoFoam_cavity__healthy.log.gz"
    inputs = from_path(healthy)
    out["healthy log-only run"] = report_html.render_html(verify_inputs(inputs), inputs)

    tmp = tmp_path_factory.mktemp("a11y")
    demo = extract_demo_case(tmp)
    spec = tmp / "spec.json"
    spec.write_text(draft_spec(demo).to_json())
    tp = demo / "constant/transportProperties"
    tp.write_text(tp.read_text().replace("1e-05", "3e-05"))
    out["run with a task specification that does not match"] = report_html.render_html(
        verify_path(demo, spec=spec))

    runs = tmp / "runs"
    runs.mkdir()
    for name in ("icoFoam_cavity__healthy", "icoFoam_cavity__transport_units"):
        for ext in (".log.gz", ".case.json"):
            shutil.copy(CORPUS / f"{name}{ext}", runs / f"{name}{ext}")
    out["compute audit"] = render_audit(audit(runs, cost_per_core_hour=0.05))
    return out


@pytest.fixture(params=["monitored run, charts and convergence", "log-only run that fails",
                        "healthy log-only run",
                        "run with a task specification that does not match", "compute audit"])
def page(request, pages):
    return pages[request.param]


class TestGeneratedReports:
    def test_structure(self, page):
        assert structure_failures(page) == []

    def test_text_contrast_in_both_themes(self, page):
        assert text_colour_failures(page_css(page)) == []

    def test_graphics_contrast_and_focus(self, page):
        assert graphic_failures(page_css(page)) == []

    def test_both_themes_define_every_colour(self, page):
        t = themes(page_css(page))
        assert set(t["light"]) == set(t["dark"])
        assert {"--bg", "--fg", "--muted", "--card", "--ok", "--bad", "--warn", "--on-fill"} <= set(
            t["dark"])
        assert t["dark"]["--bg"] != t["light"]["--bg"]

    def test_the_dark_theme_is_a_media_query_not_a_script(self, page):
        assert "@media (prefers-color-scheme:dark)" in page and "<script" not in page.lower()

    def test_the_page_still_has_no_network_reference(self, page):
        assert not re.search(r"(src|href)\s*=\s*['\"]?https?://", page)


class TestWhatTheReportsContain:
    def test_the_full_report_has_two_charts_and_every_table_labelled(self, pages):
        p = pages["monitored run, charts and convergence"]
        root, _ = parse(p)
        nodes = list(root.walk())
        assert len([n for n in nodes if n.tag == "svg"]) == 2
        assert len([n for n in nodes if n.tag == "table"]) == 2          # reasons, inputs
        titles = [n.text() for n in nodes if n.tag == "title" and n.parent.tag == "svg"]
        assert len(titles) == 2 and all(len(t) > 10 for t in titles)
        descs = [n.text() for n in nodes if n.tag == "desc"]
        assert all("Line chart of" in d and "against step" in d for d in descs)

    def test_a_chart_with_several_series_uses_several_dash_patterns(self, pages):
        p = pages["log-only run that fails"]
        root, _ = parse(p)
        charts = [n for n in root.walk() if n.tag == "svg"]
        assert charts
        for svg in charts:
            lines = [n for n in svg.walk() if n.tag == "polyline"]
            assert len({n.attrs.get("stroke-dasharray", "") for n in lines}) == min(
                len(lines), len(report_html.DASHES))
        assert max(len([n for n in s.walk() if n.tag == "polyline"]) for s in charts) >= 3

    def test_the_legend_swatch_carries_the_same_pattern(self, pages):
        p = pages["log-only run that fails"]
        assert p.count("repeating-linear-gradient") >= 2           # dashed series get a dashed swatch

    def test_the_conformance_table_is_a_labelled_table(self, pages):
        p = pages["run with a task specification that does not match"]
        assert "Specified values against the case" in p and "does not match" in p

    def test_the_dash_patterns_are_distinct_and_start_solid(self):
        assert len(set(report_html.DASHES)) == len(report_html.DASHES) >= len(report_html.PALETTE)
        assert report_html.DASHES[0] == ()

    def test_the_audit_page_has_the_same_structure(self, pages):
        p = pages["compute audit"]
        assert "<caption" in p and 'scope="col"' in p and "</main><footer>" in p

    def test_pages_are_still_deterministic(self):
        inputs = from_path(CASE)
        a = report_html.render_html(verify_inputs(inputs), inputs)
        b = report_html.render_html(verify_inputs(inputs), inputs)
        assert a == b


# --------------------------------------------------------------------------- the checker can fail
GOOD = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><title>t</title></head><body>"
        "<main><h1>One</h1><h2>Two</h2></main><footer>f</footer></body></html>")


class TestTheCheckersCanFail:
    def test_a_good_minimal_page_passes(self):
        assert structure_failures(GOOD) == []

    @pytest.mark.parametrize("page,needle", [
        (GOOD.replace(" lang='en'", ""), "lang"),
        (GOOD.replace("<title>t</title>", ""), "<title>"),
        (GOOD.replace("<h2>Two</h2>", "<h3>Two</h3>"), "jumps from h1 to h3"),
        (GOOD.replace("<h1>One</h1>", "<h2>One</h2>"), "first heading"),
        (GOOD.replace("<h2>Two</h2>", "<h1>Two</h1>"), "exactly one <h1>"),
        (GOOD.replace("<footer>f</footer>", "<p>f</p>"), "<footer>"),
        (GOOD.replace("</main><footer>f</footer>", "<footer>f</footer></main>"), "inside <main>"),
        (GOOD.replace("</main>", "</main><p>stray</p>"), "outside any landmark"),
        (GOOD.replace("<head>", "<head><meta name='viewport' content='user-scalable=no'>"), "zoom"),
        (GOOD.replace("<h2>", "<img src='x.png'><h2>"), "alt"),
        (GOOD.replace("<h1>", "<h1 onclick='x()'>"), "event handler"),
        (GOOD.replace("<h1>", "<h1 style='color:red'>"), "inline"),
        (GOOD.replace("<h1>", "<script></script><h1>"), "<script>"),
        (GOOD.replace("<h1>One</h1>", "<h1>One</h1><div tabindex='3'></div>"), "tabindex"),
        (GOOD.replace("<h1>One</h1>", "<h1>One</h1><p id='a'></p><p id='a'></p>"), "duplicate id"),
        (GOOD.replace("<!doctype html>", ""), "quirks"),
    ])
    def test_a_structural_fault_is_found(self, page, needle):
        assert any(needle in f for f in structure_failures(page)), structure_failures(page)

    def test_a_table_without_headers_scope_caption_or_region_is_found(self):
        bad = GOOD.replace("<h2>Two</h2>", "<h2>Two</h2><table><tr><th>A</th><th>B</th></tr>"
                           "<tr><td>1</td></tr></table>")
        found = " | ".join(structure_failures(bad))
        for needle in ("no caption", "has no scope", "1 cells under 2 headers", "scrolling region"):
            assert needle in found

    def test_a_scrolling_region_must_be_focusable_and_named(self):
        table = ("<table><caption>c</caption><tr><th scope='col'>A</th></tr><tr><td>1</td></tr>"
                 "</table>")
        no_tab = GOOD.replace("<h2>Two</h2>", f"<div role='region' aria-label='x'>{table}</div>")
        no_name = GOOD.replace("<h2>Two</h2>", f"<div role='region' tabindex='0'>{table}</div>")
        assert any("keyboard" in f for f in structure_failures(no_tab))
        assert any("accessible name" in f for f in structure_failures(no_name))

    def test_a_chart_without_a_title_or_with_one_dash_is_found(self):
        svg = ("<svg role='img'><polyline stroke-dasharray=''/><polyline stroke-dasharray=''/>"
               "</svg>")
        found = " | ".join(structure_failures(GOOD.replace("<h2>Two</h2>", f"<h2>Two</h2>{svg}")))
        assert "aria-labelledby" in found and "colour alone" in found

    def test_contrast_arithmetic(self):
        assert contrast((0, 0, 0), (1, 1, 1)) == pytest.approx(21.0)
        assert contrast(_hex("#777777"), (1, 1, 1)) < 4.5            # the classic near miss
        assert contrast(_hex("#767676"), (1, 1, 1)) >= 4.5

    def test_the_old_dark_badge_would_have_failed(self):
        # White text on the dark theme's light green fill, the pair this node fixed.
        assert contrast((1, 1, 1), _hex("#4cc38a")) < AA_TEXT

    def test_a_low_contrast_colour_is_reported(self):
        css = (":root{--bg:#ffffff;--card:#ffffff;--fg:#777777;--muted:#777777;--ok:#000;--bad:#000;"
               "--warn:#000;--on-fill:#fff;--shade:rgba(0,0,0,.1)}body{background:var(--bg);"
               "color:var(--fg)}")
        found = text_colour_failures(css)
        assert any("'body' text on --bg" in f for f in found)

    def test_a_new_text_colour_must_be_listed(self):
        css = (":root{--bg:#fff;--card:#fff;--fg:#000;--muted:#000;--ok:#000;--bad:#000;--warn:#000;"
               "--on-fill:#fff;--shade:rgba(0,0,0,.1)}.new{color:var(--fg)}")
        assert any("'.new'" in f for f in text_colour_failures(css))

    def test_a_new_background_must_be_listed(self):
        css = (":root{--bg:#fff;--card:#fff;--fg:#000;--muted:#000;--ok:#000;--bad:#000;--warn:#000;"
               "--on-fill:#fff;--shade:rgba(0,0,0,.1)}.fresh{background:var(--bg)}")
        assert any("'.fresh'" in f for f in text_colour_failures(css))

    def test_hairline_axes_are_reported(self):
        css = (":root{--bg:#fff;--card:#fff;--fg:#000;--muted:#555;--ok:#000;--bad:#000;--warn:#000;"
               "--on-fill:#fff;--shade:rgba(0,0,0,.1);--line:#e3e3e0;--s1:#2f6fb3;--s2:#c4472f;"
               "--s3:#2f8f5b;--s4:#8a5cb5;--s5:#a87a0a;--s6:#4a4a4a}.axis{stroke:var(--line)}")
        found = graphic_failures(css)
        assert any("axes" in f for f in found) and any("focus-visible" in f for f in found)
