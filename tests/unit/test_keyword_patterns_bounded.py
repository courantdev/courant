"""A keyword pattern the case writes cannot stall the gate (TRD SPEC-2, VER-3b, VER-16, MCP-5).

A quoted keyword in an OpenFOAM dictionary is a regular expression the case writes; the readers
match it against a name the case, the log or a specification gives. Matched with Python's `re`,
a backtracking engine, the case decided how long the match took, and `re` holds the
interpreter's lock while it matches, so not even the MCP server's wall-clock limit (MCP-5) could
answer the call. The final review of the 0.6.0 integration (21 Oct 2026) found three readers on
main doing so:

- the case reader (`_foam_reader._matches`, on the DEFAULT path of `courant verify <case>`,
  `courant preflight` and the MCP tools): `solvers { "(a+)+b" { ... } p { solver PCG; $aaa...; } }`
  hung with no `--spec` and no `--checks next`;
- the specification reader (`deck.block`): a pattern in `0/U`'s `boundaryField` stalled
  `courant verify --spec`;
- the residualControl lookup (`convergence._matches`, under `--checks next`): a pattern before a
  steady run's own entries stalled the bar of a field with a seven-letter name.

Each is run here in a fresh interpreter with a time limit, so a regression fails, not hangs.
Every case is this repository's own (`tests/fixtures/real_openfoam/cavity_extreme_aspect_ratio`,
`certification/monitored/simpleFoam_pitzDaily__shipped`) with synthetic keywords added.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"
PITZ = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__shipped"

#: Exponential in `re` on a name of a few dozen characters it does not match.
SOLVER_KEY = '"(a+)+b"'
MACRO = "$" + "a" * 40
#: Exponential in `re` on a name of seven characters (`epsilon`), and on `movingWall`.
NESTED = '"((((.*)*)*)*)*!"'


def _bounded(code: str, seconds: float = 60) -> str:
    """Run `code` in a fresh interpreter and fail, not hang, past `seconds`."""
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                           text=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        pytest.fail(f"did not return within {seconds} s: a case-written keyword stalled the gate")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _case(tmp_path: Path, name: str, source: Path = CAVITY) -> Path:
    case = tmp_path / name
    shutil.copytree(source, case)
    for gz in case.glob("log.*.gz"):
        with gzip.open(gz, "rb") as src, open(case / gz.name.removesuffix(".gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    return case


def _hostile_solvers(case: Path) -> None:
    """The final review's fvSolution: a pattern among the solvers, and a `$name;` after `p`'s
    `solver` that the reader asks whether a solver entry names (`_Dictionary._inherited`)."""
    fv = case / "system" / "fvSolution"
    text = fv.read_text()
    head = "solvers\n{\n    p\n    {\n        solver          PCG;\n"
    assert head in text
    fv.write_text(text.replace(head, (
        f"solvers\n{{\n    {SOLVER_KEY}\n    {{\n        solver PCG;\n    }}\n"
        f"    p\n    {{\n        solver          PCG;\n        {MACRO};\n")))


def test_a_solver_keyword_pattern_does_not_stall_verify_or_preflight_by_default(tmp_path):
    case = _case(tmp_path, "agent")
    _hostile_solvers(case)
    out = _bounded(
        "from courant.verify import verify_path\n"
        "from courant.verify.preflight import preflight_path\n"
        f"v = verify_path({str(case)!r})\n"
        f"p = preflight_path({str(case)!r})\n"
        "print(v.verdict, p.verdict)\n"
        "print(any('solvers/p' in s for s in v.not_assessed))", seconds=60)
    verdicts, named = out.splitlines()
    assert verdicts.split()[0] in ("accept", "reject", "insufficient")
    # the `$aaa...;` is a directive no solver entry is known to name: not read, and said so
    assert named == "True"


def test_the_mcp_tool_answers_on_a_solver_keyword_pattern(tmp_path):
    """MCP-5's limit cannot answer a call stuck inside `re` (it holds the interpreter's lock);
    the match is bounded instead, so the tool answers on its own."""
    case = _case(tmp_path, "agent")
    _hostile_solvers(case)
    out = _bounded(
        "from courant.mcp import tools\n"
        "from courant.mcp.server import within\n"
        f"confine = tools.Confinement([{str(tmp_path)!r}])\n"
        f"got = within(600, lambda: tools.verify_run({str(case)!r}, confine=confine))\n"
        "print(got['verdict'])", seconds=60)
    assert out in ("accept", "reject", "insufficient")


def test_a_boundary_keyword_pattern_does_not_stall_verify_with_a_spec(tmp_path):
    """`"(movingWall|lid)"` groups the lid, and a pattern `re` backtracks on comes after it: the
    spec's `U at movingWall` is found through the group, at once, and conforms."""
    from typer.testing import CliRunner

    from courant.cli.main import app

    ref = _case(tmp_path, "reference")
    spec = tmp_path / "spec.json"
    assert CliRunner().invoke(app, ["spec", "init", str(ref), "-o", str(spec)]).exit_code == 0
    agent = _case(tmp_path, "agent")
    u = agent / "0" / "U"
    text = u.read_text().replace("    movingWall\n", '    "(movingWall|lid)"\n', 1)
    end = text.rindex("}")                               # boundaryField's own, the file's last
    u.write_text(text[:end] + f"    {NESTED}\n    {{\n        type zeroGradient;\n    }}\n"
                 + text[end:])
    out = _bounded(
        "import json\n"
        "from courant.verify import from_path, verify_inputs\n"
        "from courant.verify.ingest import with_spec\n"
        f"v = json.loads(verify_inputs(with_spec(from_path({str(agent)!r}), {str(spec)!r}))"
        ".to_json())\n"
        "items = {i['name']: i for i in v['conformance']['items']}\n"
        "print(items['U at movingWall'].get('conforms'))", seconds=60)
    assert out == "True"


def test_a_residual_control_keyword_pattern_does_not_stall_verify_under_next(tmp_path):
    """pitzDaily's residualControl with a pattern before its own entries: every field's
    tolerance is looked up through it, `epsilon` among them. Evaluated, it names no field, so
    the verdict is the shipped case's."""
    case = _case(tmp_path, "pitz", PITZ)
    fv = case / "system" / "fvSolution"
    text = fv.read_text()
    anchor = "    residualControl\n    {\n"
    assert anchor in text
    fv.write_text(text.replace(anchor, anchor + f"        {NESTED} 1e-2;\n", 1))
    out = _bounded(
        "from courant.verify import verify_path\n"
        f"a = verify_path({str(case)!r}, checks='next')\n"
        f"b = verify_path({str(PITZ)!r}, checks='next')\n"
        "print(a.verdict, a.rule, a.verdict == b.verdict and a.rule == b.rule)", seconds=90)
    assert out.split()[-1] == "True"
