"""Evidence pack, re-verification and batch (TRD REP-4, REP-5, VER-12, VER-14)."""

from __future__ import annotations

import ast
import contextlib
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.verify import verify_path
from courant.verify.batch import discover, summarise, verify_all
from courant.verify.pack import MANIFEST, REPORT, VERDICT, check_pack, read_manifest, write_pack

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "courant"
CORPUS = ROOT / "certification" / "corpus"
MONITORED = ROOT / "certification" / "monitored"
REAL = ROOT / "tests" / "fixtures" / "real_openfoam"
MINI = ROOT / "tests" / "fixtures" / "mini_case"


def _copy(tmp: Path, sid: str) -> tuple[Path, Path]:
    log, cj = tmp / f"{sid}.log.gz", tmp / f"{sid}.case.json"
    shutil.copy(CORPUS / log.name, log)
    shutil.copy(CORPUS / cj.name, cj)
    return log, cj


def test_a_pack_has_three_files_and_names_every_input(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__transport_units")
    out = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    assert sorted(p.name for p in out.iterdir()) == sorted([VERDICT, REPORT, MANIFEST])
    entries, meta = read_manifest(out)
    assert {e[2] for e in entries} == {log.name, cj.name}
    assert meta["source"] == "corpus" and meta["target"] == log.name


def test_a_pack_is_byte_identical_when_written_twice(tmp_path):
    log, cj = _copy(tmp_path, "simpleFoam_pitzDaily__unbounded_scheme")
    a = write_pack(verify_path(log, cj), tmp_path / "a", log.name, cj.name)
    b = write_pack(verify_path(log, cj), tmp_path / "b", log.name, cj.name)
    for name in (VERDICT, REPORT, MANIFEST):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_an_untouched_pack_re_verifies_identically(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__cfl_blowup")
    pack = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    ok, msg = check_pack(pack, tmp_path)
    assert ok, msg


def test_a_changed_input_is_named(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    pack = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    with gzip.open(log, "rt") as fh:
        text = fh.read()
    with gzip.open(log, "wt") as fh:
        fh.write(text.replace("Courant Number mean: 0", "Courant Number mean: 9", 1))
    ok, msg = check_pack(pack, tmp_path)
    assert not ok and log.name in msg


def test_a_tampered_verdict_is_caught(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    pack = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    (pack / VERDICT).write_text((pack / VERDICT).read_text().replace('"accept"', '"reject"', 1))
    ok, msg = check_pack(pack, tmp_path)
    assert not ok and "verdict differs" in msg


def test_a_case_directory_pack_round_trips(tmp_path):
    case = tmp_path / "cavity"
    shutil.copytree(MINI, case)
    with gzip.open(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz", "rt") as fh:
        (case / "log.icoFoam").write_text(fh.read())
    pack = write_pack(verify_path(case), tmp_path / "pack", case.name)
    ok, msg = check_pack(pack, case)
    assert ok, msg


def test_batch_finds_case_directories_and_logs_and_counts_with_denominators(tmp_path):
    for sid in ("icoFoam_cavity__healthy", "icoFoam_cavity__cfl_blowup"):
        _copy(tmp_path, sid)
    case = tmp_path / "cases" / "cavity"
    shutil.copytree(MINI, case)
    with gzip.open(CORPUS / "icoFoam_cavity__healthy.log.gz", "rt") as fh:
        (case / "log.icoFoam").write_text(fh.read())
    (case / "processor0").mkdir()
    (case / "processor0" / "log.ignored").write_text("x")
    runs = discover(tmp_path)
    assert len(runs) == 3
    # The mini_case fixture sets nu 5.0 m^2/s on purpose: reading the case
    # directory, the verifier catches it, so the case run is not trustworthy.
    line, code = summarise(verify_all(tmp_path))
    assert line == "3 runs: 1 of 3 accept, 2 of 3 reject, 0 of 3 insufficient"
    assert code == 2


def test_cli_pack_check_and_batch_exit_codes(tmp_path):
    from courant.cli.main import app
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    runner = CliRunner()
    r = runner.invoke(app, ["verify", str(log), "--case-json", str(cj), "--pack", str(tmp_path / "p")])
    assert r.exit_code == 0 and (tmp_path / "p" / VERDICT).exists()
    assert runner.invoke(app, ["verify", "--check-pack", str(tmp_path / "p"), "--root", str(tmp_path)]).exit_code == 0
    (tmp_path / "p" / VERDICT).write_text("{}\n")
    assert runner.invoke(app, ["verify", "--check-pack", str(tmp_path / "p"), "--root", str(tmp_path)]).exit_code == 5
    assert runner.invoke(app, ["verify", "--batch", str(tmp_path)]).exit_code == 0


# ------------------------------------------------------------------ the manifest is what was read (VER-14)
#
# T-MANIFEST-1. `verdict.run.inputs` must list exactly the case files the pipeline read. The
# observer that decides "read" is the interpreter itself: an audit hook on the "open" event sees
# every file a reader opens, whichever library opens it, and knows nothing of the recorder under
# test. Opens made only to hash a file for the manifest (`InputFile.of`) are not reads. Audit
# hooks cannot be removed, so the hook is installed once per session and records only while a
# watch is active; outside one it returns at its first line.

_SCHEMA_PY = os.path.join("courant", "verify", "schema.py")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_watch: dict | None = None
_hooked = False


def _audit(event: str, args: tuple) -> None:
    watch = _watch
    if watch is None or event != "open":
        return
    try:
        target, mode, flags = (tuple(args) + (None, None))[:3]   # documented: path, mode, flags
        if isinstance(target, int):
            return                                       # an already-open descriptor
        path = os.path.realpath(os.fsdecode(os.fspath(target)))
        if not path.startswith(watch["prefix"]):
            return
        rel = Path(os.path.relpath(path, watch["root"])).as_posix()
        if (any(c in mode for c in "wax+") if isinstance(mode, str)
                else bool((flags or 0) & _WRITE_FLAGS)):
            watch["writes"].add(rel)                     # TST-4: verification never writes
        if not os.path.isfile(path):
            return                                       # a probe for a file that is not there
        frame = sys._getframe(1)
        while frame is not None:
            code = frame.f_code
            if code.co_name == "of" and code.co_filename.endswith(_SCHEMA_PY):
                return                                   # InputFile.of: hashing, not reading
            frame = frame.f_back
        watch["opened"].add(rel)
    except Exception as exc:                             # noqa: BLE001 -- a raising hook breaks the open
        watch["errors"].append(repr(exc))


@contextlib.contextmanager
def _observe(root: Path):
    """The case-relative POSIX paths of the regular files under `root` opened in the block."""
    global _watch, _hooked
    if not _hooked:
        sys.addaudithook(_audit)
        _hooked = True
    real = os.path.realpath(root)
    watch = {"root": real, "prefix": real + os.sep, "opened": set(), "writes": set(),
             "errors": []}
    _watch = watch
    try:
        yield watch["opened"]
    finally:
        _watch = None
    assert not watch["errors"], f"the observer failed: {watch['errors']}"
    assert not watch["writes"], f"verification opened case files for writing: {watch['writes']}"


def _verify_observed(case: Path, spec: Path | None = None, checks: str = "default"):
    """(verdict, files the readers opened) over `from_path` + `with_spec` + `verify_inputs`."""
    from courant.verify import from_path, verify_inputs, with_spec

    with _observe(case) as opened:
        inputs = from_path(case, checks=checks)
        if spec is not None:
            inputs = with_spec(inputs, spec)
        verdict = verify_inputs(inputs)
    return verdict, set(opened)


def _manifest(verdict) -> list[dict]:
    return [f.model_dump() for f in verdict.run.inputs]


def _foam(cls: str, obj: str, body: str) -> str:
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n    object      {obj};\n}}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            + body)


#: The initial condition T3 edits. No value in the verdict depends on it (the case-config reader
#: takes only boundary types from `0/`, and the field reader reads `0.5/`), so only the manifest
#: can notice the edit.
_U_INITIAL = "internalField   uniform (0 0 0);"

#: Format fixtures for a membership test; no parser is fitted to them. `0/U` takes its inlet
#: value through `#include "include/initialConditions"` and `$Uinlet`, the way the pisoFoam
#: tutorials write it. `0.orig/T` has no `0/T`. The last three are controls: nothing reads them.
_DECK = {
    "0/U": _foam("volVectorField", "U", f"""#include        "include/initialConditions"

dimensions      [0 1 -1 0 0 0 0];

{_U_INITIAL}

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform $Uinlet;
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    walls
    {{
        type            noSlip;
    }}
}}
"""),
    "0/include/initialConditions": "Uinlet          (1 0 0);\n",
    "0/p": _foam("volScalarField", "p", """dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }

    walls
    {
        type            zeroGradient;
    }
}
"""),
    "0.orig/T": _foam("volScalarField", "T", """dimensions      [0 0 0 1 0 0 0];

internalField   uniform 300;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 300;
    }

    outlet
    {
        type            zeroGradient;
    }

    walls
    {
        type            zeroGradient;
    }
}
"""),
    "system/decomposeParDict": _foam("dictionary", "decomposeParDict",
                                     "numberOfSubdomains 2;\n\nmethod          scotch;\n"),
    "constant/g": _foam("uniformDimensionedVectorField", "g",
                        "dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 -9.81 0);\n"),
    "Allrun": ('#!/bin/sh\ncd "${0%/*}" || exit\n. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions\n\n'
               "runApplication blockMesh\nrunApplication checkMesh\n"
               "runApplication $(getApplication)\n"),
}

#: The one item added to the drafted spec: a value of `0/T`, which this case has only as
#: `0.orig/T`, so conformance must fall back to read it whatever the drafter chooses.
_T_ITEM = {"name": "initial T", "file": "0/T", "key": "internalField", "value": 300.0}


def _case(tmp: Path) -> Path:
    """The T-MANIFEST-1 test case: `mini_case`, plus real solver output for the log, the
    checkMesh log and a monitor, `0/` with an `#include`, a `0.orig/` file with no `0/` twin, a
    solved time directory, two files no reader reads, and `constant/g`, which the case reader
    reads for the buoyant regime under `--checks next` (TRD VER-17, GATE-13) and conformance
    reads when the drafted specification names gravity."""
    from tests.unit.test_case_shapes import FIELD

    case = tmp / "cavity"
    shutil.copytree(MINI, case)
    with gzip.open(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz", "rt") as fh:
        (case / "log.icoFoam").write_text(fh.read())
    shutil.copy(REAL / "cavity_checkmesh.log", case / "log.checkMesh")
    monitor = Path("postProcessing/courantPressureInlet/0/surfaceFieldValue.dat")
    (case / monitor).parent.mkdir(parents=True)
    shutil.copy(MONITORED / "simpleFoam_pitzDaily__shipped" / monitor, case / monitor)
    for rel, text in _DECK.items():
        (case / rel).parent.mkdir(parents=True, exist_ok=True)
        (case / rel).write_text(text)
    (case / "0.5").mkdir()
    (case / "0.5" / "U").write_text(FIELD % 1)
    return case


def _spec(case: Path, out: Path, *extra: dict) -> Path:
    """`draft_spec(case)` plus `extra`, written to `out` (outside the case directory)."""
    from courant.spec import Spec, draft_spec

    data = json.loads(draft_spec(case).to_json())
    data["items"].extend(extra)
    path = out / "spec.json"
    path.write_text(Spec.model_validate(data).to_json())
    return path


@pytest.mark.parametrize("mode", ["default", "next"])
@pytest.mark.parametrize("subject", ["case", "case_with_spec", "demo", "demo_with_spec"])
def test_the_manifest_is_exactly_what_the_readers_opened(tmp_path, subject, mode):
    """C1. Read means listed, and listed means read: on the test case and on the bundled demo
    case (a real simpleFoam/pitzDaily run), with and without a specification."""
    if subject.startswith("case"):
        case, extra = _case(tmp_path), (_T_ITEM,)
    else:
        from courant.data import extract_demo_case

        case, extra = extract_demo_case(tmp_path / "demo"), ()
    spec = _spec(case, tmp_path, *extra) if subject.endswith("with_spec") else None

    verdict, opened = _verify_observed(case, spec, mode)

    listed = {f.path for f in verdict.run.inputs}
    assert {"system/controlDict", "system/fvSolution"} <= opened, "the observer sees no reader"
    assert listed == opened, (f"read but not listed: {sorted(opened - listed)}; "
                              f"listed but not read: {sorted(listed - opened)}")


@pytest.mark.parametrize("mode", ["default", "next"])
@pytest.mark.parametrize("with_spec", [False, True], ids=["no_spec", "spec"])
def test_editing_each_file_changes_the_manifest_only_if_it_was_read(tmp_path, with_spec, mode):
    """C2 and C8. Append to one file at a time and re-verify. A file the pipeline read changes
    the manifest, in that file's own entry; a file it did not read changes neither the manifest
    nor the verdict."""
    case = _case(tmp_path)
    spec = _spec(case, tmp_path, _T_ITEM) if with_spec else None
    base, read = _verify_observed(case, spec, mode)
    again = verify_path(case, spec=spec, checks=mode)
    manifest = _manifest(base)
    assert _manifest(again) == manifest and again.to_json() == base.to_json()       # TST-3
    assert [e["path"] for e in manifest] == sorted(e["path"] for e in manifest)
    # The oracle is the observer, so first check it saw what this case is built around.
    assert {"0/U", "0.5/U", "log.checkMesh"} <= read, sorted(read)
    assert not {"system/decomposeParDict", "Allrun"} & read, sorted(read)
    # The case reader's under --checks next (VER-17), conformance's when the spec names it.
    assert ("constant/g" in read) == (mode == "next" or with_spec), sorted(read)

    problems = []
    for rel in sorted(p.relative_to(case).as_posix() for p in case.rglob("*") if p.is_file()):
        path = case / rel
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        try:
            edited = verify_path(case, spec=spec, checks=mode)
        finally:
            path.write_bytes(original)
        after = _manifest(edited)
        before_by, after_by = {e["path"]: e for e in manifest}, {e["path"]: e for e in after}
        if rel in read:
            moved = sorted(p for p in before_by.keys() | after_by.keys()
                           if before_by.get(p) != after_by.get(p))
            if (moved != [rel] or rel not in before_by
                    or after_by[rel]["sha256"] == before_by[rel]["sha256"]):
                problems.append(f"{rel}: read, but editing it changed {moved or 'nothing'} "
                                "in the manifest")
        elif after != manifest:
            problems.append(f"{rel}: not read, but editing it changed the manifest")
        elif edited.to_json() != base.to_json():
            problems.append(f"{rel}: not read, but editing it changed the verdict")
    assert not problems, "\n".join(problems)


def test_a_pack_is_invalidated_by_an_edit_to_an_initial_condition(tmp_path):
    """C6. A pack re-checks every file that decided its verdict, `0/` included."""
    case = _case(tmp_path)
    pack = write_pack(verify_path(case), tmp_path / "pack", case.name)
    ok, msg = check_pack(pack, case)
    assert ok, msg
    u = case / "0" / "U"
    text = u.read_text()
    assert _U_INITIAL in text
    u.write_text(text.replace(_U_INITIAL, "internalField   uniform (1 0 0);"))

    ok, msg = check_pack(pack, case)

    assert not ok and msg.startswith("input changed: 0/U "), (ok, msg)


def test_no_fixed_list_of_case_files_remains():
    """C7. "Which files decided this verdict" has one implementation, the readers' own notes.
    The fixed guess (`CASE_FILES`) and the second copy of `discover`'s file selection
    (`monitor_files`, which ignores `discover`'s byte budget) are deleted, not moved."""
    from courant.converge import series
    from courant.verify import ingest

    remaining = [f"{mod.__name__}.{name}" for mod, name in
                 ((ingest, "CASE_FILES"), (series, "monitor_files")) if hasattr(mod, name)]
    assert remaining == [], f"a fixed list of case files remains: {remaining}"
    named = []
    for p in sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts):
        for node in ast.walk(ast.parse(p.read_text())):
            name = getattr(node, "id", None) or getattr(node, "attr", None)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.alias)):
                name = node.name
            if name in ("CASE_FILES", "monitor_files"):
                named.append(f"{p.relative_to(ROOT)}:{node.lineno} {name}")
    assert named == [], f"a fixed list of case files is still named: {named}"


# ------------------------------------------------------------------ T-MANIFEST-2 (VER-14 completed)
# A file is listed only after its read succeeded. An unreadable file read nothing, so it is left
# out and the reader's fallback stands, exactly as before T-MANIFEST-1; a listed file that cannot
# then be hashed is an input error naming it. Mode 000 does not stop root, hence the skip.

_root_reads_everything = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")


def _app():
    from courant.cli.main import app

    return app


def _without_inputs(verdict) -> dict:
    doc = json.loads(verdict.to_json())
    doc["run"].pop("inputs")
    return doc


@contextlib.contextmanager
def _unreadable(path: Path):
    mode = path.stat().st_mode
    path.chmod(0)
    try:
        yield
    finally:
        path.chmod(mode)


@_root_reads_everything
@pytest.mark.parametrize("rel", ["0/U", "0/p", "0.5/U", "0/include/initialConditions"])
@pytest.mark.parametrize("with_spec", [False, True], ids=["no_spec", "spec"])
def test_an_unreadable_file_is_left_out_and_the_verdict_stands(tmp_path, rel, with_spec):
    """An unreadable `0/` or field file gives the verdict it gave before the manifest existed:
    the one a case without that file gives (apart from the sentence a specification item writes
    about it). The file is not listed, and every other file is listed as it was."""
    case = _case(tmp_path)
    spec = _spec(case, tmp_path, _T_ITEM) if with_spec else None
    readable = verify_path(case, spec=spec)
    if rel == "0/include/initialConditions" and not with_spec:
        assert rel not in {f.path for f in readable.run.inputs}   # only conformance reads it
    else:
        assert rel in {f.path for f in readable.run.inputs}, "the control: a readable file"

    with _unreadable(case / rel):
        verdict, opened = _verify_observed(case, spec)        # no traceback
        cli = CliRunner().invoke(_app(), ["verify", str(case), "--json"]
                                 + (["--spec", str(spec)] if spec else []))

    listed = {f.path for f in verdict.run.inputs}
    assert rel not in listed, f"{rel} could not be read, and it is listed"
    assert listed == opened - {rel}, (sorted(opened - listed), sorted(listed - opened))
    assert cli.exit_code in (0, 2, 3), (cli.exit_code, cli.output)
    assert json.loads(cli.output)["run"]["inputs"]

    gone = tmp_path / "gone" / case.name                 # the run id is the directory name
    shutil.copytree(case, gone)
    (gone / rel).unlink()
    absent = verify_path(gone, spec=spec)
    assert _manifest(verdict) == _manifest(absent)
    if not with_spec:
        expected = _without_inputs(absent)
        if rel == "0.5/U":
            # TRD VER-7b (the review of 21 Oct 2026): a field file that is there and could not be
            # opened is named, not called missing: NO_FIELDS gives way to the sentence naming it,
            # and nothing else in the document moves.
            from courant.verify import sentences
            from courant.verify.pipeline import _dedupe

            lines = [s for s in expected["not_assessed"] if s != sentences.NO_FIELDS]
            assert lines[-1] == sentences.NO_SPEC
            expected["not_assessed"] = [*_dedupe([*lines[:-1], sentences.fields_unread(
                "0.5", ["U"])]), sentences.NO_SPEC]
        assert _without_inputs(verdict) == expected
    else:
        assert (verdict.verdict, verdict.rule) == (absent.verdict, absent.rule)


@_root_reads_everything
def test_an_unreadable_dictionary_is_named_as_unreadable_not_a_traceback(tmp_path):
    """`system/fvSolution` at mode 000: the case reader's fallback (`unreadable`, reason
    `case.unreadable_dict`) stands, and the file is not listed, since nothing was read."""
    case = _case(tmp_path)
    with _unreadable(case / "system" / "fvSolution"):
        verdict = verify_path(case)
    listed = {f.path for f in verdict.run.inputs}
    assert "system/fvSolution" not in listed and "system/controlDict" in listed
    assert "system/fvSolution" in verdict.to_json()


@_root_reads_everything
def test_a_batch_with_one_unreadable_case_reports_every_case(tmp_path):
    """Before this node, one unreadable `0/U` stopped `verify --batch` and reported none."""
    root = tmp_path / "runs"
    root.mkdir()
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
        shutil.copytree(_case(tmp_path / name), root / name)
    with _unreadable(root / "b" / "0" / "U"):
        results = verify_all(root)
    assert [(t.name, err) for t, _, err in results] == [("a", None), ("b", None), ("c", None)]
    line, _ = summarise(results)
    assert line.startswith("3 runs:") and "unreadable" not in line, line


def test_a_file_that_vanishes_before_it_is_hashed_is_an_input_error_naming_it(tmp_path,
                                                                            monkeypatch):
    """A field file read and then purged (`purgeWrite` during a live verify) cannot be hashed:
    an `InputError` (exit 4) that names it, never a traceback, never a manifest that omits it."""
    from courant.verification import _fluid_reader
    from courant.verify import InputError
    from courant.verify.schema import manifest

    case = _case(tmp_path)
    field = (case / "0.5" / "U").read_text()
    real = _fluid_reader.read_fields

    def read_then_purge(case_dir, time=None, **kw):     # kw: `fluxes`, VER-18
        data = real(case_dir, time, **kw)
        (Path(case_dir) / "0.5" / "U").unlink(missing_ok=True)
        return data

    monkeypatch.setattr(_fluid_reader, "read_fields", read_then_purge)
    with pytest.raises(InputError, match=r"^0\.5/U "):
        verify_path(case)

    (case / "0.5" / "U").write_text(field)
    cli = CliRunner().invoke(_app(), ["verify", str(case), "--json"])
    assert cli.exit_code == 4 and "0.5/U" in cli.output, (cli.exit_code, cli.output)
    assert cli.exception is None or isinstance(cli.exception, SystemExit), cli.exception

    with pytest.raises(InputError, match="^system/nothing "):
        manifest(case, ["system/nothing"])


def test_a_recording_inside_a_recording_is_seen_by_both(tmp_path):
    """`courant convergence` records its own reads inside the verdict's recording; the verdict
    still lists them. `reading` notes a file read and unparsed, and not one that failed to open."""
    from courant.case.reads import note, reading, recording

    (tmp_path / "a").write_text("x")
    (tmp_path / "b").write_text("y")
    (tmp_path / "c").write_text("z")
    with recording(tmp_path) as outer:
        with recording(tmp_path) as inner:
            note(tmp_path / "a")
        with pytest.raises(PermissionError), reading(tmp_path / "b"):
            raise PermissionError("not read")
        with pytest.raises(ValueError), reading(tmp_path / "c"):
            raise ValueError("read, did not parse")
    assert inner.paths() == ["a"] and outer.paths() == ["a", "c"]


# ------------------------------------------------------------------ a pack never chooses where its run is (REP-5)
#
# Hard-proofing audit, security (21 Oct 2026). The manifest is the pack's own claim, so nothing in
# it may decide which file is verified: `check_pack` took `meta.target` and `meta.case_json` from
# it and joined them to the root unconfined, and re-hashed only the lines the manifest still
# listed. A pack made from a good log, with its manifest rewritten to name a copy of that log
# inside the pack (or anywhere on disk) and no input lines, re-derived "identical" with exit 0
# while the run under --root rejects: a log verdict names only the log's file name, so any copy
# of a good log gives the same bytes. Now the pack names its log by file name, directly under the
# root; every listed file resolves under the root; the lines must be the re-derived verdict's
# inputs, line for line; and a pack that lists none is refused.

GOOD_LOG, BAD_LOG = REAL / "cavity_converged.log", REAL / "cavity_cfl_diverged.log"


def _rewrite(pack: Path, lines: list[str], **meta) -> None:
    """Rewrite only `inputs.manifest`, as anyone holding the pack can."""
    _entries, old = read_manifest(pack)
    (pack / MANIFEST).write_text("".join(f"{ln}\n" for ln in lines)
                                 + "# meta " + json.dumps({**old, **meta}, sort_keys=True))


def _line(path: Path, name: str) -> str:
    import hashlib

    data = path.read_bytes()
    return f"{hashlib.sha256(data).hexdigest()}  {len(data)}  {name}"


def _good_and_bad(tmp: Path, *, pack_in_run: bool = True) -> tuple[Path, Path, Path]:
    """(good, run, pack): a pack honestly made from `good/log.icoFoam` (accept), and a run
    whose own `log.icoFoam` rejects; the pack sits inside the run, as an agent hands it over."""
    good, run = tmp / "good", tmp / "run"
    good.mkdir()
    run.mkdir()
    shutil.copy(GOOD_LOG, good / "log.icoFoam")
    shutil.copy(BAD_LOG, run / "log.icoFoam")
    assert verify_path(good / "log.icoFoam").verdict == "accept"
    assert verify_path(run / "log.icoFoam").verdict == "reject"
    pack = write_pack(verify_path(good / "log.icoFoam"),
                      run / "evidence" if pack_in_run else tmp / "evidence", "log.icoFoam")
    ok, msg = check_pack(pack, run)                     # untampered: the run's own log differs
    assert not ok and msg.startswith("input changed: log.icoFoam"), msg
    return good, run, pack


def test_a_pack_cannot_name_a_copy_of_a_good_log_inside_itself(tmp_path):
    """The audit's reproduction (A): a copy of the good log inside the pack, the target pointed
    at it, every input line deleted. It was `identical: 0 inputs re-hashed`, exit 0."""
    good, run, pack = _good_and_bad(tmp_path)
    (pack / ".src").mkdir()
    shutil.copy(good / "log.icoFoam", pack / ".src" / "log.icoFoam")
    _rewrite(pack, [], target="evidence/.src/log.icoFoam")

    ok, msg = check_pack(pack, run)

    assert not ok, msg
    assert "evidence/.src/log.icoFoam" in msg and "file name" in msg, msg


def test_a_pack_cannot_list_the_copy_it_points_at(tmp_path):
    """(B) the same, with the copy listed, so the check said `1 inputs re-hashed`."""
    good, run, pack = _good_and_bad(tmp_path)
    (pack / ".src").mkdir()
    shutil.copy(good / "log.icoFoam", pack / ".src" / "log.icoFoam")
    _rewrite(pack, [_line(pack / ".src" / "log.icoFoam", "evidence/.src/log.icoFoam")],
             target="evidence/.src/log.icoFoam")

    ok, msg = check_pack(pack, run)

    assert not ok, msg
    assert "evidence/.src/log.icoFoam" in msg and "file name" in msg, msg   # refused for B, not by chance


@pytest.mark.parametrize("how", ["absolute", "dotdot"])
def test_a_pack_cannot_name_a_run_outside_the_root(tmp_path, how):
    """(C, D) the pack outside the root, its target an absolute path or `../good/log.icoFoam`:
    no copy needed. Each was `identical`, exit 0."""
    good, run, pack = _good_and_bad(tmp_path, pack_in_run=False)
    target = str(good / "log.icoFoam") if how == "absolute" else "../good/log.icoFoam"
    _rewrite(pack, [], target=target)

    ok, msg = check_pack(pack, run)

    assert not ok and target in msg, msg
    _rewrite(pack, [_line(good / "log.icoFoam", target)], target=target)
    ok, msg = check_pack(pack, run)
    assert not ok and target in msg, msg


def test_the_cli_exits_5_on_the_audit_reproduction(tmp_path):
    """`courant verify --check-pack run/evidence --root run` on (A): MISMATCH, exit 5."""
    good, run, pack = _good_and_bad(tmp_path)
    (pack / ".src").mkdir()
    shutil.copy(good / "log.icoFoam", pack / ".src" / "log.icoFoam")
    _rewrite(pack, [], target="evidence/.src/log.icoFoam")

    r = CliRunner().invoke(_app(), ["verify", "--check-pack", str(pack), "--root", str(run)])

    assert r.exit_code == 5 and r.output.startswith("MISMATCH "), (r.exit_code, r.output)


def test_a_pack_that_lists_no_inputs_is_refused(tmp_path):
    """Even naming the right run: with no line, nothing is re-hashed and nothing ties the
    pack to the run's bytes."""
    good, _run, pack = _good_and_bad(tmp_path)
    assert check_pack(pack, good)[0]
    _rewrite(pack, [])

    ok, msg = check_pack(pack, good)

    assert not ok and "lists no inputs" in msg, msg


def test_a_listed_file_must_resolve_under_the_root(tmp_path):
    """A line naming `..`, an absolute path, or a link in the root that leads out of it."""
    good, run, pack = _good_and_bad(tmp_path)
    for rel in ("../good/log.icoFoam", str(good / "log.icoFoam")):
        _rewrite(pack, [_line(good / "log.icoFoam", rel)])
        ok, msg = check_pack(pack, run)
        assert not ok and rel in msg and "under the root" in msg, (rel, msg)

    (run / "log.good").symlink_to(good / "log.icoFoam")    # the target alone links out
    _rewrite(pack, [_line(run / "log.icoFoam", "log.icoFoam")], target="log.good")
    ok, msg = check_pack(pack, run)
    assert not ok and "log.good" in msg and "under the root" in msg, msg

    (run / "log.icoFoam").unlink()
    (run / "log.icoFoam").symlink_to(good / "log.icoFoam")
    _rewrite(pack, [_line(good / "log.icoFoam", "log.icoFoam")], target="log.icoFoam")
    ok, msg = check_pack(pack, run)
    assert not ok and "log.icoFoam" in msg and "under the root" in msg, msg
    assert str(good) not in msg, msg                     # never where a link leads (MCP-3)


def test_a_case_json_and_a_specification_are_named_by_file_name(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    pack = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    assert check_pack(pack, tmp_path)[0]
    other = tmp_path / "elsewhere"
    other.mkdir()
    shutil.copy(cj, other / cj.name)
    entries, _meta = read_manifest(pack)
    lines = [f"{s}  {b}  {p}" for s, b, p in entries]

    _rewrite(pack, lines, case_json=f"elsewhere/{cj.name}")
    ok, msg = check_pack(pack, tmp_path)
    assert not ok and f"elsewhere/{cj.name}" in msg and "file name" in msg, msg

    _rewrite(pack, lines, case_json=cj.name, spec="../spec.json")
    ok, msg = check_pack(pack, tmp_path)
    assert not ok and "../spec.json" in msg and "file name" in msg, msg


def test_the_manifest_lines_must_be_the_verdicts_inputs(tmp_path):
    """A line dropped (the file is no longer re-hashed) or one added (a file the verdict never
    read) is a pack that does not say what decided it, as `admit` holds for a record."""
    case = _case(tmp_path)
    pack = write_pack(verify_path(case), tmp_path / "pack", case.name)
    assert check_pack(pack, case)[0]
    entries, _meta = read_manifest(pack)
    lines = [f"{s}  {b}  {p}" for s, b, p in entries]
    assert any(p == "0/U" for _s, _b, p in entries)

    _rewrite(pack, [ln for ln in lines if not ln.endswith("  0/U")])
    ok, msg = check_pack(pack, case)
    assert not ok and "0/U" in msg and "not listed" in msg, msg

    (case / "notes.txt").write_text("not read by any reader\n")
    _rewrite(pack, sorted(lines + [_line(case / "notes.txt", "notes.txt")],
                          key=lambda ln: ln.split("  ", 2)[2]))
    ok, msg = check_pack(pack, case)
    assert not ok and "notes.txt" in msg and "not read" in msg, msg


def test_an_identical_pack_says_which_run_it_re_verified(tmp_path):
    """A log pack names its log by file name in the root, and the OK line says which, so a
    reviewer sees the file that was judged."""
    good, _run, pack = _good_and_bad(tmp_path)

    ok, msg = check_pack(pack, good)

    assert ok and msg.startswith("identical: log.icoFoam under "), msg
    assert "1 inputs re-hashed" in msg, msg
    case = _case(tmp_path)
    ok, msg = check_pack(write_pack(verify_path(case), tmp_path / "case_pack", case.name), case)
    assert ok and msg.startswith("identical: the case directory "), msg


# ------------------------------------------------------------------ nor how its run is judged (REP-5, review)
#
# Review of 4a136e6 (21 Oct 2026). The pack no longer chose where its run is, and still chose how
# it is judged: whether a log goes with the sidecar beside it, whether a case directory is judged
# by its log alone, which hash its specification was locked at, and, under a name a reviewer
# cannot tell from the solver's (a zero-width space, a trailing space, a lookalike), which log.
# An honest pack of the reduced run re-derived "identical", exit 0, while `courant verify` on the
# run as it stands rejects. Now the root decides each of them, or the reviewer does.

OVERCONSTRAINED = "simpleFoam_pitzDaily__overconstrained_pu"   # accepts alone, rejects with its sidecar


def test_a_log_pack_is_checked_with_the_sidecar_the_root_holds(tmp_path):
    """Scenario 1: an honest pack of the log alone (`courant verify x.log.gz --pack P`), beside
    the sidecar `--batch` pairs it with. The pack can neither leave that sidecar out nor name
    another; with no sidecar in the root, the log alone still checks."""
    run = tmp_path / "run"
    run.mkdir()
    log, cj = _copy(run, OVERCONSTRAINED)
    assert verify_path(log, cj).verdict == "reject"          # the setup, read from the sidecar
    assert verify_path(log).verdict == "accept"
    pack = write_pack(verify_path(log), tmp_path / "pack", log.name)

    ok, msg = check_pack(pack, run)

    assert not ok and cj.name in msg and "sidecar" in msg, msg
    r = CliRunner().invoke(_app(), ["verify", "--check-pack", str(pack), "--root", str(run)])
    assert r.exit_code == 5 and r.output.startswith("MISMATCH "), (r.exit_code, r.output)

    lenient = run / "lenient.case.json"
    shutil.copy(CORPUS / "simpleFoam_pitzDaily__healthy.case.json", lenient)
    other = write_pack(verify_path(log, lenient), tmp_path / "other", log.name, lenient.name)
    ok, msg = check_pack(other, run)
    assert not ok and lenient.name in msg and cj.name in msg, msg

    cj.unlink()                                              # the root holds no sidecar for it
    ok, msg = check_pack(other, run)
    assert not ok and lenient.name in msg, msg
    ok, msg = check_pack(pack, run)
    assert ok, msg


def test_a_log_pack_does_not_re_verify_a_case_directory(tmp_path):
    """Scenario 2: a case directory `courant verify R` rejects on its setup (mini_case sets nu
    5.0 m^2/s) and an honest pack of its log alone, which accepts. A log pack checks under a case
    directory only when `courant verify R` reads nothing there that its verdict did not."""
    case = tmp_path / "cavity"
    shutil.copytree(MINI, case)
    with gzip.open(CORPUS / "icoFoam_cavity__healthy.log.gz", "rt") as fh:
        (case / "log.icoFoam").write_text(fh.read())
    assert verify_path(case).verdict == "reject"
    assert verify_path(case / "log.icoFoam").verdict == "accept"
    pack = write_pack(verify_path(case / "log.icoFoam"), tmp_path / "pack", "log.icoFoam")

    ok, msg = check_pack(pack, case)

    assert not ok and "case directory" in msg and "system/controlDict" in msg, msg
    assert check_pack(write_pack(verify_path(case), tmp_path / "case_pack", case.name), case)[0]


def test_a_log_beside_the_runs_own_cannot_stand_in_for_it(tmp_path):
    """Major 2: the run's own `log.icoFoam` rejects; a copy of a good log sits beside it under a
    name that reads the same (a zero-width space, a trailing space, a tab) or nearly (`l` for
    `i`), with the copy's own honest pack. Each was `identical: log.icoFoam under R`, exit 0."""
    good, run, _pack = _good_and_bad(tmp_path)
    for i, name in enumerate(("log.icoFoam​", "log.icoFoam ", "log.icoFoam\t")):
        shutil.copy(good / "log.icoFoam", run / name)
        pack = write_pack(verify_path(run / name), tmp_path / f"copy{i}", name)
        ok, msg = check_pack(pack, run)
        assert not ok and repr(name) in msg and "printable ASCII" in msg, (name, msg)
        (run / name).unlink()

    shutil.copy(good / "log.icoFoam", run / "log.lcoFoam")
    pack = write_pack(verify_path(run / "log.lcoFoam"), tmp_path / "lookalike", "log.lcoFoam")
    ok, msg = check_pack(pack, run)
    assert not ok and "case directory" in msg and "log.icoFoam" in msg, msg   # the run's own


def test_the_reviewer_gives_the_locked_hash_not_the_pack(tmp_path):
    """Scenario 3: the pack drops its specification, or carries a lenient one locked at that
    file's own hash. With `--spec-sha256` the hash is the reviewer's and the pack is held to it;
    without, the OK line says whose hash it was."""
    from courant.spec import draft_spec, spec_sha256

    case = _case(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(draft_spec(case).to_json())
    locked = spec_sha256(spec)
    honest = write_pack(verify_path(case, spec=spec, spec_locked_sha256=locked),
                        tmp_path / "honest", case.name, spec_path=spec, spec_locked_sha256=locked)
    ok, msg = check_pack(honest, case, spec_sha256=locked)
    assert ok and f"specification {locked[:12]}, locked at the hash you gave" in msg, msg
    ok, msg = check_pack(honest, case)
    assert ok and f"specification {locked[:12]}, locked at the hash the pack carries" in msg, msg

    dropped = write_pack(verify_path(case), tmp_path / "dropped", case.name)
    assert check_pack(dropped, case)[0]
    ok, msg = check_pack(dropped, case, spec_sha256=locked)
    assert not ok and "carries no specification" in msg and locked[:12] in msg, msg

    lenient = tmp_path / "lenient.json"
    lenient.write_text(spec.read_text() + "\n")
    mine = spec_sha256(lenient)
    swapped = write_pack(verify_path(case, spec=lenient, spec_locked_sha256=mine),
                         tmp_path / "swapped", case.name, spec_path=lenient,
                         spec_locked_sha256=mine)
    assert check_pack(swapped, case)[0]                  # locked at its own hash, it re-derives
    ok, msg = check_pack(swapped, case, spec_sha256=locked)
    assert not ok and mine[:12] in msg and locked[:12] in msg, msg

    unlocked = write_pack(verify_path(case, spec=spec), tmp_path / "unlocked", case.name,
                          spec_path=spec)
    ok, msg = check_pack(unlocked, case)
    assert ok and f"specification {locked[:12]}, not locked" in msg, msg
    ok, msg = check_pack(unlocked, case, spec_sha256=locked.upper())
    assert not ok and "not locked" in msg and locked[:12] in msg, msg

    def cli(pack):
        return CliRunner().invoke(_app(), ["verify", "--check-pack", str(pack), "--root",
                                           str(case), "--spec-sha256", locked])

    assert cli(honest).exit_code == 0, cli(honest).output
    r = cli(swapped)
    assert r.exit_code == 5 and r.output.startswith("MISMATCH "), (r.exit_code, r.output)


def test_the_ok_line_says_how_the_run_was_judged(tmp_path):
    """The scope a pack re-derived, at exit 0: a log alone or with its sidecar, a case
    directory, and the specification (above)."""
    good, _run, pack = _good_and_bad(tmp_path)
    ok, msg = check_pack(pack, good)
    assert ok and "re-verified (a log, no sidecar, no specification)" in msg, msg
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    ok, msg = check_pack(write_pack(verify_path(log, cj), tmp_path / "p", log.name, cj.name),
                         tmp_path)
    assert ok and f"re-verified (a log with its sidecar {cj.name}, no specification)" in msg, msg
    case = _case(tmp_path)
    ok, msg = check_pack(write_pack(verify_path(case), tmp_path / "c", case.name), case)
    assert ok and "re-verified (no specification)" in msg, msg


def test_a_log_pack_names_a_file_not_a_directory(tmp_path):
    """A log pack's target that is a directory under R: the hash loop hashed R/<line> while the
    verdict read R/<target>/<line>, so a refusal named another file than the one read."""
    good, run, pack = _good_and_bad(tmp_path)
    (run / "sub").mkdir()
    shutil.copy(good / "log.icoFoam", run / "sub" / "log.icoFoam")
    _rewrite(pack, [_line(run / "log.icoFoam", "log.icoFoam")], target="sub")

    ok, msg = check_pack(pack, run)

    assert not ok and "'sub'" in msg and "not a file" in msg, msg


def test_a_root_that_is_a_link_loop_is_refused_without_a_traceback(tmp_path):
    """Python 3.11 raises RuntimeError resolving a link loop; the root's resolution did not
    catch it (MCP-3's `Confinement` caught it for a path, not for a root)."""
    from courant.case.reads import Confinement

    _good, _run, pack = _good_and_bad(tmp_path)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    Confinement([loop])                                  # does not raise

    ok, msg = check_pack(pack, loop)

    assert not ok and "not a directory" in msg, msg
    r = CliRunner().invoke(_app(), ["verify", "--check-pack", str(pack), "--root", str(loop)])
    assert r.exit_code == 5 and r.output.startswith("MISMATCH "), (r.exit_code, r.output)

