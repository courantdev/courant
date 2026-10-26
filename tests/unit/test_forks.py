"""Version detection (TRD ADP-1) and VER-5.10, for G-UNIT-FORKS.

The fork and version are read from the solver log's banner, or, when the log has none, from the
header of a file the run wrote (the latest time directory's), into `run.fork` and
`run.solver_version`; that and a not-checked line naming a version outside the measured set (ESI
v2406, `courant.forks.MEASURED`) are on by default. `system/controlDict`'s header is an input (it
may have been copied from another version's tutorial): what it names is reported in the
not-checked line only, and the version counts as unread. VER-5.10, a provisional accept on a
version outside the set made `insufficient`, changes a default accept, so it ships behind
`--checks next` in 0.6.0 (GATE-13).

The banners are synthetic: the ESI form is the one the committed v2406 runs print
(`certification/corpus/`) with its numbers changed, the Foundation form the one that fork prints.
ADP-2's real per-fork logs (T-FORK-FIX) wait on H-FORK-IMAGES; ADP-3 and ADP-4 are not here.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
from pathlib import Path

import pytest

from courant import forks
from courant.verify import from_path, verify_inputs, verify_path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
HEALTHY = "icoFoam_cavity__healthy"            # accepted, VER-5.6
BLOWUP = "icoFoam_cavity__cfl_blowup"          # rejected, VER-5.3


def esi(n: str, *, v_prefix: bool = False, build: str | None = None) -> str:
    """An ESI banner, as the committed v2406 runs print it, for release `n` (`2412`)."""
    shown = f"v{n}" if v_prefix else n
    build = build if build is not None else f"_be01ca78-20240625 OPENFOAM={n} version={n}"
    return ("/*---------------------------------------------------------------------------*\\\n"
            "| =========                 |                                                 |\n"
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
            f"|  \\\\    /   O peration     | Version:  {shown:<38s}|\n"
            "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
            "|    \\\\/     M anipulation  |                                                 |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            f"Build  : {build}\n"
            'Arch   : "LSB;label=32;scalar=64"\n')


def foundation(n: str, *, build: str | None = None) -> str:
    """A Foundation banner for release `n` (`12`)."""
    build = build if build is not None else f"{n}-0d1dab2ac6e8"
    return ("/*---------------------------------------------------------------------------*\\\n"
            "  =========                 |\n"
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
            "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
            f"    \\\\  /    A nd           | Version:  {n}\n"
            "     \\\\/     M anipulation  |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            f"Build  : {build}\n")


EXEC = "Exec   : icoFoam\nDate   : Aug 01 2026\n\nCreate time\n"


# ---------------------------------------------------------------- the banner reader
@pytest.mark.parametrize("banner, fork, version", [
    (esi("2406"), "esi", "v2406"),
    (esi("2412"), "esi", "v2412"),
    (esi("2512"), "esi", "v2512"),
    (esi("2012", v_prefix=True, build="_7bdb509494-20201222 OPENFOAM=2012"), "esi", "v2012"),
    (esi("2512", build=""), "esi", "v2512"),                  # no Build value: the Version line
    (foundation("12"), "foundation", "12"),
    (foundation("11"), "foundation", "11"),
    (foundation("dev", build="dev-2f1a2c9b3e7d"), "foundation", "dev"),
    (foundation("12", build="OpenFOAM-12"), "foundation", "12"),
], ids=["esi-v2406", "esi-v2412", "esi-v2512", "esi-v2012", "esi-version-line",
        "foundation-12", "foundation-11", "foundation-dev", "foundation-OpenFOAM-12"])
def test_the_banner_names_the_fork_and_the_version(banner, fork, version):
    seen = forks.read(banner + EXEC + "Time = 0.005\n")
    assert (seen.fork, seen.version, seen.source) == (fork, version, forks.LOG_BANNER)
    assert seen.measured is ((fork, version) == ("esi", "v2406"))


def test_the_build_line_wins_over_the_version_line_and_a_token_is_read():
    # A Version line that disagrees with the Build line: the Build line is the binary's own.
    assert forks.from_banner(esi("2406", build="_x-20251217 OPENFOAM=2512 version=2512")) == \
        ("esi", "v2512")
    assert forks.from_banner("| Version:  OpenFOAM-12\n") == ("foundation", "12")
    assert forks.from_banner("| Version:  OpenFOAM-v2406\n") == ("esi", "v2406")


def test_an_empty_build_line_does_not_take_the_next_line_for_its_value():
    """`Build  :` with nothing after it is an empty Build line: the reader goes on to the Version
    line, and never reads the line below the empty one as the build."""
    text = "| Version:  2512 |\n| Website:  www.openfoam.com |\nBuild  :\nOPENFOAM=2406 version=2406\n"
    assert forks.from_banner(text) == ("esi", "v2512")
    assert forks.from_banner("| Version:  12\nBuild  : \n11-0d1dab2ac6e8\n") == ("foundation", "12")


def _box(version: str, web: str, build: str, project: str = "OpenFOAM: The Open Source CFD Toolbox"):
    return ("/*---------------------------------------------------------------------------*\\\n"
            "| =========                 |                                                 |\n"
            f"| \\\\      /  F ield         | {project:<48s}|\n"
            f"|  \\\\    /   O peration     | Version:  {version:<38s}|\n"
            f"|   \\\\  /    A nd           | Web:      {web:<38s}|\n"
            "|    \\\\/     M anipulation  |                                                 |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            f"Build    : {build}\n")


@pytest.mark.parametrize("banner, fork, version, shown", [
    # foam-extend numbers its releases as the Foundation does; its website names it.
    (_box("4.1", "http://www.foam-extend.org", "4.1-d5c4d1c8b1e9", "foam-extend: Open Source CFD"),
     "foam-extend", "4.1", "OpenFOAM 4.1 (foam-extend)"),
    # OpenCFD's own releases (2.0 and earlier) under www.OpenFOAM.com, before the Foundation.
    (_box("2.0.1", "www.OpenFOAM.com", "2.0.1-51f1de99a4bc"), "opencfd", "2.0.1",
     "OpenFOAM 2.0.1 (OpenCFD)"),
    (_box("2.1.1", "www.OpenFOAM.com", "2.1.1-221db2718bbb"), "opencfd", "2.1.1",
     "OpenFOAM 2.1.1 (OpenCFD)"),
    # A website and a numbering of different forks, with no project that explains them.
    (_box("2412", "https://openfoam.org", "_x OPENFOAM=2412 version=2412"), None, "v2412",
     "OpenFOAM v2412 (fork not told)"),
    # The website of a project that is neither fork (a synthetic one): its numbering names no
    # fork, whichever fork's numbering it looks like.
    (_box("9.04", "www.example-cfd.test", "9.04-1a2b3c4d5e6f", "ExampleCFD: a derived toolbox"),
     None, "9.04", "OpenFOAM 9.04 (fork not told)"),
    (_box("2412", "https://example-cfd.test", "_x OPENFOAM=2412 version=2412"), None, "v2412",
     "OpenFOAM v2412 (fork not told)"),
], ids=["foam-extend-4.1", "opencfd-2.0.1", "opencfd-2.1.1", "org-website-esi-numbering",
        "other-project-dotted", "other-project-esi-numbering"])
def test_a_website_that_contradicts_the_version_line_is_never_overruled_by_it(
        banner, fork, version, shown):
    """The version's numbering alone would call each of these Foundation (or ESI): the website
    names the project, so the verdict never names the Foundation for a run it did not make."""
    seen = forks.read(banner + EXEC)
    assert (seen.fork, seen.version, seen.source) == (fork, version, forks.LOG_BANNER)
    assert seen.label == shown and seen.measured is False
    line = forks.not_checked(seen)
    assert "Foundation" not in line and f"names {shown}, which they do not cover" in line


@pytest.mark.parametrize("web", ["https://openfoam.org", "www.example-cfd.test"],
                         ids=["the-other-fork", "another-project"])
def test_a_contradicting_banner_on_a_measured_number_leaves_the_fork_unread(web):
    seen = forks.read(_box("2406", web, "_x OPENFOAM=2406 version=2406") + EXEC)
    assert (seen.fork, seen.version, seen.measured) == (None, "v2406", None)
    assert forks.not_checked(seen) == (
        "The OpenFOAM fork was not read: the solver log's banner names version v2406 under the "
        "website of another fork or project; the gate's error rates were measured on OpenFOAM "
        "v2406 (ESI) only.")


def test_a_website_alone_names_the_fork_and_not_the_version():
    seen = forks.read("|   \\\\  /    A nd           | Website:  https://openfoam.org\n" + EXEC)
    assert (seen.fork, seen.version, seen.measured) == ("foundation", None, False)
    seen = forks.read("| Website:  www.openfoam.com |\n" + EXEC)
    assert (seen.fork, seen.version, seen.measured) == ("esi", None, None)
    # Another project's website alone names neither fork nor version.
    assert forks.from_banner("| Web:      www.example-cfd.test |\n") == (None, None)


def test_only_the_banner_before_the_first_exec_line_is_read():
    """A log that runs two applications holds two banners: the first one, the solver's install."""
    text = esi("2406") + EXEC + "Time = 1\n" + foundation("12") + "Exec   : foamRun\n"
    assert (forks.read(text).fork, forks.read(text).version) == ("esi", "v2406")


def test_a_log_with_no_banner_reads_no_version_and_says_so():
    seen = forks.read("Create time\n\nTime = 0.005\nCourant Number mean: 0 max: 0\n")
    assert (seen.fork, seen.version, seen.source, seen.measured) == (None, None, None, None)
    assert forks.not_checked(seen) == (
        "The OpenFOAM fork and version were not read: the solver log has no banner; the gate's "
        "error rates were measured on OpenFOAM v2406 (ESI) only.")
    assert forks.read("") == forks.SolverVersion() and forks.read(None) == forks.SolverVersion()


def test_the_measured_set_is_one_constant_and_every_sentence_reads_it():
    assert forks.MEASURED == frozenset({("esi", "v2406")})
    assert forks.measured_label() == "OpenFOAM v2406 (ESI)"
    outside = forks.read(esi("2512") + EXEC)
    assert forks.not_checked(outside) == (
        "The gate's error rates were measured on OpenFOAM v2406 (ESI) only; the solver log's "
        "banner names OpenFOAM v2512 (ESI), which they do not cover.")
    assert forks.not_checked(forks.read(esi("2406") + EXEC)) is None


def test_every_committed_run_was_v2406_and_reads_so():
    """The 31 corpus logs and the logs of the case directories GATE-13's baseline verifies (the
    demo case is `test_a_case_whose_log_has_a_banner_reads_it`): all ESI v2406, from the banner,
    so ADP-1 adds no not-checked line to any committed verdict."""
    logs = sorted(CORPUS.glob("*.log.gz"))
    logs += sorted((ROOT / "certification" / "monitored").glob("*/log.simpleFoam"))
    logs += [ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"
             / "log.icoFoam.gz",
             ROOT / "tests" / "fixtures" / "buoyant" / "hotRoom_laminar_044" / "log.buoyantPimpleFoam"]
    assert len(logs) == 31 + 6 + 2
    for log in logs:
        raw = log.read_bytes()
        text = (gzip.decompress(raw) if log.suffix == ".gz" else raw).decode("utf-8", "replace")
        seen = forks.read(text)
        assert (seen.fork, seen.version, seen.source) == ("esi", "v2406", forks.LOG_BANNER), log
        assert forks.not_checked(seen) is None


# ---------------------------------------------------------------- the case's own headers
def _header(banner_lines: str) -> str:
    return (banner_lines + "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n    object      controlDict;\n}\n")


ESI_HEADER = ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
              "| =========                 |                                                 |\n"
              "|  \\\\    /   O peration     | Version:  v{n}                                 |\n"
              "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
              "\\*---------------------------------------------------------------------------*/\n")
FOUNDATION_HEADER = ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
                     "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
                     "    \\\\  /    A nd           | Version:  {n}\n"
                     "\\*---------------------------------------------------------------------------*/\n")


def _case(tmp_path: Path, control: str | None, times: dict[str, str] | None = None) -> Path:
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    if control is not None:
        (case / "system" / "controlDict").write_text(_header(control))
    (case / "0").mkdir()
    (case / "0" / "U").write_text(_header(FOUNDATION_HEADER.format(n="9")))   # never read
    for rel, text in (times or {}).items():
        (case / rel).parent.mkdir(parents=True, exist_ok=True)
        (case / rel).write_text(text)
    return case


NO_BANNER = "Create time\n\nTime = 0.005\n"


CONTROLDICT_NAMES = (
    "The OpenFOAM fork and version the run was solved on were not read: the solver log has no "
    "banner, and no file the run wrote names one; the header of system/controlDict, an input "
    "that may have been copied from another version's tutorial, names {label}. The gate's error "
    "rates were measured on OpenFOAM v2406 (ESI) only.")


@pytest.mark.parametrize("header, given", [
    (FOUNDATION_HEADER.format(n="12"), ("foundation", "12")),
    (ESI_HEADER.format(n="2406"), ("esi", "v2406")),
], ids=["foundation-12", "esi-v2406"])
def test_a_controldict_header_is_reported_and_never_taken_for_the_runs_version(
        tmp_path, header, given):
    """`system/controlDict` is an input: the solver read it, it did not write its header comment,
    which may come from another version's tutorial. What it names is said in the not-checked
    line, and the run's fork and version stay unread, as with no banner, so VER-5.10 never acts
    on it (`measured` is None) and `run.fork` stays null. Its read joins the manifest (VER-14)."""
    from courant.case.reads import recording

    case = _case(tmp_path, header)
    with recording(case) as reads:
        seen = forks.read(NO_BANNER, str(case))
    assert reads.paths() == ["system/controlDict"]
    assert (seen.fork, seen.version, seen.source, seen.measured) == (None, None, None, None)
    assert (seen.given.fork, seen.given.version) == given
    assert seen.given.source == "the header of system/controlDict"
    assert forks.not_checked(seen) == CONTROLDICT_NAMES.format(label=forks.label(*given))


def test_what_the_run_wrote_wins_over_the_controldict_it_was_given(tmp_path):
    """The latest time directory's files were written by the solver that ran; `controlDict` may
    be a tutorial of another version. 0/ is the run's input, never read for this."""
    written = _header(ESI_HEADER.format(n="2512").replace("v2512", " 2512"))
    case = _case(tmp_path, ESI_HEADER.format(n="2406"),
                 {"0.5/p": "not a header\n", "100/U": written, "100/p": written,
                  "100/uniform/time": "x"})
    seen = forks.read(NO_BANNER, str(case))
    assert (seen.fork, seen.version, seen.source) == ("esi", "v2512", "the header of 100/U")
    assert seen.measured is False and seen.given is None
    assert forks.not_checked(seen) == (
        "The gate's error rates were measured on OpenFOAM v2406 (ESI) only; the header of 100/U "
        "(a file the run wrote; the solver log has no banner) names OpenFOAM v2512 (ESI), which "
        "they do not cover.")


def test_a_time_directorys_dotfiles_are_passed_over_and_its_files_tried_until_one_names_a_fork(
        tmp_path):
    """macOS Finder writes `.DS_Store`, which sorts before every field ('.' before 'U'): it is
    never read. A file whose header names no fork (written by a script) is passed over for the
    next, so the header the run wrote is found and the controlDict is never reached."""
    from courant.case.reads import recording

    written = _header(ESI_HEADER.format(n="2512"))
    case = _case(tmp_path, FOUNDATION_HEADER.format(n="12"),
                 {"100/.DS_Store": "\x00\x00\x00\x01Bud1", "100/T": "FoamFile { }\n",
                  "100/U": written})
    with recording(case) as reads:
        seen = forks.read(NO_BANNER, str(case))
    assert (seen.fork, seen.version, seen.source) == ("esi", "v2512", "the header of 100/U")
    assert seen.given is None and reads.paths() == ["100/T", "100/U"]


def test_the_log_banner_wins_and_then_no_header_is_read(tmp_path):
    from courant.case.reads import recording

    case = _case(tmp_path, FOUNDATION_HEADER.format(n="12"))
    with recording(case) as reads:
        seen = forks.read(esi("2406") + EXEC, str(case))
    assert (seen.fork, seen.version, seen.source) == ("esi", "v2406", forks.LOG_BANNER)
    assert reads.paths() == []


def test_a_header_read_joins_the_manifest_and_one_measured_says_where_it_came_from(tmp_path):
    from courant.case.reads import recording

    case = _case(tmp_path, FOUNDATION_HEADER.format(n="12"),
                 {"100/U": _header(ESI_HEADER.format(n="2406"))})
    with recording(case) as reads:
        seen = forks.read(NO_BANNER, str(case))
    assert reads.paths() == ["100/U"]
    assert (seen.fork, seen.version, seen.measured) == ("esi", "v2406", True)
    assert forks.not_checked(seen) == (
        "The OpenFOAM fork and version (OpenFOAM v2406 (ESI)) were read from the header of "
        "100/U, a file the run wrote: the solver log has no banner.")


def test_under_next_a_time_directory_later_than_the_log_is_not_read_for_the_version(tmp_path):
    """A banner-less window of a run that ended at 0.005, in a case where another run, on v2312,
    wrote 100/ later. Under next the field checks hold that 100/ is not this run's
    (`fields.log_time`, VER-7b), and so does the version: it is read from 0.005/, the run's own.
    By default the latest directory is read, as in 0.6.0's first integration (the final review,
    21 Oct 2026, found the two readings of which directory is the run's in one next verdict)."""
    own, other = _header(ESI_HEADER.format(n="2406")), _header(ESI_HEADER.format(n="2312"))
    case = _case(tmp_path, None, {"0.005/U": own, "100/U": other})
    assert forks.read(NO_BANNER, str(case)).version == "v2312"                 # the default
    seen = forks.read(NO_BANNER, str(case), until=0.005)
    assert (seen.version, seen.source, seen.measured) == ("v2406", "the header of 0.005/U", True)
    log = case / "log.icoFoam"
    log.write_text(NO_BANNER + "smoothSolver:  Solving for Ux, Initial residual = 1e-06, "
                   "Final residual = 1e-07, No Iterations 1\n")
    for mode, label in (("default", "OpenFOAM v2312 (ESI)"), ("next", "OpenFOAM v2406 (ESI)")):
        v = verify_inputs(from_path(case, checks=mode))
        assert any(label in s for s in v.not_assessed), (mode, v.not_assessed)


def test_a_banner_less_case_reads_no_version_and_never_raises(tmp_path):
    case = _case(tmp_path, None, {"100/U.gz": "not gzip at all"})    # unreadable: skipped
    seen = forks.read(NO_BANNER, str(case))
    assert (seen.fork, seen.version, seen.measured, seen.headers_searched) == \
        (None, None, None, True)
    assert forks.not_checked(seen) == (
        "The OpenFOAM fork and version were not read: the solver log has no banner, and no "
        "header of the case's files names one; the gate's error rates were measured on OpenFOAM "
        "v2406 (ESI) only.")
    assert forks.read(NO_BANNER, str(tmp_path / "missing")).fork is None


def test_a_corrupt_compressed_field_is_passed_over_for_the_controldict(tmp_path):
    """A gzip header over data that does not inflate (zlib.error, not an OSError) is a file whose
    bytes were read and did not parse: noted (VER-14's rule, `courant.case.reads.reading`), and
    the next header is read rather than giving up."""
    from courant.case.reads import recording

    case = _case(tmp_path, FOUNDATION_HEADER.format(n="12"))
    (case / "100").mkdir()
    (case / "100" / "U.gz").write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"\xff" * 64)
    with recording(case) as reads:
        seen = forks.read(NO_BANNER, str(case))
    assert (seen.fork, seen.version) == (None, None)
    assert (seen.given.fork, seen.given.version, seen.given.source) == \
        ("foundation", "12", "the header of system/controlDict")
    assert reads.paths() == ["100/U.gz", "system/controlDict"]


# ---------------------------------------------------------------- in the verdict
def _rebanner(tmp_path: Path, sid: str, banner: str | None, name: str = "run") -> tuple[Path, Path]:
    """A corpus run with its banner replaced (or, with None, removed with every header line)."""
    text = gzip.decompress((CORPUS / f"{sid}.log.gz").read_bytes()).decode()
    body = text[text.index("Exec   :"):]
    if banner is None:
        body = body[body.index("Create time"):]
    log, cj = tmp_path / f"{name}.log", tmp_path / f"{name}.case.json"
    log.write_text((banner or "") + body)
    shutil.copy(CORPUS / f"{sid}.case.json", cj)
    return log, cj


def _verdict(log: Path, cj: Path | None, checks: str = "default"):
    return verify_inputs(from_path(log, cj, checks=checks))


def test_a_measured_run_states_its_fork_and_version_and_nothing_else_changes():
    v = verify_path(CORPUS / f"{HEALTHY}.log.gz", CORPUS / f"{HEALTHY}.case.json")
    doc = json.loads(v.to_json())
    assert (doc["run"]["fork"], doc["run"]["solver_version"]) == ("esi", "v2406")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    assert not any("OpenFOAM" in s for s in v.not_assessed)
    assert _verdict(CORPUS / f"{HEALTHY}.log.gz", CORPUS / f"{HEALTHY}.case.json",
                    "next").verdict == "accept"


@pytest.mark.parametrize("banner, fork, version", [
    (esi("2412"), "esi", "v2412"), (esi("2512"), "esi", "v2512"),
    (foundation("11"), "foundation", "11"), (foundation("12"), "foundation", "12"),
], ids=["esi-v2412", "esi-v2512", "foundation-11", "foundation-12"])
def test_by_default_an_unmeasured_version_is_reported_and_its_accept_stands(
        tmp_path, banner, fork, version):
    v = _verdict(*_rebanner(tmp_path, HEALTHY, banner))
    assert (v.verdict, v.rule, v.grade) == ("accept", "VER-5.6", "A")
    assert (v.run.fork, v.run.solver_version) == (fork, version)
    line = forks.not_checked(forks.SolverVersion(fork, version, forks.LOG_BANNER))
    assert line in v.not_assessed and "measured on OpenFOAM v2406 (ESI) only" in line
    assert "checks" not in json.loads(v.to_json())["tool"]


@pytest.mark.parametrize("banner, fork, version", [
    (esi("2412"), "esi", "v2412"), (esi("2512"), "esi", "v2512"),
    (foundation("11"), "foundation", "11"), (foundation("12"), "foundation", "12"),
], ids=["esi-v2412", "esi-v2512", "foundation-11", "foundation-12"])
def test_under_next_ver_5_10_makes_that_accept_insufficient(tmp_path, banner, fork, version):
    v = _verdict(*_rebanner(tmp_path, HEALTHY, banner), "next")
    assert (v.verdict, v.rule, v.grade) == ("insufficient", "VER-5.10", "A")
    [r] = [r for r in v.reasons if r.kind == "insufficient"]
    assert r.code == "insufficient.unsupported_version"
    assert (r.value, r.threshold) == (forks.label(fork, version), "OpenFOAM v2406 (ESI)")
    assert v.exit_code == 3 and json.loads(v.to_json())["tool"]["checks"] == "next"
    assert "measured" in v.next_action and "Point Courant at the solver log" not in v.next_action
    v1 = v.to_v1()
    assert (v1.verdict, v1.rule) == ("NOT_ASSESSABLE", "VER-5.10")
    assert not [r for r in v1.reasons if r.kind == "insufficient"]


def test_under_next_a_reject_on_evidence_stands_whatever_the_version(tmp_path):
    v = _verdict(*_rebanner(tmp_path, BLOWUP, esi("2512")), "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.3")
    assert (v.run.fork, v.run.solver_version) == ("esi", "v2512")


def test_under_next_an_insufficient_keeps_its_own_rule(tmp_path):
    log = tmp_path / "log.checkMesh"
    text = (ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_checkmesh.log").read_text()
    log.write_text(foundation("12") + text[text.index("Exec"):])
    v = _verdict(log, None, "next")
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.5")
    assert (v.run.fork, v.run.solver_version) == ("foundation", "12")


def test_a_log_with_no_banner_says_so_and_its_verdict_does_not_change(tmp_path):
    for checks in ("default", "next"):
        v = _verdict(*_rebanner(tmp_path, HEALTHY, None), checks)
        assert (v.verdict, v.rule) == ("accept", "VER-5.6"), checks
        assert (v.run.fork, v.run.solver_version) == (None, None)
        assert forks.not_checked(forks.SolverVersion()) in v.not_assessed


# ---------------------------------------------------------------- a case directory
@pytest.fixture
def demo(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


def _strip_banner(case: Path) -> None:
    log = case / "log.simpleFoam"
    text = log.read_text()
    log.write_text(text[text.index("Create time"):])


def _set_header(case: Path, header: str | None) -> None:
    path = case / "system" / "controlDict"
    text = path.read_text()
    body = text[text.index("FoamFile"):]
    path.write_text((header or "") + body)


def test_a_case_whose_log_has_a_banner_reads_it_and_its_manifest_is_unchanged(demo):
    v = verify_path(demo)
    assert (v.run.fork, v.run.solver_version, v.verdict) == ("esi", "v2406", "accept")
    assert not any("OpenFOAM" in s for s in v.not_assessed)


def test_under_next_a_banner_less_v2406_run_keeps_its_accept_whatever_its_controldict_header_says(
        demo):
    """The review's case (20 Oct 2026): an ESI v2406 run whose log has no banner (a cut window),
    with no time directory it wrote, and a `controlDict` copied from a Foundation 12 tutorial.
    The header names a version the solver never reported: it is said in the not-checked line,
    `run.fork` stays null, and under `--checks next` the accept stands (VER-5.10 sees an
    unread version, as with no banner)."""
    before = verify_path(demo)
    _strip_banner(demo)
    _set_header(demo, FOUNDATION_HEADER.format(n="12"))
    for checks in ("default", "next"):
        v = verify_path(demo, checks=checks)
        assert (v.verdict, v.rule) == (before.verdict, before.rule) == ("accept", "VER-5.6"), checks
        assert v.exit_code == 0
        assert (v.run.fork, v.run.solver_version) == (None, None)
        assert "system/controlDict" in [i.path for i in v.run.inputs]
        assert CONTROLDICT_NAMES.format(label="OpenFOAM 12 (Foundation)") in v.not_assessed
        assert not [r for r in v.reasons if r.kind == "insufficient"]


def test_a_banner_less_case_is_judged_as_before_and_says_the_version_was_not_read(demo):
    before = verify_path(demo)
    _strip_banner(demo)
    _set_header(demo, None)
    for checks in ("default", "next"):
        v = verify_path(demo, checks=checks)
        assert (v.verdict, v.rule) == (before.verdict, before.rule), checks
        assert (v.run.fork, v.run.solver_version) == (None, None)
        assert forks.not_checked(forks.SolverVersion(headers_searched=True)) in v.not_assessed


def _set_release(case: Path, release: str) -> None:
    """The demo's ESI v2406 banner, renumbered to `release` (`2512`); the run is unchanged."""
    log = case / "log.simpleFoam"
    text = log.read_text()
    at = text.index("Exec")
    log.write_text(text[:at].replace("2406", release) + text[at:])


def _spec_file(where: Path, *items: dict) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    path = where / "spec.json"
    path.write_text(json.dumps({"schema": "courant.spec/1", "items": list(items)}))
    return path


END_TIME = {"name": "end time", "file": "system/controlDict", "key": "endTime", "value": 2000}
UNREADABLE = {"name": "diffusivity", "file": "constant/transportProperties",
              "key": "noSuchKeyAnywhere", "value": 5}


@pytest.mark.parametrize("items, default, nxt", [
    ([{**END_TIME, "value": 1000}], ("reject", "VER-5.7"), ("reject", "VER-5.7")),
    ([END_TIME, UNREADABLE], ("insufficient", "VER-5.9"), ("insufficient", "VER-5.9")),
    ([END_TIME], ("accept", "VER-5.6"), ("insufficient", "VER-5.10")),
], ids=["a-mismatch", "an-unreadable-item", "conforming"])
def test_under_next_the_specification_rules_decide_before_ver_5_10(demo, tmp_path, items, default,
                                                                    nxt):
    """VER-15: the specification's rules (step 4) come before VER-5.10 (step 5), which acts only
    on the accept they leave. On a v2512 run under `next`, a mismatching item is VER-5.7's reject
    and an unreadable item on an otherwise conforming specification VER-5.9's `insufficient`,
    never VER-5.10's; only a specification that conforms leaves an accept for VER-5.10."""
    _set_release(demo, "2512")
    spec = _spec_file(tmp_path / "spec", *items)
    got = {checks: verify_path(demo, spec=spec, checks=checks) for checks in ("default", "next")}
    assert (got["default"].verdict, got["default"].rule) == default
    assert (got["next"].verdict, got["next"].rule) == nxt
    for v in got.values():
        assert (v.run.fork, v.run.solver_version) == ("esi", "v2512")
    codes = [r.code for r in got["next"].reasons if r.kind == "insufficient"]
    assert codes == {"VER-5.9": ["insufficient.spec_value_unreadable"],
                     "VER-5.10": ["insufficient.unsupported_version"]}.get(nxt[1], [])


# ---------------------------------------------------------------- the adapter boundary
def test_the_measured_set_is_defined_in_one_place():
    """The measured set has one definition, `courant.forks.MEASURED`: no other module of the
    package defines a `MEASURED` of its own. That the decision code (the policy, the pipeline,
    the verdict, the sentences) names no fork and no version is ARC-14's lint, in
    `tests/unit/test_architecture_lint.py`, not this test."""
    pkg = ROOT / "courant"
    named = []
    for p in sorted(pkg.rglob("*.py")):
        if p.name == "forks.py" or "__pycache__" in p.parts:
            continue
        if re.search(r"""\bMEASURED\s*[:=][^=]""", p.read_text()):
            named.append(p.relative_to(ROOT).as_posix())
    assert named == []
