"""Real-world case shapes, and the one contract that holds across all of them.

A customer's case will not look like a tutorial. It may be decomposed two
different ways, written in binary, split into regions, produced by the other
fork of OpenFOAM, or launched through a scheduler. Courant does not need to
extract full value from every one of those — but it must never extract the
*wrong* value, because a check that passes on data it should not have read is a
false "physics-OK", and zero of those is the invariant this product is sold on.

So the contract these tests hold is deliberately narrow and absolute:

    Given any case shape, `read_fields` returns either the real result or
    nothing. Never the initial condition, never a partial rank, never t=0.

Each test names the shape it stands for, because the failure mode is silent and
the next person to touch this file needs to know what is being protected.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pytest

from courant.case.solvers import honours_adjust_time_step, is_steady
from courant.verification._fluid_reader import processor_dirs, read_fields

FIELD = ("FoamFile{version 2.0;format ascii;class volVectorField;object U;}\n"
         "internalField uniform (%s 0 0);\n")


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


# --------------------------------------------------------------------------
# Decomposition layouts
# --------------------------------------------------------------------------

def test_uncollated_decomposition_is_found(tmp_path) -> None:
    """`fileHandler uncollated` — the default. One directory per rank."""
    case = str(tmp_path)
    _write(os.path.join(case, "0", "U"), FIELD % 0)
    for rank in range(4):
        _write(os.path.join(case, f"processor{rank}", "0", "U"), FIELD % 0)
        _write(os.path.join(case, f"processor{rank}", "0.5", "U"), FIELD % 1)
    assert len(processor_dirs(case)) == 4
    assert read_fields(case).time == 0.5


def test_collated_decomposition_is_found(tmp_path) -> None:
    """`fileHandler collated` — one `processors<N>/` for every rank.

    Common on large runs, because 4096 ranks otherwise means 4096 directories.
    Before this was handled the reader saw a serial case, found only `0/` at the
    root, and returned nothing — safe, but it lost verification entirely on a
    configuration a serious user is likely to have chosen.
    """
    case = str(tmp_path)
    _write(os.path.join(case, "0", "U"), FIELD % 0)
    _write(os.path.join(case, "processors4", "0", "U"), FIELD % 0)
    _write(os.path.join(case, "processors4", "0.5", "U"), FIELD % 1)
    assert [os.path.basename(p) for p in processor_dirs(case)] == ["processors4"]
    assert read_fields(case).time == 0.5


def test_uncollated_wins_when_both_layouts_are_present(tmp_path) -> None:
    """A case re-decomposed with a different handler keeps the stale directory."""
    case = str(tmp_path)
    for rank in range(2):
        _write(os.path.join(case, f"processor{rank}", "0.5", "U"), FIELD % 1)
    _write(os.path.join(case, "processors2", "0.5", "U"), FIELD % 1)
    names = [os.path.basename(p) for p in processor_dirs(case)]
    assert names == ["processor0", "processor1"], (
        "per-rank files are readable and collated ones may not be; prefer them"
    )


# --------------------------------------------------------------------------
# The contract: never the initial condition, never t=0
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["uncollated", "collated", "serial", "empty"])
def test_a_case_that_never_wrote_a_result_yields_nothing(tmp_path, shape) -> None:
    """Crashed on step one. Every layout must skip, not pass on `0/`."""
    case = str(tmp_path)
    _write(os.path.join(case, "0", "U"), FIELD % 0)
    if shape == "uncollated":
        for rank in range(4):
            _write(os.path.join(case, f"processor{rank}", "0", "U"), FIELD % 0)
    elif shape == "collated":
        _write(os.path.join(case, "processors4", "0", "U"), FIELD % 0)

    fields = read_fields(case)
    assert fields.internal == {}, f"{shape}: graded the initial condition"
    assert fields.time == 0.0
    assert fields.inlet_speed is None


def test_a_binary_case_skips_rather_than_guesses(tmp_path) -> None:
    """`writeFormat binary` with no data. Refusing is the right answer."""
    case = str(tmp_path)
    _write(os.path.join(case, "0.5", "U"),
           "FoamFile{version 2.0;format binary;class volVectorField;object U;}\n")
    fields = read_fields(case)
    assert fields.internal == {}
    assert fields.time == 0.5 and fields.inlet_speed is None


def _binary(cls: str, obj: str, kind: str, values: np.ndarray, arch: str | None = None) -> bytes:
    arch_line = f'    arch "{arch}";\n' if arch is not None else ""
    return (f"FoamFile\n{{\n    version 2.0;\n    format binary;\n{arch_line}    class {cls};\n"
            f"    object {obj};\n}}\n").encode() + \
        f"internalField nonuniform List<{kind}> {len(values)}(".encode() + \
        values.tobytes() + b");\nboundaryField\n{\n}\n"


def test_a_binary_field_is_read_byte_for_byte(tmp_path) -> None:
    """`writeFormat binary` with data. 8.0 is `0x4020000000000000`: its bytes hold `(` and
    `@`, which fluidfoam took for structure and read as 3.25. A wrong value is worse than none."""
    u = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p = np.array([1.5, 2.5, -3.0])
    os.makedirs(tmp_path / "0.5")
    (tmp_path / "0.5" / "U").write_bytes(_binary("volVectorField", "U", "vector", u))
    (tmp_path / "0.5" / "p").write_bytes(_binary("volScalarField", "p", "scalar", p))
    fields = read_fields(str(tmp_path))
    assert fields.internal["U"].tolist() == u.tolist()
    assert fields.internal["p"].tolist() == p.tolist()


def test_a_compressed_field_is_read(tmp_path) -> None:
    """`writeCompression on` writes `U.gz`; it is the same field."""
    os.makedirs(tmp_path / "0.5")
    with gzip.open(tmp_path / "0.5" / "U.gz", "wt") as fh:
        fh.write("FoamFile{version 2.0;format ascii;class volVectorField;object U;}\n"
                 "internalField nonuniform List<vector> 2((1 2 3)(4 5 6));\nboundaryField{}\n")
    assert read_fields(str(tmp_path)).internal["U"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_a_uniform_field_is_one_value_per_field(tmp_path) -> None:
    """`uniform` is one value for every cell: (1,) for a scalar, (1, 3) for a vector."""
    _write(os.path.join(str(tmp_path), "0.5", "U"), FIELD % 2)
    _write(os.path.join(str(tmp_path), "0.5", "p"),
           "FoamFile{version 2.0;format ascii;class volScalarField;object p;}\n"
           "internalField uniform 3;\n")
    fields = read_fields(str(tmp_path))
    assert fields.internal["U"].tolist() == [[2.0, 0.0, 0.0]]
    assert fields.internal["p"].tolist() == [3.0]
    assert fields.inlet_speed == 2.0


def test_the_field_reader_needs_only_the_core_dependencies(tmp_path, monkeypatch) -> None:
    """Every install reads fields the same way. The reader parses with foamlib, a core
    dependency in the release lock and the binary; fluidfoam (once the reader's parser, never
    in the binary) is blocked here, and the field is still read."""
    import sys

    monkeypatch.setitem(sys.modules, "fluidfoam", None)
    _write(os.path.join(str(tmp_path), "0.5", "U"), FIELD % 1)
    assert "U" in read_fields(str(tmp_path)).internal


def test_a_multi_region_case_does_not_invent_a_field(tmp_path) -> None:
    """chtMultiRegionFoam puts fields under `<time>/<region>/`, not `<time>/`.

    Nothing reads them yet. What matters is that the root time directory, which
    holds no field files at all, does not produce a field set.
    """
    case = str(tmp_path)
    _write(os.path.join(case, "0.5", "fluid", "U"), FIELD % 1)
    _write(os.path.join(case, "0.5", "solid", "T"), FIELD % 300)
    assert read_fields(case).internal == {}


# --------------------------------------------------------------------------
# The other fork, and the scheduler
# --------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "foamRun -solver incompressibleFluid",
    "mpirun -np 64 foamRun -solver compressibleFluid -parallel",
    "srun foamMultiRun -parallel",
])
def test_the_openfoam_org_fork_is_recognised(command) -> None:
    """v11 replaced every solver executable with `foamRun -solver <module>`.

    Unrecognised, it defaulted to "does not read adjustTimeStep" — so the
    timestep repair proposed a fixed step to every user of that fork.
    """
    assert honours_adjust_time_step(command), command


def test_steadiness_of_a_foamrun_case_comes_from_its_schemes() -> None:
    """`foamRun` says nothing about steadiness; `ddtSchemes` does.

    `CaseConfig.is_steady` prefers the declared scheme and only falls back to the
    solver name, which is the correct precedence — so `foamRun` is deliberately
    absent from `STEADY_SOLVERS` rather than guessed at.
    """
    assert not is_steady("foamRun -solver incompressibleFluid")


# --------------------------------------------------------------------------
# A check that cannot fail must not claim to have passed
# --------------------------------------------------------------------------

def test_plausibility_does_not_claim_a_velocity_check_it_cannot_make() -> None:
    """`inlet_speed` is the 95th percentile of the velocity field itself.

    So the old spike test compared a distribution's maximum against its own p95,
    a ratio that sits near 1 whatever the values are, and could never reach its
    10x threshold. Measured on a decomposed cavity run that had diverged:
    max|U| 14.7 against a lid moving at 1 m/s — a true overspeed of 14.7x — and
    the check computed 1.27 and reported "fields within plausible bounds".

    The detail line must now say what was actually verified. An affirmative
    printed on the strength of a comparison that cannot fail is the false
    reassurance this product exists to remove.
    """
    import numpy as np

    from courant.verification import VerifyContext
    from courant.verification.checks import PlausibilityCheck
    from courant.verification.fields import FieldData

    # A velocity field 14.7x the reference, with the same p95 artefact.
    mag = np.linspace(1.0, 14.7, 400)
    u = np.stack([mag, np.zeros_like(mag), np.zeros_like(mag)], axis=1)
    fields = FieldData(time=7.0, internal={"U": u}, nu=None,
                       inlet_speed=float(np.percentile(mag, 95)))

    result = PlausibilityCheck().run(
        VerifyContext(run_id="x", events=[], config=None, fields=fields))

    assert "plausible bounds" not in (result.detail or ""), (
        "claimed the fields are plausible on the strength of a check it did not run"
    )
    assert "boundary reference" in (result.detail or "")


# --------------------------------------------------------------------------
# Binary fields: read only what the header says can be read as written
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arch,dtype,readable", [
    ("LSB;label=32;scalar=64", "<f8", True),
    ("LSB;label=64;scalar=64", "<f8", True),
    (None, "<f8", True),                              # no arch: OpenFOAM's native default
    ("MSB;label=32;scalar=64", ">f8", False),         # big-endian: never byte-swapped by guess
    ("LSB;label=32;scalar=32", "<f4", False),         # single precision: the reader would guess
    ("garbage", "<f8", False),
])
def test_a_binary_field_is_read_only_when_its_arch_says_how(tmp_path, arch, dtype, readable):
    """foamlib reads binary data in the host's byte order and guesses the scalar size from the
    byte count; the header's `arch` is ignored. A field written another way is left unread,
    never read as different numbers."""
    p = np.array([1.5, 2.5, -3.0], dtype=dtype)
    os.makedirs(tmp_path / "0.5")
    (tmp_path / "0.5" / "p").write_bytes(_binary("volScalarField", "p", "scalar", p, arch))
    fields = read_fields(str(tmp_path))
    if readable:
        assert fields.internal["p"].tolist() == [1.5, 2.5, -3.0]
    else:
        assert fields.internal == {}


# --------------------------------------------------------------------------
# Plausibility: a non-finite value is never "non-negative"
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["k", "epsilon", "omega", "T", "p", "nut"])
@pytest.mark.parametrize("bad", ["nan", "-nan", "inf", "-inf"])
def test_a_non_finite_field_value_is_a_major_finding(tmp_path, name, bad):
    """`np.min` of an array holding nan is nan, and `nan < 0` is False, so a k field of nan
    read as "k non-negative". OpenFOAM writes `nan` and `-nan` when a run has blown up."""
    from courant.verification.checks import PlausibilityCheck, VerifyContext
    from courant.verification.grade import VSeverity

    _write(os.path.join(str(tmp_path), "0.5", name),
           f"FoamFile{{version 2.0;format ascii;class volScalarField;object {name};}}\n"
           f"internalField nonuniform List<scalar> 3(0.1 {bad} 0.2);\n")
    fields = read_fields(str(tmp_path))
    assert name in fields.internal
    res = PlausibilityCheck().run(VerifyContext(run_id="r", events=[], fields=fields))
    assert res.severity is VSeverity.MAJOR and not res.passed and not res.skipped
    assert "non-finite" in res.detail and name in res.detail


def test_a_non_finite_velocity_is_a_major_finding(tmp_path):
    from courant.verification.checks import PlausibilityCheck, VerifyContext
    from courant.verification.grade import VSeverity

    _write(os.path.join(str(tmp_path), "0.5", "U"),
           "FoamFile{version 2.0;format ascii;class volVectorField;object U;}\n"
           "internalField nonuniform List<vector> 2((1 0 0)(nan 0 0));\n")
    res = PlausibilityCheck().run(VerifyContext(run_id="r", events=[],
                                                fields=read_fields(str(tmp_path))))
    assert res.severity is VSeverity.MAJOR and not res.passed and "U" in res.detail


def test_an_empty_field_is_not_a_crash_and_not_a_pass(tmp_path):
    """`nonuniform List<scalar> 0()`: nothing to bound. np.min of it raised ValueError."""
    from courant.verification.checks import PlausibilityCheck, VerifyContext

    _write(os.path.join(str(tmp_path), "0.5", "k"),
           "FoamFile{version 2.0;format ascii;class volScalarField;object k;}\n"
           "internalField nonuniform List<scalar> 0();\n")
    fields = read_fields(str(tmp_path))
    res = PlausibilityCheck().run(VerifyContext(run_id="r", events=[], fields=fields))
    assert res.skipped or not res.passed
    assert "non-negative" not in (res.detail or "")
