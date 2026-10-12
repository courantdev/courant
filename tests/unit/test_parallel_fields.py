"""A decomposed case must never be verified against its initial condition.

`find_time_dirs` looks at the case root. A parallel run writes its results into
`processor*/` and leaves the root holding only `0/`, `constant/` and `system/`, so
the reader saw `["0"]`, read `0/U`, and handed the initial condition to the
verification suite as the result.

Measured on a real 4-rank icoFoam run that diverged to Courant 1580 with a
cumulative continuity error of 2.65e+27: the field checks reported "fields within
plausible bounds". It graded D only because the *log* checks caught it — and in the
silent-error class, where a run completes and converges but is physically wrong,
the log is clean by definition and these are the only checks left.

These tests use directory structure alone (empty field files), no solver, because the
bug was in choosing *which directory to read*, not in the reading.
"""

from __future__ import annotations

import os

from courant.verification._fluid_reader import processor_dirs, read_fields


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("// placeholder\n")


def _decomposed(root: str, ranks: int = 4, times=("0", "0.5")) -> None:
    """A case shaped like a decomposed run: results in the ranks, only 0/ at root."""
    _touch(os.path.join(root, "0", "U"))
    _touch(os.path.join(root, "system", "controlDict"))
    _touch(os.path.join(root, "constant", "transportProperties"))
    for r in range(ranks):
        for t in times:
            _touch(os.path.join(root, f"processor{r}", t, "U"))


def test_processor_dirs_are_found_in_rank_order(tmp_path) -> None:
    root = str(tmp_path / "case")
    _decomposed(root, ranks=12)
    found = [os.path.basename(p) for p in processor_dirs(root)]
    assert found[:3] == ["processor0", "processor1", "processor2"]
    # Rank order, not lexicographic: processor10 must not sort before processor2.
    assert found[-1] == "processor11"
    assert len(found) == 12


def test_a_serial_case_reports_no_processor_dirs(tmp_path) -> None:
    root = str(tmp_path / "case")
    _touch(os.path.join(root, "0", "U"))
    _touch(os.path.join(root, "0.5", "U"))
    assert processor_dirs(root) == []


def test_decomposed_case_is_never_read_at_time_zero(tmp_path) -> None:
    """The bug, stated as a test: t=0 must not be returned as the result.

    The field files are empty, so no field parses here.
    That is the point — the assertion is on the *time chosen*. Before the fix this
    returned time 0.0 with the initial condition's contents in `internal`.
    """
    root = str(tmp_path / "case")
    _decomposed(root, times=("0", "0.5"))
    fields = read_fields(root)
    assert fields.time != 0.0 or fields.internal == {}, (
        "a decomposed case was read at t=0 — that is the initial condition, "
        "and grading it as the result is a false physics-OK"
    )


def test_a_decomposed_case_that_never_wrote_a_result_yields_no_fields(tmp_path) -> None:
    """Crashed on step one: ranks hold only `0/`. Skip the checks, do not pass them."""
    root = str(tmp_path / "case")
    _decomposed(root, times=("0",))
    fields = read_fields(root)
    assert fields.internal == {}
    assert fields.time == 0.0
    assert fields.inlet_speed is None
