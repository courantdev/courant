"""Real defects the static-typing triage (group MISC) found, each reproduced; xfail until fixed."""

from __future__ import annotations

import pytest

from courant.spec.check import check_conformance

_BLOCK_MESH = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1) );
"""


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="TYPES-MISC-1: a cells item with a scalar value crashes conforms()")
def test_types_misc_1_scalar_cells_value_does_not_crash(tmp_path):
    # `courant.spec/1` accepts `blocks: "cells"` with a scalar `value` (SpecItem.value is
    # float | tuple | str). Such an item is `exact`, and `conforms` then iterates the float:
    # `[float(x) for x in want]` -> TypeError, so `courant spec check` (and `courant verify
    # --spec`) end in "internal error (TypeError)" (exit 4, fail-closed) instead of a
    # conformance block or a clean SpecError (cli/guard.py: a bad input is the command's to
    # report). If the fix rejects a scalar cells value at load instead, rewrite this test.
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "blockMeshDict").write_text(_BLOCK_MESH)
    spec = {"schema": "courant.spec/1",
            "items": [{"name": "cells", "file": "system/blockMeshDict", "blocks": "cells",
                       "value": 400}]}
    block = check_conformance(spec, tmp_path, spec_sha256="0" * 64)
    assert len(block.items) == 1
    assert block.items[0].conforms is not False        # 20 x 20 x 1 = 400 cells
