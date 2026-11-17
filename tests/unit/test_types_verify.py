"""Defects the static type check found on the verdict path (TYPES-VER-n), each pinned by a test
that asserts the correct behaviour and is expected to fail until the defect is fixed."""

from __future__ import annotations

import pytest

from courant.verify import support


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="TYPES-VER-1: load_verdict calls .get on JSON that is not an object")
def test_types_ver_1_json_array_is_refused_as_not_a_verdict(tmp_path):
    # `courant doctor --bundle OUT --verdict FILE` -> support.build -> load_verdict. A verdict
    # file that is valid JSON but not an object must be refused like any other non-verdict
    # (ValueError "is not a Courant verdict"), not end in an internal AttributeError.
    verdict = tmp_path / "verdict.json"
    verdict.write_text("[]")
    with pytest.raises(ValueError, match="is not a Courant verdict"):
        support.build(["courant", "doctor"], verdict)
