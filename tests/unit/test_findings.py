"""Unit tests for the Finding contract."""

import pytest
from pydantic import ValidationError

from courant.diagnosis.findings import Finding, FindingSet, FindingSeverity


def _f(id="cfl.max", checker="cfl", sev=FindingSeverity.CRITICAL, suggests=None, **kw):
    return Finding(
        id=id, checker=checker, title="t", severity=sev,
        suggests=suggests if suggests is not None else ["NUM.CFL_VIOLATION"], **kw,
    )


class TestFinding:
    def test_builds_and_is_frozen(self):
        f = _f(value=87.3, threshold=1.0, field="p")
        assert f.value == 87.3
        with pytest.raises(ValidationError):
            f.value = 1.0

    def test_rejects_unknown_suggested_code(self):
        with pytest.raises(ValidationError):
            _f(suggests=["NOPE.NOPE"])

    def test_severity_rank_order(self):
        assert FindingSeverity.CRITICAL.rank > FindingSeverity.WARN.rank
        assert FindingSeverity.WARN.rank > FindingSeverity.INFO.rank


class TestFindingSet:
    def test_id_uniqueness_enforced(self):
        fs = FindingSet([_f(id="a")])
        with pytest.raises(ValueError):
            fs.add(_f(id="a"))

    def test_get_and_ids(self):
        fs = FindingSet([_f(id="a"), _f(id="b")])
        assert fs.get("a") is not None
        assert fs.get("missing") is None
        assert fs.ids == {"a", "b"}
        assert len(fs) == 2

    def test_suggested_codes_weighted_by_severity(self):
        fs = FindingSet([
            _f(id="a", sev=FindingSeverity.CRITICAL, suggests=["NUM.CFL_VIOLATION"]),
            _f(id="b", sev=FindingSeverity.INFO, suggests=["NUM.OVER_RELAXATION"]),
            _f(id="c", sev=FindingSeverity.WARN, suggests=["NUM.OVER_RELAXATION"]),
        ])
        codes = fs.suggested_codes()
        # CFL: weight 3; OVER_RELAXATION: 1+2=3 -> tie, both present
        assert set(codes[:2]) == {"NUM.CFL_VIOLATION", "NUM.OVER_RELAXATION"}

    def test_by_checker_and_severity_filters(self):
        fs = FindingSet([
            _f(id="a", checker="cfl", sev=FindingSeverity.CRITICAL),
            _f(id="b", checker="mesh", sev=FindingSeverity.INFO,
               suggests=["MESH.SKEWNESS"]),
        ])
        assert len(fs.by_checker("cfl")) == 1
        assert len(fs.by_severity(FindingSeverity.WARN)) == 1  # only the critical one
        assert fs.max_severity() is FindingSeverity.CRITICAL

    def test_empty_set_max_severity_is_info(self):
        assert FindingSet().max_severity() is FindingSeverity.INFO
