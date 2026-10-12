"""Unit tests for the failure taxonomy."""

import pytest

from courant.taxonomy import (
    TAXONOMY,
    EscalationTier,
    FailureClass,
    all_codes,
    by_class,
    get,
)


def test_has_25_entries():
    assert len(TAXONOMY) == 25


def test_codes_unique_and_namespaced():
    codes = all_codes()
    assert len(codes) == len(set(codes))
    for code in codes:
        assert "." in code  # CLASS.NAME namespacing
        prefix = code.split(".")[0]
        assert prefix in {"NUM", "MESH", "BC", "PHYS", "CONV", "ENV"}


def test_lookup_and_fields():
    e = get("NUM.CFL_VIOLATION")
    assert e.failure_class is FailureClass.NUMERICS
    assert e.tier is EscalationTier.NUMERICS
    assert e.summary


def test_unknown_code_raises():
    with pytest.raises(KeyError):
        get("NOPE.NOT_REAL")


def test_by_class_partitions_all_entries():
    total = sum(len(by_class(fc)) for fc in FailureClass)
    assert total == len(TAXONOMY)
    assert len(by_class(FailureClass.NUMERICS)) == 6
