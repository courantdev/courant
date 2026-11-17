"""A log, a sidecar or a specification that cannot be read is refused with `InputError` (TST-9:
"no input makes `verify_path` raise anything but `InputError`"; GATE-10: exit 4 with a sentence),
never another exception (research/notes/invariants-070/FINDINGS.md, INV6-F2).

Shrunk from `tests/invariants/test_inv6_inputs.py`'s mutants: a `.log.gz` that is not gzip, is
cut, or is corrupted inside (`BadGzipFile`, `EOFError`, `zlib.error` escaped `from_log`); a
sidecar or specification that is not UTF-8 (`UnicodeDecodeError`), nests deeper than the JSON
decoder recurses (`RecursionError`), holds an integer longer than Python converts (`ValueError`),
or is JSON of the wrong shape (`AttributeError`, pydantic's `ValidationError`).
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from courant.verify import InputError, from_log, verify_path, with_spec

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification/corpus"
RUN = "icoFoam_cavity__healthy"


def _log(tmp_path: Path) -> bytes:
    return gzip.decompress((CORPUS / f"{RUN}.log.gz").read_bytes())


@pytest.mark.parametrize("damage", ["not_gzip", "cut", "corrupt"])
def test_a_damaged_gzip_log_is_an_input_error(tmp_path, damage):
    z = gzip.compress(_log(tmp_path), mtime=0)
    data = {"not_gzip": b"\x00\x00plain bytes", "cut": z[: len(z) // 2],
            "corrupt": z[:40] + bytes(b ^ 0x5A for b in z[40:80]) + z[80:]}[damage]
    p = tmp_path / f"{RUN}.log.gz"
    p.write_bytes(data)
    with pytest.raises(InputError, match="could not be read"):   # main's one wording (VER-14)
        verify_path(p)


@pytest.mark.parametrize("text", [
    b"\xff\xfe{}",                                     # not UTF-8
    b"[" * 100_000 + b"]" * 100_000,                   # deeper than the decoder recurses
    b'{"config": "a string"}',                         # a view that is not an object
    b'{"config": {}, "mesh": [1, 2]}',
    b'["a", "list"]',
    b'{"scenario_id": 7}',
    b'{"mesh": {"n_negative_volume_cells": 1' + b"0" * 400 + b'}}',   # past a double: inv6
], ids=["not_utf8", "deep", "config_str", "mesh_list", "list", "id_int", "int_past_double"])
def test_a_sidecar_that_cannot_be_read_is_an_input_error(tmp_path, text):
    log = tmp_path / f"{RUN}.log"
    log.write_bytes(_log(tmp_path))
    side = tmp_path / f"{RUN}.case.json"
    side.write_bytes(text)
    with pytest.raises(InputError):
        from_log(log, side)


@pytest.mark.parametrize("text", [
    b"\x90\x00not utf-8",
    b"[" * 100_000 + b"]" * 100_000,
    b'{"schema": "courant.spec/1", "items": [{"name": "n", "file": "f", "key": "k", "value": '
    + b"9" * 5000 + b"}]}",                             # longer than int() converts
], ids=["not_utf8", "deep", "huge_int"])
def test_a_specification_that_cannot_be_read_is_an_input_error(tmp_path, text):
    log = tmp_path / f"{RUN}.log"
    log.write_bytes(_log(tmp_path))
    spec = tmp_path / "spec.json"
    spec.write_bytes(text)
    with pytest.raises(InputError):
        with_spec(from_log(log), spec)


def test_a_good_sidecar_and_specification_still_read(tmp_path):
    log = tmp_path / f"{RUN}.log"
    log.write_bytes(_log(tmp_path))
    side = CORPUS / f"{RUN}.case.json"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": "courant.spec/1", "items": [
        {"name": "nu", "file": "constant/transportProperties", "key": "nu", "value": 0.01}]}))
    inputs = with_spec(from_log(log, side), spec)
    assert inputs.config_view is not None and inputs.spec is not None


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_a_specification_json_reads_in_any_encoding_json_detects(tmp_path, encoding):
    """What `json.loads` reads from bytes (RFC 8259's detection: a UTF-8 BOM, UTF-16, UTF-32)
    is read as before the INV6-F2 fix; `courant spec init > spec.json` in a Windows shell
    writes UTF-16."""
    log = tmp_path / f"{RUN}.log"
    log.write_bytes(_log(tmp_path))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": "courant.spec/1", "items": [
        {"name": "nu", "file": "constant/transportProperties", "key": "nu", "value": 0.01}]}),
        encoding=encoding)
    assert with_spec(from_log(log), spec).spec is not None


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_a_specification_that_cannot_be_opened_is_an_input_error(tmp_path):
    log = tmp_path / f"{RUN}.log"
    log.write_bytes(_log(tmp_path))
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    spec.chmod(0)
    try:
        with pytest.raises(InputError, match="could not be read"):
            with_spec(from_log(log), spec)
    finally:
        spec.chmod(0o600)


def test_a_specification_read_error_is_an_input_error(tmp_path, monkeypatch):
    """The read itself failing (EIO, a permission root does not see): `SpecError`, so
    `InputError`, as for the sidecar."""
    log = tmp_path / f"{RUN}.log"
    log.write_bytes(_log(tmp_path))
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    real = Path.read_bytes

    def failing(self):
        if self.name == "spec.json":
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", failing)
    with pytest.raises(InputError, match="could not be read"):
        with_spec(from_log(log), spec)
