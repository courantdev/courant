"""The pinned release key (H-SIGNING-KEY, OPS-7): a real Ed25519 public key, and the one
`courant update` checks against.

A release signed by any other key is refused while the pin is in place, and the same release is
accepted only when that key's public half is the pin: the pin is used, not a placeholder.
"""

from __future__ import annotations

import base64

import pytest

from courant import release as crel

#: The public key S published on 11 Oct 2026, as base64 of the raw 32 bytes.
PUBLISHED = "BJNjkWdYITix4o8Yg1QwH/I8bIgc0GTHeKLbXVyCEZI="
NEWER = "99.0.0"


@pytest.fixture
def test_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.generate()          # a throwaway test key, never S's


def _signed_release(tmp_path, key, version=NEWER):
    from courant.signing import public_key_hex

    d = tmp_path / "release"
    d.mkdir()
    (d / "courant-darwin-arm64").write_bytes(b"new binary")
    crel.write_sums(d, version)
    sums = (d / crel.SUMS).read_bytes()
    doc = crel.signature_doc(sums, public_key_hex(key), key.sign(sums).hex())
    return d, sums, doc


def test_the_pin_is_the_published_key_and_decodes_to_an_ed25519_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    assert crel.RELEASE_PUBLIC_KEY is not None
    raw = bytes.fromhex(crel.RELEASE_PUBLIC_KEY)
    assert len(raw) == 32
    assert raw == base64.b64decode(PUBLISHED)
    Ed25519PublicKey.from_public_bytes(raw)          # a valid point, or this raises


def test_a_release_signed_by_another_key_is_refused_under_the_pin(tmp_path, test_key):
    _, sums, doc = _signed_release(tmp_path, test_key)
    with pytest.raises(crel.ReleaseError, match="different key"):
        crel.verify_sums(sums, doc, crel.RELEASE_PUBLIC_KEY)
    # the same bytes verify when, and only when, the test key's public half is the pin
    assert crel.verify_sums(sums, doc, doc["public_key"])


def _update(monkeypatch, tmp_path, signed, pin):
    from typer.testing import CliRunner

    from courant.cli.commands import update
    from courant.cli.main import app

    served = {n: (signed / n).read_bytes()
              for n in ("courant-darwin-arm64", crel.SUMS, crel.SUMS_SIG)}
    monkeypatch.setattr(update, "is_packaged", lambda: True)
    monkeypatch.setattr(update, "target_triple", lambda: "darwin-arm64")
    monkeypatch.setattr(update, "_download", lambda url: served[url.rsplit("/", 1)[1]])
    if pin is not None:
        monkeypatch.setattr(crel, "RELEASE_PUBLIC_KEY", pin)
    binary = tmp_path / "bin" / "courant"
    binary.parent.mkdir(exist_ok=True)
    binary.write_bytes(b"old binary")
    binary.chmod(0o755)
    monkeypatch.setattr(update.sys, "executable", str(binary))
    return CliRunner().invoke(app, ["update"]), binary


def test_update_uses_the_pin(monkeypatch, tmp_path, test_key):
    import json

    signed, _, doc = _signed_release(tmp_path, test_key)
    (signed / crel.SUMS_SIG).write_text(json.dumps(doc))
    res, binary = _update(monkeypatch, tmp_path, signed, None)       # the real pin
    assert res.exit_code == 1 and binary.read_bytes() == b"old binary", res.output
    res, binary = _update(monkeypatch, tmp_path, signed, doc["public_key"])
    assert res.exit_code == 0, res.output
    assert binary.read_bytes() == b"new binary"
