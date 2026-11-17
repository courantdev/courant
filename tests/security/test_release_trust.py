"""Security review of 0.7.0: release integrity (research/notes/security-review-070/FINDINGS.md).

Each test names its finding. A finding still open is a strict xfail, so it turns into a failure
the day it is fixed and the marker must go.
"""

from __future__ import annotations

import os

import pytest

from courant import release as crel


def _attacker_release(version: str = "9.9.9"):
    """(sums bytes, signature doc) signed by a key that is not the release key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    sums = crel.sums_text({"courant-linux-x86_64": b"EVIL"}, version).encode()
    # The doc names the pinned key, so the "different key" check passes; only the signature
    # itself can stop it.
    doc = crel.signature_doc(sums, crel.RELEASE_PUBLIC_KEY, key.sign(sums).hex())
    return key, sums, doc


def test_sec070_1_a_file_named_after_the_pinned_key_cannot_replace_it(tmp_path, monkeypatch):
    """SEC070-1: `public_bytes` read its argument as a path when a file of that name existed, so
    a file in the working directory named after the pinned key's hex supplied the key."""
    key, sums, doc = _attacker_release()
    from courant.signing import public_key_hex

    (tmp_path / crel.RELEASE_PUBLIC_KEY).write_text(public_key_hex(key) + "\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(crel.ReleaseError, match="does not verify"):
        crel.verify_asset("courant-linux-x86_64", b"EVIL", sums, doc, current="0.7.0")


def test_sec070_1_verify_bytes_takes_hex_only(tmp_path):
    from courant.signing import verify_bytes

    key, sums, _ = _attacker_release()
    from courant.signing import public_key_hex

    keyfile = tmp_path / "k"
    keyfile.write_text(public_key_hex(key))
    assert verify_bytes(sums, key.sign(sums).hex(), public_key_hex(key))
    assert not verify_bytes(sums, key.sign(sums).hex(), str(keyfile))


def test_sec070_4_the_update_writes_through_no_planted_symlink(tmp_path):
    """SEC070-4: `_replace` copied the old binary to `<name>.old` with shutil.copy2, which
    follows a symlink planted there and overwrites its target."""
    from courant.cli.commands.update import _replace

    binary = tmp_path / "courant"
    binary.write_bytes(b"OLDBIN")
    os.chmod(binary, 0o755)
    victim = tmp_path / "victim"
    victim.write_bytes(b"precious")
    (tmp_path / "courant.old").symlink_to(victim)
    _replace(binary, b"NEWBIN")
    assert binary.read_bytes() == b"NEWBIN"
    assert victim.read_bytes() == b"precious"


def test_sec070_2_the_action_installs_only_a_signed_binary():
    """SEC070-2: the installer goes on without a signature check when the machine has no
    verifier; the Action, which runs a trusted copy of the installer, must refuse instead."""
    from pathlib import Path

    import yaml

    action = Path(__file__).resolve().parents[2] / ".github/actions/courant-gate/action.yml"
    steps = yaml.safe_load(action.read_text())["runs"]["steps"]
    install = next(s for s in steps if s.get("name") == "Install Courant")
    assert str(install.get("env", {}).get("COURANT_REQUIRE_SIGNATURE")) == "1"
