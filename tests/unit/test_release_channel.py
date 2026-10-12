"""The installer and the updater must name the same release repository and the same key.

They did not. `courant update` pointed at the private source repository, whose
release assets need authentication no customer has — so the update path would
have 404'd for everyone, and nothing would have noticed until a customer tried
to take a fix.

Seventh instance in this codebase of one fact with two implementations. The
fact now lives in `courant.RELEASE_REPO`; this is what keeps the shell script
honest about it, since a POSIX `sh` installer cannot import Python. The same holds
for the release key (`courant.release.RELEASE_PUBLIC_KEY`): the installer checks the
release's SHA256SUMS against its Ed25519 signature under that key, with python3 and
`cryptography` or with OpenSSL 3, and says so plainly when it has neither.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from courant import RELEASE_REPO
from courant import release as crel
from courant.cli.commands.update import REPO

INSTALLER = Path(__file__).resolve().parents[2] / "packaging" / "install.sh"
SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")      # Ed25519 SubjectPublicKeyInfo


def test_updater_reads_the_one_definition() -> None:
    assert REPO == RELEASE_REPO


def test_installer_names_the_same_repository() -> None:
    assignments = re.findall(r'^REPO="([^"]+)"', INSTALLER.read_text(), re.M)
    assert assignments == [RELEASE_REPO], (
        f"packaging/install.sh names {assignments}, courant.RELEASE_REPO is "
        f"{RELEASE_REPO!r}. A customer would be sent to the wrong repository."
    )


def test_the_release_repository_is_not_the_source_repository() -> None:
    """A private repo's release assets need a token. The installer has none."""
    assert RELEASE_REPO != "shivanshudwivedi/Courant"


# ---------------------------------------------------------------- the installer checks the signature
def _installer_constant(name: str) -> str:
    found = re.findall(rf'^{name}="([^"]+)"', INSTALLER.read_text(), re.M)
    assert len(found) == 1, name
    return found[0]


def test_the_installer_pins_the_release_key():
    assert _installer_constant("RELEASE_PUBLIC_KEY") == crel.RELEASE_PUBLIC_KEY
    spki = base64.b64decode(_installer_constant("RELEASE_PUBLIC_KEY_SPKI"))
    assert spki == SPKI_PREFIX + bytes.fromhex(crel.RELEASE_PUBLIC_KEY)


@pytest.fixture
def release(tmp_path):
    """A release signed with a throwaway test key: (dir, its public key hex)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from courant.signing import public_key_hex

    key = Ed25519PrivateKey.generate()
    d = tmp_path / "release"
    d.mkdir()
    (d / "courant-linux-x86_64").write_bytes(b"binary")
    crel.write_sums(d, "9.9.9")
    sums = (d / crel.SUMS).read_bytes()
    doc = crel.signature_doc(sums, public_key_hex(key), key.sign(sums).hex())
    (d / crel.SUMS_SIG).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return d, public_key_hex(key)


def _sh(script: str, path: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/sh", "-c", f'COURANT_INSTALL_LIB=1 . "{INSTALLER}"; {script}'],
                          env={"PATH": path, "HOME": os.environ.get("HOME", "/tmp"), **env},
                          capture_output=True, text=True, timeout=60)


def _spki(public_hex: str) -> str:
    return base64.b64encode(SPKI_PREFIX + bytes.fromhex(public_hex)).decode()


def _python_path() -> str:
    """This interpreter (which has `cryptography`) first on PATH, as python3."""
    return os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"])


def test_the_installer_verifies_with_python(release):
    d, pub = release
    sums, sig = d / crel.SUMS, d / crel.SUMS_SIG
    run = f'have_python_crypto && verify_sig_python "{sums}" "{sig}" '
    assert _sh(run + pub, _python_path()).returncode == 0
    assert _sh(run + crel.RELEASE_PUBLIC_KEY, _python_path()).returncode == 1   # not the pin
    sums.write_bytes(sums.read_bytes() + b"0" * 64 + b"  extra\n")             # tampered
    assert _sh(run + pub, _python_path()).returncode == 1


def _openssl3() -> str | None:
    for cand in (shutil.which("openssl"), "/opt/homebrew/opt/openssl@3/bin/openssl",
                 "/usr/local/opt/openssl@3/bin/openssl", "/usr/bin/openssl"):
        if cand and Path(cand).is_file():
            out = subprocess.run([cand, "version"], capture_output=True, text=True).stdout
            if out.startswith("OpenSSL 3"):
                return cand
    return None


@pytest.mark.skipif(_openssl3() is None, reason="no OpenSSL 3 on this machine")
def test_the_installer_verifies_with_openssl(release):
    d, pub = release
    sums, sig = d / crel.SUMS, d / crel.SUMS_SIG
    path = os.pathsep.join([str(Path(_openssl3()).parent), "/usr/bin", "/bin"])
    run = f'have_openssl3 && verify_sig_openssl "{sums}" "{sig}" '
    assert _sh(run + f'{pub} {_spki(pub)}', path).returncode == 0
    pin = crel.RELEASE_PUBLIC_KEY
    assert _sh(run + f'{pin} {_spki(pin)}', path).returncode == 1              # not the pin
    sums.write_bytes(sums.read_bytes() + b"0" * 64 + b"  extra\n")
    assert _sh(run + f'{pub} {_spki(pub)}', path).returncode == 1


def test_the_installer_refuses_a_release_not_signed_by_the_pinned_key(release):
    d, _ = release
    r = _sh(f'verify_release "{d / crel.SUMS}" "{d / crel.SUMS_SIG}"', _python_path())
    assert r.returncode == 1 and "does not verify" in r.stderr


def test_without_a_verifier_it_says_only_the_checksum_was_checked(release, tmp_path):
    """No python3 with `cryptography` and no OpenSSL 3: it goes on, and says exactly what was
    not verified and how to check it by hand, unless COURANT_REQUIRE_SIGNATURE=1."""
    d, _ = release
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("sed", "awk", "head", "cut", "grep", "mktemp", "rm", "cat", "shasum",
                 "sha256sum"):
        found = shutil.which(name, path="/usr/bin:/bin")
        if found:
            (tools / name).symlink_to(found)
    run = f'verify_release "{d / crel.SUMS}" "{d / crel.SUMS_SIG}"'
    r = _sh(run, str(tools))
    assert r.returncode == 0, r.stderr
    assert "Not verified: the release signature" in r.stderr
    assert "only the binary's SHA-256" in r.stderr and "openssl pkeyutl -verify" in r.stderr
    assert _sh(run, str(tools), COURANT_REQUIRE_SIGNATURE="1").returncode == 1
