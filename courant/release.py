"""A release's checksums and their signature (TRD OPS-4, OPS-7).

A release directory holds its artefacts, `SHA256SUMS` (one `<sha256>  <name>` line per artefact,
sorted by name) and `SHA256SUMS.sig.json` (`courant.release-signature/1`): the Ed25519 signature
of the exact bytes of `SHA256SUMS`, under the release key (H-SIGNING-KEY). The release workflow
writes them; `courant update` checks them before it replaces anything. One format, one reader:
this module.

The signature is checked against the public key pinned in this build (`RELEASE_PUBLIC_KEY`), never
against the key a signature file names. A build with no key pinned refuses to install: an update
it cannot verify is not an update.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SIG_SCHEMA = "courant.release-signature/1"
SUMS = "SHA256SUMS"
SUMS_SIG = "SHA256SUMS.sig.json"
#: The release public key, Ed25519, 32 bytes as hex (H-SIGNING-KEY, created 11 Oct 2026; base64
#: `BJNjkWdYITix4o8Yg1QwH/I8bIgc0GTHeKLbXVyCEZI=`). Its private half never enters a repository.
RELEASE_PUBLIC_KEY: str | None = "0493639167582138b1e28f188354301ff23c6c881cd064c778a2db5d5c821192"
#: The variable naming the file that holds the release's private key, where it is signed.
KEY_ENV = "COURANT_RELEASE_SIGNING_KEY"

_LINE = re.compile(r"([0-9a-f]{64}) [ *]([^\s/\\][^/\\]*)")
_VERSION = re.compile(r"# courant-dev (\d+\.\d+\.\d+)")


class ReleaseError(ValueError):
    """A release that cannot be trusted. The message says why, in one line."""


def sums_text(files: dict[str, bytes], version: str) -> str:
    """`SHA256SUMS` for these artefacts {name: bytes}, sorted by name, after a first line
    `# courant-dev <version>`: the signature covers the version, so an older signed release
    cannot be served as a newer one (OPS-7)."""
    return f"# courant-dev {version}\n" + "".join(
        f"{hashlib.sha256(files[n]).hexdigest()}  {n}\n" for n in sorted(files))


def sums_version(text: str) -> str | None:
    """The release version `SHA256SUMS` names on its first line, or None."""
    first = text.splitlines()[0].strip() if text.strip() else ""
    m = _VERSION.fullmatch(first)
    return m.group(1) if m else None


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(x) for x in str(v).strip().removeprefix("v").split("."))
    except ValueError:
        raise ReleaseError(f"{v!r} is not a version MAJOR.MINOR.PATCH") from None
    if len(parts) != 3:
        raise ReleaseError(f"{v!r} is not a version MAJOR.MINOR.PATCH")
    return parts


def check_version(signed: str | None, requested: str, current: str,
                  allow_older: bool = False) -> str:
    """The signed release's version, once it is what was asked for. Under `latest` it must be
    strictly above this agent's; an explicit version must equal it, and one below this agent's
    installs only with `allow_older` (`courant update --allow-downgrade`)."""
    if signed is None:
        raise ReleaseError(f"the signed {SUMS} names no version")
    if requested == "latest":
        if _version_tuple(signed) <= _version_tuple(current):
            raise ReleaseError(f"the release served as latest is {signed}, not newer than this "
                               f"agent ({current})")
        return signed
    if _version_tuple(signed) != _version_tuple(requested):
        raise ReleaseError(f"asked for {requested.removeprefix('v')}, the signed release is "
                           f"{signed}")
    if _version_tuple(signed) < _version_tuple(current) and not allow_older:
        raise ReleaseError(f"{signed} is older than this agent ({current}); pass "
                           "--allow-downgrade to install it")
    return signed


def parse_sums(text: str) -> dict[str, str]:
    """{artefact name: sha256} from `SHA256SUMS`. A malformed line or a name given twice is an
    error, never skipped; the first line may name the version."""
    out: dict[str, str] = {}
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip() or (i == 1 and _VERSION.fullmatch(line.strip())):
            continue
        m = _LINE.fullmatch(line.strip())
        if not m:
            raise ReleaseError(f"{SUMS} line {i} is not '<sha256>  <name>'")
        if m.group(2) in out:
            raise ReleaseError(f"{SUMS} names {m.group(2)} twice")
        out[m.group(2)] = m.group(1)
    if not out:
        raise ReleaseError(f"{SUMS} lists no artefacts")
    return out


def signature_doc(sums: bytes, public_key_hex: str, signature_hex: str) -> dict:
    return {"schema": SIG_SCHEMA, "algorithm": "ed25519", "signed": SUMS,
            "signed_sha256": hashlib.sha256(sums).hexdigest(),
            "public_key": public_key_hex, "signature": signature_hex}


def verify_sums(sums: bytes, doc: dict, public_key: str | None) -> dict[str, str]:
    """The checksums, once their signature verifies under `public_key`; ReleaseError otherwise."""
    from courant.signing import verify_bytes

    if not public_key:
        raise ReleaseError("no release public key is pinned in this build (H-SIGNING-KEY), "
                           "so no release can be verified")
    if not isinstance(doc, dict) or doc.get("schema") != SIG_SCHEMA:
        raise ReleaseError(f"{SUMS_SIG} is not {SIG_SCHEMA}")
    if doc.get("algorithm") != "ed25519":
        raise ReleaseError(f"{SUMS_SIG} is not an Ed25519 signature")
    if doc.get("signed_sha256") != hashlib.sha256(sums).hexdigest():
        raise ReleaseError(f"{SUMS} is not the file {SUMS_SIG} signed")
    if str(doc.get("public_key", "")).lower() != public_key.lower():
        raise ReleaseError("signed with a different key than the pinned release key")
    if not verify_bytes(sums, str(doc.get("signature", "")), public_key):
        raise ReleaseError("the signature does not verify")
    return parse_sums(sums.decode())


def verify_asset(name: str, payload: bytes, sums: bytes, doc: dict,
                 public_key: str | None = RELEASE_PUBLIC_KEY, *, requested: str = "latest",
                 current: str | None = None, allow_older: bool = False) -> str:
    """The asset's SHA-256, once the signed checksums name it with exactly that hash and name
    the version that was asked for (`check_version`)."""
    from courant import __version__

    listed = verify_sums(sums, doc, public_key)
    check_version(sums_version(sums.decode()), requested, current or __version__, allow_older)
    if name not in listed:
        raise ReleaseError(f"the signed {SUMS} does not list {name}")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != listed[name]:
        raise ReleaseError(f"{name} does not match the signed {SUMS}: expected "
                           f"{listed[name]}, got {actual}")
    return actual


def write_sums(dist: str | Path, version: str) -> Path:
    """SHA256SUMS over every artefact in `dist`, its first line naming the release's version, so
    the signature binds the version too (OPS-7: no replayed older release)."""
    dist = Path(dist)
    files = {p.name: p.read_bytes() for p in sorted(dist.iterdir())
             if p.is_file() and p.name not in (SUMS, SUMS_SIG)}
    path = dist / SUMS
    path.write_text(sums_text(files, version))
    return path


def sign_sums(dist: str | Path, key_path: str | Path | None = None,
              variable: str = KEY_ENV) -> tuple[Path, str]:
    """Sign `dist`'s SHA256SUMS with the key `key_path` (or the file `variable` names) points
    at; returns (the signature file, the public key hex). The key is never printed."""
    import json

    from courant.signing import sign_bytes

    dist = Path(dist)
    sums = (dist / SUMS).read_bytes()
    public, sig = sign_bytes(sums, key_path=key_path, env=variable)
    path = dist / SUMS_SIG
    path.write_text(json.dumps(signature_doc(sums, public, sig), indent=2, sort_keys=True) + "\n")
    return path, public


def verify_directory(directory: str | Path, public_key: str | None) -> dict[str, str]:
    """Every artefact in a release directory against its signed checksums."""
    import json

    d = Path(directory)
    sums = (d / SUMS).read_bytes()
    listed = verify_sums(sums, json.loads((d / SUMS_SIG).read_text()), public_key)
    for name, want in listed.items():
        p = d / name
        if not p.is_file():
            raise ReleaseError(f"{name} is listed in {SUMS} and missing")
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise ReleaseError(f"{name} does not match {SUMS}")
    return listed


__all__ = [
    "KEY_ENV",
    "RELEASE_PUBLIC_KEY",
    "SIG_SCHEMA",
    "SUMS",
    "SUMS_SIG",
    "ReleaseError",
    "check_version",
    "parse_sums",
    "sign_sums",
    "signature_doc",
    "sums_text",
    "sums_version",
    "verify_asset",
    "verify_directory",
    "verify_sums",
    "write_sums",
]
