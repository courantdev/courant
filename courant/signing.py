"""Ed25519 signatures: the one implementation of signing and verifying (TRD OPS-7, SEC-5).

A release's checksums (`courant.release`) are signed with an Ed25519 key that is held outside
every repository and checked against the public key pinned in the build. The private key is read
from the file a path or an environment variable names, at the moment of signing, refused if it
lies inside a repository, and never logged, printed or copied: only its public half is written.

`cryptography` is imported only inside the functions that need it.
"""

from __future__ import annotations

import os
from pathlib import Path


class SignatureError(ValueError):
    """A key or a signature that cannot be used. The message never contains key material."""


def repository_root(path: str | Path) -> Path | None:
    """The repository `path` lies in, if any (a parent holding `.git`, file or directory)."""
    p = Path(path).expanduser().resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return d
    return None


def load_private(path: str | Path | None, env: str):
    """The Ed25519 private key at `path`, else at the file the variable `env` names."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    where = path or os.environ.get(env)
    if not where:
        raise SignatureError(f"no signing key: set {env} to the key file (H-SIGNING-KEY)")
    repo = repository_root(where)
    if repo is not None:
        raise SignatureError(f"refusing to read a signing key inside the repository at {repo}")
    where = Path(where).expanduser().resolve()
    try:
        key = load_pem_private_key(where.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        # Never echo the file's content: only that it could not be read as a key.
        raise SignatureError(f"the signing key at {where} could not be read as an Ed25519 "
                             "private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureError("the signing key is not an Ed25519 key")
    return key


def public_key_hex(private_key) -> str:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def public_bytes(public_key: str | Path) -> bytes:
    """A published public key (32 bytes as hex, or a short file holding that) as raw bytes."""
    text = str(public_key)
    p = Path(text)
    if len(text) < 200 and p.is_file():
        text = p.read_text().strip()
    try:
        raw = bytes.fromhex(text.strip())
    except ValueError as exc:
        raise SignatureError("a public key is 32 bytes, hex") from exc
    if len(raw) != 32:
        raise SignatureError("a public key is 32 bytes, hex")
    return raw


def sign_with(key, data: bytes) -> tuple[str, str]:
    """(public key hex, signature hex) of `data` under a loaded private key."""
    return public_key_hex(key), key.sign(data).hex()


def sign_bytes(data: bytes, *, key_path: str | Path | None = None,
               env: str) -> tuple[str, str]:
    """(public key hex, signature hex) of `data` under the key `key_path` or `env` names."""
    return sign_with(load_private(key_path, env), data)


def verify_bytes(data: bytes, signature_hex: str, public_key: str | Path) -> bool:
    """Whether `signature_hex` signs `data` under the published public key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_bytes(public_key)).verify(
            bytes.fromhex(signature_hex), data)
    except (InvalidSignature, ValueError, SignatureError):
        return False
    return True


__all__ = ["SignatureError", "load_private", "public_bytes", "public_key_hex",
           "repository_root", "sign_bytes", "sign_with", "verify_bytes"]
