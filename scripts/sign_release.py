#!/usr/bin/env python
"""Write a release's SHA256SUMS and sign it with the release key (OPS-7).

    python scripts/sign_release.py DIR [--version X.Y.Z]

Writes `DIR/SHA256SUMS` (its first line names the version, so the signature binds it) over every
artefact in DIR, and `DIR/SHA256SUMS.sig.json`, the Ed25519 signature of those exact bytes under
the key in the file COURANT_RELEASE_SIGNING_KEY names. The key file must lie outside every
repository; it is never printed. Prints the public key, which must be the one pinned in
`courant/release.py` for `courant update` to accept the release, and refuses (exit 1) when it is
not. The version defaults to `courant.__version__`.

The public repository's release workflow runs this in its `sign` job (environment `release`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courant import __version__  # noqa: E402
from courant.release import (  # noqa: E402
    KEY_ENV,
    RELEASE_PUBLIC_KEY,
    SUMS_SIG,
    sign_sums,
    verify_directory,
    write_sums,
)
from courant.signing import SignatureError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dist", type=Path, help="the directory holding the release's artefacts")
    ap.add_argument("--version", default=__version__)
    ap.add_argument("--key", default=None, help=f"the key file (default: ${KEY_ENV})")
    ap.add_argument("--dry-run", action="store_true",
                    help="a throwaway key may sign (a dry run of the workflow); the release it "
                         "signs is not one `courant update` accepts")
    args = ap.parse_args(argv)
    if not args.dist.is_dir():
        print(f"sign_release: no such directory {args.dist}", file=sys.stderr)
        return 2
    write_sums(args.dist, args.version)
    try:
        _, public = sign_sums(args.dist, key_path=args.key)
    except SignatureError as exc:
        print(f"sign_release: {exc}", file=sys.stderr)
        return 1
    pinned = RELEASE_PUBLIC_KEY is not None and public.lower() == RELEASE_PUBLIC_KEY.lower()
    if not pinned and not args.dry_run:
        (args.dist / SUMS_SIG).unlink()
        print("sign_release: the key is not the release key pinned in courant/release.py, so "
              "`courant update` would refuse this release; nothing is signed", file=sys.stderr)
        return 1
    listed = verify_directory(args.dist, public)
    print(f"signed {len(listed)} artefacts of courant-dev {args.version} with {public}"
          + ("" if pinned else " (a dry run: not the release key)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
