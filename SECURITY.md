# Security

Please report a vulnerability privately, through GitHub's private vulnerability reporting on
this repository (Security, then "Report a vulnerability"), or by email to hello@courant.dev.
Do not open a public issue for it. Include the version (`courant version`), the platform and
the smallest input that shows the problem.

What counts: a verdict that a crafted input can turn to `accept` when it should not; a command
that reads outside the directories it was given, writes where it was not told, or opens a
network connection that is not its documented purpose; a release whose signature check can be
bypassed by `courant update` or the installer.

Releases are signed: every release carries `SHA256SUMS` and its Ed25519 signature
(`SHA256SUMS.sig.json`) under the release key, whose public half is pinned in
`courant/release.py` and in `packaging/install.sh`.

- `courant update` installs nothing unless the signature verifies against the key built into the
  binary, the signed `SHA256SUMS` lists the downloaded file with its hash, and the version it
  names is the one asked for.
- The installer (`install.sh`) checks the binary's SHA-256 against the release's `SHA256SUMS`,
  and the signature of `SHA256SUMS` against the pinned key when python3 with the `cryptography`
  package or OpenSSL 3 is available; it refuses on a mismatch. With neither, it installs after
  the checksum alone and says so, with the commands to check the signature by hand;
  `COURANT_REQUIRE_SIGNATURE=1` makes it refuse instead.
