#!/usr/bin/env sh
# Courant installer — attached to every release of github.com/courantdev/courant
#
#     curl -fsSL https://github.com/courantdev/courant/releases/latest/download/install.sh | sh
#
# POSIX sh, not bash: this has to run on HPC login nodes, minimal containers,
# and whatever the customer's cluster considers /bin/sh. No arrays, no [[ ]], no
# `local` outside functions.
#
# Design rules, in order of how much they matter:
#
#   1. **Never need root.** Installs to ~/.local/bin. An engineer who has to
#      open a ticket to try your tool does not try your tool.
#   2. **Verify what was downloaded.** The release's SHA256SUMS is checked
#      against its Ed25519 signature under the release key pinned below, and the
#      binary against SHA256SUMS, before anything is made executable. The
#      signature needs python3 with the `cryptography` package or OpenSSL 3; with
#      neither, the installer says that only the checksum was checked, and how to
#      check the signature by hand.
#   3. **Fail loudly and early.** Partial installs are worse than none: they
#      leave a `courant` on PATH that does not work.
#   4. **Say what happened.** Including the PATH line, if we just installed
#      somewhere that is not on it.

set -eu

# The public repository of the open-source gate: its releases carry the binaries,
# their checksums and this script (courant.RELEASE_REPO; tests/unit/test_release_channel.py).
REPO="courantdev/courant"
# The release public key (Ed25519), the one courant/release.py pins: 32 bytes as hex, and the same
# key as base64 SubjectPublicKeyInfo DER for OpenSSL (tests/unit/test_release_channel.py).
RELEASE_PUBLIC_KEY="0493639167582138b1e28f188354301ff23c6c881cd064c778a2db5d5c821192"
RELEASE_PUBLIC_KEY_SPKI="MCowBQYDK2VwAyEABJNjkWdYITix4o8Yg1QwH/I8bIgc0GTHeKLbXVyCEZI="
VERSION="${COURANT_VERSION:-latest}"
INSTALL_DIR="${COURANT_INSTALL_DIR:-$HOME/.local/bin}"
BINARY="courant"

red()  { printf '\033[31m%s\033[0m\n' "$1" >&2; }
bold() { printf '\033[1m%s\033[0m\n' "$1"; }
dim()  { printf '\033[2m%s\033[0m\n' "$1"; }

die() {
    red "$1"
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || die "This installer needs '$1', which is not on PATH."
}

# --- what are we installing on? ---------------------------------------------

detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)  os="linux" ;;
        Darwin) os="darwin" ;;
        *)
            die "Courant does not ship a binary for $os yet.
Supported today: Linux (x86_64, arm64) and macOS (arm64).
Write to hello@courant.dev and say what you are running."
            ;;
    esac

    case "$arch" in
        x86_64 | amd64)  arch="x86_64" ;;
        arm64 | aarch64) arch="arm64" ;;
        *)
            die "Courant does not ship a binary for $arch yet.
Supported today: Linux (x86_64, arm64) and macOS (arm64).
Write to hello@courant.dev and say what you are running."
            ;;
    esac

    printf '%s-%s' "$os" "$arch"
}

download() {
    # curl on macOS, wget on the minimal Linux images that ship without it.
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" "$1"
    else
        die "This installer needs curl or wget, and neither is on PATH."
    fi
}

verify_checksum() {
    # file="$1" expected="$2"
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$1" | cut -d' ' -f1)"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$1" | cut -d' ' -f1)"
    else
        red "Warning: no sha256sum or shasum available — cannot verify the download."
        red "Set COURANT_SKIP_VERIFY=1 to install anyway."
        [ "${COURANT_SKIP_VERIFY:-}" = "1" ] || exit 1
        return 0
    fi

    [ "$actual" = "$2" ] || die "Checksum mismatch — refusing to install.
  expected  $2
  actual    $actual
This means the download was corrupted or tampered with. Please report it:
  https://github.com/$REPO/issues"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        return 1
    fi
}

# --- the release signature (OPS-7) -------------------------------------------
# SHA256SUMS.sig.json (courant.release-signature/1) holds the Ed25519 signature of the exact bytes
# of SHA256SUMS. It is checked against the key pinned above, never the key the file names.

json_field() {
    # the value of "name": "value" in the pretty-printed signature file (one key per line)
    sed -n "s/^ *\"$1\": *\"\([^\"]*\)\".*/\1/p" "$2" | head -n 1
}

have_python_crypto() {
    command -v python3 >/dev/null 2>&1 \
        && python3 -c "import cryptography.hazmat.primitives.asymmetric.ed25519" >/dev/null 2>&1
}

have_openssl3() {
    command -v openssl >/dev/null 2>&1 && openssl version 2>/dev/null | grep -q '^OpenSSL 3'
}

verify_sig_python() {
    # sums="$1" sigfile="$2" public_key_hex="$3"
    python3 - "$1" "$2" "$3" <<'PY'
import hashlib, json, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
sums, doc, key = open(sys.argv[1], "rb").read(), json.load(open(sys.argv[2])), sys.argv[3]
ok = (doc.get("schema") == "courant.release-signature/1" and doc.get("algorithm") == "ed25519"
      and doc.get("signed_sha256") == hashlib.sha256(sums).hexdigest()
      and str(doc.get("public_key", "")).lower() == key.lower())
if ok:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(
            bytes.fromhex(doc["signature"]), sums)
    except Exception:
        ok = False
sys.exit(0 if ok else 1)
PY
}

hex_to_bin() {
    # hex="$1" out="$2": lowercase hex to bytes, with awk and printf only
    esc="$(printf '%s' "$1" | awk '{ h = "0123456789abcdef"; s = "";
        for (i = 1; i <= length($0); i += 2)
            s = s sprintf("\\%03o", (index(h, substr($0, i, 1)) - 1) * 16 + index(h, substr($0, i + 1, 1)) - 1);
        printf "%s", s }')"
    # shellcheck disable=SC2059 — the escapes are the format
    printf "$esc" > "$2"
}

verify_sig_openssl() {
    # sums="$1" sigfile="$2" public_key_hex="$3" public_key_spki_base64="$4"
    sig="$(json_field signature "$2")"
    printf '%s' "$sig" | grep -Eq '^[0-9a-f]{128}$' || return 1
    [ "$(json_field public_key "$2")" = "$3" ] || return 1
    [ "$(json_field signed_sha256 "$2")" = "$(sha256_of "$1")" ] || return 1
    work="$(mktemp -d)"
    printf -- '-----BEGIN PUBLIC KEY-----\n%s\n-----END PUBLIC KEY-----\n' "$4" > "$work/key.pem"
    hex_to_bin "$sig" "$work/sig.bin"
    if openssl pkeyutl -verify -pubin -inkey "$work/key.pem" -rawin -in "$1" \
            -sigfile "$work/sig.bin" >/dev/null 2>&1; then
        rm -rf "$work"
        return 0
    fi
    rm -rf "$work"
    return 1
}

verify_release() {
    # sums="$1" sigfile="$2": refuses on a bad signature; says so when it cannot check one
    if have_python_crypto; then
        verify_sig_python "$1" "$2" "$RELEASE_PUBLIC_KEY" || die "The release signature does not verify — refusing to install.
SHA256SUMS is not signed by the Courant release key. Please report it:
  https://github.com/$REPO/issues"
        dim "  signature verified (Ed25519, release key ${RELEASE_PUBLIC_KEY%"${RELEASE_PUBLIC_KEY#????????????????}"}…, python3)"
    elif have_openssl3; then
        verify_sig_openssl "$1" "$2" "$RELEASE_PUBLIC_KEY" "$RELEASE_PUBLIC_KEY_SPKI" \
            || die "The release signature does not verify — refusing to install.
SHA256SUMS is not signed by the Courant release key. Please report it:
  https://github.com/$REPO/issues"
        dim "  signature verified (Ed25519, release key ${RELEASE_PUBLIC_KEY%"${RELEASE_PUBLIC_KEY#????????????????}"}…, OpenSSL)"
    else
        red "Not verified: the release signature. This machine has neither python3 with the"
        red "'cryptography' package nor OpenSSL 3, so only the binary's SHA-256 against the"
        red "release's SHA256SUMS was checked, and SHA256SUMS itself was not authenticated."
        red "To check it by hand on a machine with OpenSSL 3, download SHA256SUMS and"
        red "SHA256SUMS.sig.json from the same release and run:"
        red "  printf -- '-----BEGIN PUBLIC KEY-----\\n$RELEASE_PUBLIC_KEY_SPKI\\n-----END PUBLIC KEY-----\\n' > courant-release.pem"
        red "  sed -n 's/.*\"signature\": *\"\\([0-9a-f]*\\)\".*/\\1/p' SHA256SUMS.sig.json | xxd -r -p > sig.bin"
        red "  openssl pkeyutl -verify -pubin -inkey courant-release.pem -rawin -in SHA256SUMS -sigfile sig.bin"
        red "and compare the binary's SHA-256 with its line in SHA256SUMS."
        [ "${COURANT_REQUIRE_SIGNATURE:-}" != "1" ] \
            || die "COURANT_REQUIRE_SIGNATURE=1 is set: refusing to install without a verified signature."
    fi
}

# --- install ----------------------------------------------------------------

main() {
    need uname
    need mktemp

    target="$(detect_target)"
    bold "Installing Courant for $target"

    if [ "$VERSION" = "latest" ]; then
        base="https://github.com/$REPO/releases/latest/download"
    else
        base="https://github.com/$REPO/releases/download/v${VERSION#v}"
    fi

    asset="courant-$target"
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064 — expand $tmp now, not at trap time
    trap "rm -rf '$tmp'" EXIT INT TERM

    dim "  downloading $asset"
    download "$base/$asset" "$tmp/$BINARY" \
        || die "Could not download $base/$asset
If this is a new platform or a version that does not exist, check:
  https://github.com/$REPO/releases"

    dim "  verifying the release's signed checksums"
    download "$base/SHA256SUMS" "$tmp/SHA256SUMS" \
        && download "$base/SHA256SUMS.sig.json" "$tmp/SHA256SUMS.sig.json" \
        || die "This release has no signed checksums (SHA256SUMS, SHA256SUMS.sig.json).
Refusing to install an unverified binary."
    verify_release "$tmp/SHA256SUMS" "$tmp/SHA256SUMS.sig.json"
    if [ "$VERSION" != "latest" ]; then
        named="$(sed -n '1s/^# courant-dev //p' "$tmp/SHA256SUMS")"
        [ "$named" = "${VERSION#v}" ] || die "Asked for ${VERSION#v}; the signed release is ${named:-unnamed}.
Refusing to install."
    fi
    expected="$(awk -v a="$asset" '$2 == a { print $1 }' "$tmp/SHA256SUMS")"
    [ -n "$expected" ] || die "The signed SHA256SUMS does not list $asset. Refusing to install."
    verify_checksum "$tmp/$BINARY" "$expected"

    # Run it before installing it. A binary that cannot start on this machine
    # should not end up on PATH — finding that out at install time is a message,
    # finding out at 2am mid-solve is an outage.
    chmod +x "$tmp/$BINARY"
    "$tmp/$BINARY" version >/dev/null 2>&1 \
        || die "The downloaded binary does not run on this machine.
Please report it with the output of 'uname -a':
  https://github.com/$REPO/issues"

    mkdir -p "$INSTALL_DIR"
    # mv within one filesystem is atomic, so an interrupted install cannot leave
    # a half-written `courant` on PATH. Fall back to cp when $TMPDIR is on a
    # different mount, which is normal on a cluster.
    mv "$tmp/$BINARY" "$INSTALL_DIR/$BINARY" 2>/dev/null \
        || { cp "$tmp/$BINARY" "$INSTALL_DIR/$BINARY" && chmod +x "$INSTALL_DIR/$BINARY"; }

    installed="$("$INSTALL_DIR/$BINARY" version 2>/dev/null || echo courant)"
    printf '\n'
    bold "Installed $installed"
    dim  "  $INSTALL_DIR/$BINARY"

    case ":$PATH:" in
        *":$INSTALL_DIR:"*)
            printf '\nGet started:\n'
            printf '  courant demo\n'
            printf '  courant verify /path/to/your/case\n'
            ;;
        *)
            printf '\n'
            red "$INSTALL_DIR is not on your PATH."
            printf 'Add it, then reload your shell:\n\n'
            printf '  echo '\''export PATH="%s:$PATH"'\'' >> ~/.profile\n' "$INSTALL_DIR"
            printf '  . ~/.profile\n\n'
            printf 'Then:\n'
            printf '  courant demo\n'
            printf '  courant verify /path/to/your/case\n'
            ;;
    esac
}

# Sourcing with COURANT_INSTALL_LIB=1 defines the functions and installs nothing (the tests).
[ "${COURANT_INSTALL_LIB:-}" = "1" ] || main "$@"
