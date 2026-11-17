#!/bin/sh
# GATE-9, installed in a minute: each published install route (pipx, uvx, the installer, the
# binary with its checksum) from nothing to a verdict on the demo, timed, in a fresh HOME.
#
#     bash scripts/install_smoke.sh                         # every route, 3 runs, 60 s budget
#     bash scripts/install_smoke.sh --route binary --runs 1 --out result.json
#
# The harness is scripts/install_smoke.py: standard library only, so any python3 runs it on a
# machine that has never seen Courant. Exit 0 only when every run of every route passed in
# under the budget; 1 otherwise; 2 on a usage error. The clean-runner measurement is
# .github/workflows/install-smoke.yml in the public repository.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
command -v python3 >/dev/null 2>&1 || { echo "install_smoke: python3 is not on PATH" >&2; exit 1; }
exec python3 "$here/install_smoke.py" "$@"
