#!/usr/bin/env bash
# Smoke-test a packaged Courant agent (Tech Bible §16, Phase B.4).
#
# Runs the real product loop through the real binary: a case directory, a log
# written live by a *separate* process, no network, no account, and — crucially
# — no Python on PATH. That last part is the whole point of shipping a binary,
# and it is the one thing `pytest` cannot check, because pytest is Python.
#
#     ./packaging/test_binary.sh dist/agent/courant
#
# Exits non-zero on the first failure. Wired into the release workflow so a
# broken binary cannot be published.

set -euo pipefail

BINARY="${1:-dist/agent/courant}"
if [[ ! -x "$BINARY" ]]; then
    echo "not an executable: $BINARY" >&2
    echo "build it first:  pyinstaller packaging/courant.spec --clean --noconfirm" >&2
    exit 2
fi
BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
check() {
    local what="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  ok    $what"
        pass=$((pass + 1))
    else
        echo "  FAIL  $what" >&2
        exit 1
    fi
}

echo "Testing $BINARY"
echo

# --- 1. it runs at all, with nothing inherited from this shell --------------
# `env -i` is the real test: no PYTHONPATH, no VIRTUAL_ENV, no PATH to a
# Python. If the binary needs anything from the build machine, it dies here.
check "runs with an empty environment" \
    env -i HOME="$WORK" PATH=/usr/bin:/bin "$BINARY" version
check "prints its own version, not (dev)" \
    bash -c "env -i HOME='$WORK' PATH=/usr/bin:/bin '$BINARY' version | grep -qE '[0-9]+\.[0-9]+\.[0-9]+'"
check "reports itself as packaged" \
    bash -c "env -i HOME='$WORK' PATH=/usr/bin:/bin '$BINARY' version | grep -q packaged"
check "help lists the verifier's commands" \
    bash -c "'$BINARY' --help | grep -q convergence && '$BINARY' --help | grep -q verify"

# --- 2. the server's dependencies are genuinely absent ----------------------
# If these are in the bundle the excludes in courant.spec have stopped working,
# and the agent is carrying a web server to an HPC login node.
for module in fastapi sqlalchemy uvicorn anthropic; do
    if "$BINARY" --help 2>&1 | grep -qi "$module"; then
        echo "  FAIL  $module leaked into the agent bundle" >&2
        exit 1
    fi
done
echo "  ok    server dependencies excluded from the bundle"
pass=$((pass + 1))

# --- 3. the v0.3 product, from packaged data only ----------------------------
# Run from an empty directory with an empty environment: nothing may come from the repository.
cd "$WORK"
check "--about reads the packaged certification" \
    bash -c "env -i HOME='$WORK' PATH=/usr/bin:/bin '$BINARY' verify --about | grep -q 'false accepts 0 of 23'"
check "demo verifies three real runs offline" \
    bash -c "env -i HOME='$WORK' PATH=/usr/bin:/bin '$BINARY' demo --keep '$WORK/demo' > '$WORK/demo.out' && grep -q '1,023 of 1,999 past convergence' '$WORK/demo.out'"
# (Not `demo | grep -q`: grep exits at the first match and the demo, still writing, dies of
# SIGPIPE before it writes the report the next check reads. That raced on CI's macOS runners.)
check "demo wrote a self-contained HTML report" test -s "$WORK/demo/report.html"
check "preflight catches the broken setup before the run" \
    bash -c "'$BINARY' preflight x --case-json '$WORK/demo/icoFoam_cavity__transport_units/icoFoam_cavity__transport_units.case.json'; test \$? -eq 2"
check "the audit totals the demo folder" \
    bash -c "'$BINARY' audit '$WORK/demo' | grep -q 'Untrusted'"
check "convergence on the demo case finds its stopping point" \
    bash -c "'$BINARY' convergence '$WORK/demo/pitzDaily_fixed_budget' | grep -q 'stopped changing at step 977'"
check "verify on a case directory carries the convergence block" \
    bash -c "'$BINARY' verify '$WORK/demo/pitzDaily_fixed_budget' --json | grep -q '\"converged_at_step\":977'"
check "the evidence pack re-derives byte for byte" \
    "$BINARY" verify --check-pack "$WORK/demo/pack" --root "$WORK/demo/pitzDaily_fixed_budget"
check "watch --once reads a finished run and ends with its verdict" \
    "$BINARY" watch "$WORK/demo/pitzDaily_fixed_budget" --once
check "the Courant Test runs from the packaged tasks" \
    bash -c "'$BINARY' test run --judge courant --split all | grep -q 'false accepts 0/23'"
check "a broken run exits 2" \
    bash -c "'$BINARY' verify '$WORK/demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.log.gz' --case-json '$WORK/demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.case.json'; test \$? -eq 2"
check "mcp starts, or names the missing extra without a traceback" \
    bash -c "'$BINARY' mcp --root '$WORK' </dev/null 2>&1 | grep -qv Traceback"
cd - >/dev/null

# --- 4. `courant watch` follows a log another process is writing --------------
CASE="$WORK/cavity"
cp -r "$ROOT/tests/fixtures/mini_case" "$CASE"

# A separate process writes the log while the binary follows it — the same
# arrangement as a real solve, and the only way to exercise the tailer.
(
    sleep 1
    exec 3>"$CASE/log.icoFoam"
    for i in 1 2 3 4 5; do
        printf 'Time = 0.0%d\n' "$i" >&3
        printf 'smoothSolver:  Solving for p, Initial residual = 1e-0%d, Final residual = 1e-9, No Iterations 12\n' "$((i + 2))" >&3
        printf 'ExecutionTime = %d.0 s  ClockTime = %d s\n\n' "$i" "$i" >&3
        sleep 0.1
    done
    for i in 1 2 3 4 5; do
        printf 'Time = 0.1%d\n' "$i" >&3
        printf 'Courant Number mean: %d0.0 max: %d00.0\n' "$i" "$i" >&3
        printf 'smoothSolver:  Solving for p, Initial residual = 1e+0%d, Final residual = 1e-1, No Iterations 200\n' "$i" >&3
        printf 'ExecutionTime = %d.0 s  ClockTime = %d s\n\n' "$((i + 5))" "$i" >&3
        sleep 0.1
    done
    exec 3>&-
) &
WRITER=$!

OUTPUT="$WORK/watch.out"
set +e
env -i HOME="$WORK" PATH=/usr/bin:/bin \
    "$BINARY" watch "$CASE" --exit-on-fail --interval 0.2 --idle-timeout 3 \
    --start-timeout 10 >"$OUTPUT" 2>&1
code=$?
set -e
wait "$WRITER" 2>/dev/null || true

grep -q "is watching" "$OUTPUT" || { echo "  FAIL  never started watching"; cat "$OUTPUT"; exit 1; }
echo "  ok    followed a log written by another process"
grep -q "FAILING" "$OUTPUT"     || { echo "  FAIL  never flagged the divergence"; cat "$OUTPUT"; exit 1; }
[[ $code -eq 2 ]]               || { echo "  FAIL  --exit-on-fail exited $code, not 2"; cat "$OUTPUT"; exit 1; }
echo "  ok    flagged the divergence and exited 2 for the job script"
pass=$((pass + 2))

# --- 5. the binary is the open-source gate, and nothing else ------------------
# The private repository's commands (The Bench, private evaluations, the previous product)
# are not in the bundle (packaging/courant.spec builds from pyproject's public packages).
for command in bench eval copilot diagnose demo-fix fix-bench revert; do
    if "$BINARY" "$command" --help >/dev/null 2>&1; then
        echo "  FAIL  the private command '$command' is in the binary" >&2
        exit 1
    fi
done
echo "  ok    no private command in the binary"
pass=$((pass + 1))

echo
echo "$pass checks passed — $BINARY is shippable."
