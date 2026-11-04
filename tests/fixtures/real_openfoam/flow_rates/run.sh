#!/bin/bash
# OpenFOAM's own patch sums of phi, the reference T-MASSFLUX-2's tests hold Courant's reader to.
# Runs inside opencfd/openfoam-default:2406 with a scratch directory mounted at /case that holds
# unpacked copies (every .gz unzipped) of the five committed solves, under these names:
#   wedge_pipe_flow    tests/fixtures/real_openfoam/wedge_pipe/flow
#   heated_cavity      tests/fixtures/real_openfoam/heated_cavity/run
#   laminar_cylinder   tests/fixtures/real_openfoam/laminar_cylinder/run
#   duct_converged     tests/fixtures/real_openfoam/duct/converged
#   duct_early         tests/fixtures/real_openfoam/duct/early
# For each, `postProcess -latestTime` runs flowRatePatch on every patch of the mesh's boundary
# file that is not `empty`, and its stdout is written to /case/<name>.log, committed as written
# beside this script. The logs stay out of the case directories, where a log.* would change which
# log `courant verify` reads; postProcessing/ is not committed.
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
for name in wedge_pipe_flow heated_cavity laminar_cylinder duct_converged duct_early; do
    cd /case/$name
    # Each patch named in constant/polyMesh/boundary, with its type, the empty ones left out.
    patches=$(awk '
        /^[ \t]*[A-Za-z_][A-Za-z0-9_.:-]*[ \t]*$/ { name = $1; next }
        $1 == "type" { t = $2; sub(/;$/, "", t); if (name != "" && t != "empty") print name; name = "" }
    ' constant/polyMesh/boundary)
    funcs="("
    for p in $patches; do funcs="$funcs flowRatePatch(name=$p)"; done
    funcs="$funcs )"
    postProcess -latestTime -funcs "$funcs" > /case/$name.log 2>&1
done
echo done
