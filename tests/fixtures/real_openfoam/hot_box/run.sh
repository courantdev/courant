#!/bin/bash
# Runs inside opencfd/openfoam-default:2406 with the scratch hot_box directory (make_case.py's
# output) mounted at /case. converged and early: blockMesh, then buoyantSimpleFoam. The logs
# keep OpenFOAM's names (log.<application>).
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
for run in converged early; do
    cd /case/$run
    blockMesh > log.blockMesh 2>&1
    buoyantSimpleFoam > log.buoyantSimpleFoam 2>&1
done
echo done
