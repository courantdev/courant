#!/bin/bash
# Runs inside opencfd/openfoam-default:2406 with the scratch cht_plate directory (make_case.py's
# output) mounted at /case. converged and early: blockMesh (two cell zones), then
# splitMeshRegions -cellZones -overwrite (with 0/ set aside, so it maps no field and the
# region fields make_case.py wrote stay as written), then chtMultiRegionSimpleFoam. The logs
# keep OpenFOAM's names (log.<application>).
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
for run in converged early; do
    cd /case/$run
    mv 0 0.fields
    blockMesh > log.blockMesh 2>&1
    splitMeshRegions -cellZones -overwrite > log.splitMeshRegions 2>&1
    rm -rf 0
    mv 0.fields 0
    chtMultiRegionSimpleFoam > log.chtMultiRegionSimpleFoam 2>&1
done
echo done
