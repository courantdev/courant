#!/bin/bash
# Runs inside opencfd/openfoam-default:2406 with the scratch duct directory (make_case.py's
# output) mounted at /case. converged and early: blockMesh, then rhoSimpleFoam. binary: a copy of
# the converged run with writeFormat binary, foamFormatConvert of its last time; only that
# time's phi is committed. The logs keep OpenFOAM's names (log.<application>).
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
for run in converged early; do
    cd /case/$run
    blockMesh > log.blockMesh 2>&1
    rhoSimpleFoam > log.rhoSimpleFoam 2>&1
done
rm -rf /case/binary-scratch
cp -r /case/converged /case/binary-scratch
cd /case/binary-scratch
foamDictionary system/controlDict -entry writeFormat -set binary > /dev/null
foamFormatConvert -latestTime > log.foamFormatConvert 2>&1
t=$(foamListTimes -latestTime)
mkdir -p /case/binary/$t
cp $t/phi /case/binary/$t/phi
echo done
