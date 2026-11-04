#!/bin/bash
# Runs inside opencfd/openfoam-default:2406 with the scratch case directory mounted at /case:
# blockMesh, checkMesh, decomposePar over two ranks, pimpleFoam on both, and reconstructPar of the
# last time. The logs keep OpenFOAM's names (log.<application>), as Courant reads them.
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
cd /case
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh.out 2>&1 || true
decomposePar > log.decomposePar 2>&1
mpirun --allow-run-as-root --oversubscribe -np 2 pimpleFoam -parallel > log.pimpleFoam 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1
echo done
