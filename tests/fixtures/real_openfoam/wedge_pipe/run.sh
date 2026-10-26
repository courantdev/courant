#!/bin/bash
# Runs inside opencfd/openfoam-default:2406 with the scratch wedge directory mounted at /case.
# flow: blockMesh, checkMesh -allGeometry -allTopology, simpleFoam.
# energy: the flow's mesh and final velocity and flux, then scalarTransportFoam (steadyState).
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
cd /case/flow
blockMesh > log.blockMesh 2>&1
checkMesh -allGeometry -allTopology > log.checkMesh 2>&1 || true
simpleFoam > log.simpleFoam 2>&1
cd /case/energy
rm -rf constant/polyMesh 1000
cp -r /case/flow/constant/polyMesh constant/polyMesh
rm -rf constant/polyMesh/sets
mkdir -p 1000
cp /case/flow/1000/U /case/flow/1000/phi /case/flow/1000/p 1000/
cp 0/T 1000/T
cp /case/flow/log.checkMesh log.checkMesh
scalarTransportFoam > log.scalarTransportFoam 2>&1
echo done
