#!/bin/bash
# Runs inside opencfd/openfoam-default:2406 with the scratch cavity directory mounted at /case:
# blockMesh, then buoyantBoussinesqSimpleFoam to its endTime.
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true
cd /case/run
blockMesh > log.blockMesh 2>&1
buoyantBoussinesqSimpleFoam > log.buoyantBoussinesqSimpleFoam 2>&1
echo done
