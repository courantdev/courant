#!/bin/bash
# Run inside opencfd/openfoam-default@sha256:dd5aa206...a35a (ESI v2406), with /case a copy of the
# bundled demo pitzDaily case (`courant.data.extract_demo_case`) and /out this directory: meshes
# the demo with this directory's coarsened blockMeshDict and the demo's own controlDict, as the
# re-solve's blockMesh does, and keeps the polyMesh gzipped (gzip -n, no name or time stamp).
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc || true
d=/tmp/case; rm -rf $d; cp -r /case $d
rm -rf $d/constant/polyMesh
cp /out/system/blockMeshDict $d/system/blockMeshDict
(cd $d && blockMesh > log.blockMesh 2>&1) || { cat $d/log.blockMesh; exit 1; }
rm -rf /out/constant; mkdir -p /out/constant/polyMesh
for f in $d/constant/polyMesh/*; do gzip -n -9 -c "$f" > /out/constant/polyMesh/$(basename "$f").gz; done
cp $d/log.blockMesh /out/log.blockMesh
echo "$WM_PROJECT_VERSION $WM_LABEL_SIZE $WM_PRECISION_OPTION $(uname -m)"
