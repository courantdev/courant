#!/bin/bash
# Run inside opencfd/openfoam-default@sha256:dd5aa206...a35a (ESI v2406): writes each mesh with
# blockMesh, once with writeFormat binary and once with writeFormat ascii.
set -e
source /usr/lib/openfoam/openfoam2406/etc/bashrc || true
T=$FOAM_TUTORIALS/compressible/rhoPimpleFoam/RAS/cavity
make() {  # name nx ny nz
  for fmt in binary ascii; do
    d=/out/$1/$fmt; rm -rf $d; mkdir -p $d; cp -r $T/system $T/constant $d/
    sed -i "s/^writeFormat .*/writeFormat     $fmt;/" $d/system/controlDict
    sed -i "s/hex (0 1 2 3 4 5 6 7) ([0-9]* [0-9]* [0-9]*)/hex (0 1 2 3 4 5 6 7) ($2 $3 $4)/" $d/system/blockMeshDict
    (cd $d && blockMesh > log.blockMesh 2>&1)
    find $d/system -type f ! -name blockMeshDict ! -name controlDict -delete; rm -rf $d/constant/thermophysicalProperties $d/constant/momentumTransport $d/constant/turbulenceProperties $d/constant/g 2>/dev/null || true
  done
}
[ -n "$ONLY_SCALED" ] || make cavity_2x3 2 3 1
[ -n "$ONLY_SCALED" ] || make one_cell 1 1 1
grep -h "hex\|convertToMeters\|scale" $T/system/blockMeshDict
echo "$WM_PROJECT_VERSION $WM_LABEL_SIZE $WM_PRECISION_OPTION $(uname -m)"
# The 2 x 3 cavity's binary mesh scaled by 0.9 after blockMesh, as transformPoints writes it.
d=/out/cavity_2x3_scaled/binary; rm -rf $d; mkdir -p $d; cp -r /out/cavity_2x3/binary/. $d/
cp $T/system/controlDict $d/system/controlDict
(cd $d && transformPoints -scale 0.9 > log.transformPoints 2>&1)
find $d/system -type f ! -name blockMeshDict ! -name controlDict -delete
