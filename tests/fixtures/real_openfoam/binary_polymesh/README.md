# Binary and ASCII polyMesh written by the pinned image's blockMesh (T-BINARY-POLYMESH, TRD FLD-7)

Written on 4 Nov 2026 by `make.sh` in `opencfd/openfoam-default@sha256:dd5aa20630a55722663bf83ba0cb74870cba130081303e32e3865007fa2aa35a`
(`packaging/images.lock`, linux/amd64; ESI v2406, `WM_LABEL_SIZE=32`, `WM_PRECISION_OPTION=DP`):

    docker run --rm --cpus 2 -v $PWD:/gen -v $PWD:/out --entrypoint bash <image> /gen/make.sh

Each mesh is the rhoPimpleFoam RAS cavity tutorial's `blockMeshDict` (`scale 0.1`) with its cell
counts changed, meshed twice: once with the tutorial's own `writeFormat binary`, once with
`writeFormat ascii`. Nothing was edited after blockMesh wrote it.

- `cavity_2x3`: 2 x 3 x 1 cells (6 cells, 12 points per layer x 2).
- `cavity_2x3_scaled`: `cavity_2x3`'s binary mesh after `transformPoints -scale 0.9` (run with
  `ONLY_SCALED=1` after the others): its points are 0.9 times blockMeshDict's, written binary.
- `one_cell`: 1 x 1 x 1 cell, so no internal face: the binary `neighbour` is a count of 0 and no
  brackets, as OpenFOAM writes an empty binary list.

The binary files carry `arch "LSB;label=32;scalar=64"`. Other label and scalar sizes and the MSB
byte order are not written by this image; the tests make them from these files' own arrays.
