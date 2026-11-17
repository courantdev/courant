# The demo pitzDaily mesh below the floor (T-ENV-1 S1, TRD TST-13, SB-22)

Written on 8 Nov 2026 by `make.sh` in
`opencfd/openfoam-default@sha256:dd5aa20630a55722663bf83ba0cb74870cba130081303e32e3865007fa2aa35a`
(`packaging/images.lock`; ESI v2406, `WM_LABEL_SIZE=32`, `WM_PRECISION_OPTION=DP`, run as the
image's linux/arm64 variant on the Mac, under `scripts/heavy`):

    scripts/heavy docker run --rm --cpus 1 --network none --pull never \
      -v <demo case>:/case:ro -v $PWD:/out --entrypoint bash <image> /out/make.sh

`<demo case>` is the bundled demo run extracted by `courant.data.extract_demo_case`.

- `system/blockMeshDict`: the demo's own `blockMeshDict` with every hex block's cell counts
  halved (integer division, at least 1), written by the exploit corpus's own mutation
  (`coarsened`); the corpus entry asserts that its mutation of the demo case gives exactly this
  file. Hex counts (9 15 1) (90 13 1) (90 15 1)
  (12 13 1) (12 15 1): 2,991 cells by `courant.spec.check.bounded_quantity`, against the demo's
  12,225.
- `constant/polyMesh/*.gz`: the mesh blockMesh wrote from it with the demo's own
  `system/controlDict` (`writeFormat ascii`, `writePrecision 6`, `writeCompression off`),
  gzipped with `gzip -n -9` (no name, no time stamp). Its owner note reads `nCells: 2991`.
  Nothing was edited after blockMesh wrote it. The test decompresses each file into the
  re-solve's copy, where an uncompressed mesh is what blockMesh leaves.
- `log.blockMesh`: blockMesh's own log of that run.

SHA-256 on 8 Nov 2026:

    0ea38f1fbe4367d7a9b008623c28e0e68585cba87016282e96add2579995ad0d  system/blockMeshDict
    2468a329fef546ab32de5fb6fd83a4ce6a06e33b75794b03d7880ad5e5b213dc  constant/polyMesh/boundary.gz
    4031a0f9acf5fbc1cae71ae4da4bbbca52afc949c94b1b5fd7e9897ef0192393  constant/polyMesh/faces.gz
    20b15bb368645b79215e47250878c41cec51b1d315d81c8eef32ebce13d2049b  constant/polyMesh/neighbour.gz
    aed6c87c480afd766c8340b151bab4244b45357ef14b52bceb8342bd1d396bf9  constant/polyMesh/owner.gz
    7eaac828622bea0315ef7b4bc7f33eec8f232e1e4e842f57f3e04961df140e6c  constant/polyMesh/points.gz
