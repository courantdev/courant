"""Held-in and held-out splits (TRD CT-4, CT-11). Hash-locked (DEV-9).

Families are held out by tutorial case, never by fault. Anything tuned on the held-in families
is scored on case families it has never seen.

- **v0** (28 Sep 2026) held out the two solver families with the most distinctive logs: a
  free-surface VOF run and a buoyant compressible run.
- **v1** (30 Sep 2026, decided by S) holds out `buoyantPimpleFoam_hotRoom` only.
  `interFoam_damBreak` moved to held-in when its false alarm was fixed (TRD VER-13), because
  that fix was informed by its own held-out result. A family whose result shaped the code is
  not unseen, and reporting it as held out would overstate the evidence.
"""

HELD_OUT_FAMILIES_V0 = ("buoyantPimpleFoam_hotRoom", "interFoam_damBreak")
HELD_OUT_FAMILIES_V1 = ("buoyantPimpleFoam_hotRoom",)
HELD_OUT_FAMILIES = HELD_OUT_FAMILIES_V1


def split_of(family: str) -> str:
    return "held_out" if family in HELD_OUT_FAMILIES else "held_in"
