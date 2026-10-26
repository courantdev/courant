"""Whether a case's mesh moves, and whether it is joined by a non-conformal interface.

One definition, for the case reader (`CaseConfig.mesh_motion`), the field reader's flux balance
(`courant.verification._fluid_reader`, which does not judge a moving mesh's) and the line a verdict
gives a run on a moving or sliding mesh (`courant.verify.sentences.moving_mesh`, TRD VER-5a).

**The mesh moves** when `constant/dynamicMeshDict` says so: ESI names the mesh's type in
`dynamicFvMesh` (`dynamicFvMesh::New`, dynamicFvMeshNew.C, reads that word, and `staticFvMesh` is
the type that never changes, staticFvMesh.C); OpenFOAM.org writes `mover`, `topoChanger` or
`refiner` sub-dictionaries. Such a mesh moves or changes during the run (`mesh.changing()`), which
is when pimpleFoam corrects the flux and solves `pcorr` (pimpleFoam.C, `correctPhi` defaulting to
`mesh.dynamic()` in createDyMControls.H).

**A non-conformal interface** is a patch held by an arbitrary mesh interface (ESI's: v2406's
patch-field types `cyclicAMI`, `cyclicACMI`, `cyclicPeriodicAMI`, `jumpCyclicAMI`, `fixedJumpAMI`
and `uniformJumpAMI`, read from its sources' `TypeName`s; every type naming `AMI` or `ACMI`;
OpenFOAM.org's `nonConformal...` types): the faces of its two sides do not match, and a field
crosses it by interpolation weights. On a moving mesh it slides (a rotor turning inside a stator).

Neither is judged: the flux and torque that cross a sliding interface are not summed, so whether
the interpolation conserved them is not assessed, and the verdict says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: ESI's type of a mesh that never changes (staticFvMesh.C, `TypeName("staticFvMesh")`).
STATIC = "staticFvMesh"
#: OpenFOAM.org's sub-dictionaries of a mesh that moves, changes topology or refines.
ORG_KEYS = ("mover", "topoChanger", "refiner")


@dataclass(frozen=True)
class MeshMotion:
    #: The keyword that says the mesh moves: "dynamicFvMesh" (ESI), "mover", "topoChanger" or
    #: "refiner" (OpenFOAM.org), or "unreadable" when the file exists and could not be read.
    by: str
    #: ESI's `dynamicFvMesh` word, as the case writes it (the field reader's local note quotes
    #: it); None otherwise. Never in a finding.
    kind: str | None = None


def read_mesh_motion(case_dir: str | os.PathLike[str] | None) -> MeshMotion | None:
    """Whether `constant/dynamicMeshDict` moves or changes the mesh: a `MeshMotion`, or None for a
    fixed mesh (no such file, or one that names `staticFvMesh` and none of OpenFOAM.org's keys).
    The file is noted for the manifest once read (VER-14). One that resolves outside a
    confinement (a link out of the case) is not read (MCP-3) and is "unreadable": the solver
    reads it, so whether the mesh moves is not known. Its target is never looked at."""
    from courant.case.reads import may_read, reading

    if not case_dir:
        return None
    path = os.path.join(os.fspath(case_dir), "constant", "dynamicMeshDict")
    if not os.path.lexists(path):
        return None
    if not may_read(path):
        return MeshMotion("unreadable")
    if not os.path.isfile(path):                        # a directory, a FIFO: not a dictionary
        return None
    from foamlib import FoamFile

    try:
        with reading(path):
            d = FoamFile(path)
            kind = d.get("dynamicFvMesh")
            org = [k for k in ORG_KEYS if k in d]
    except Exception:                                   # noqa: BLE001
        return MeshMotion("unreadable")
    if kind is not None and str(kind) != STATIC:
        return MeshMotion("dynamicFvMesh", str(kind))
    if org:
        return MeshMotion(org[0])
    return None


def is_interface_type(bc_type: str | None) -> bool:
    """Whether a boundary condition's type is a non-conformal interface's (module docstring)."""
    t = str(bc_type or "")
    return "AMI" in t or "ACMI" in t or t.startswith("nonConformal")
