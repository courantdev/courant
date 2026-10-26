"""FieldData — a structured view of a run's solution fields (Tech Arch §7, §2.2).

The field-based verification checks (conservation, y+, regime, plausibility)
operate on this. It is populated live from a solved case (foamlib); offline
it is ``None`` and the field-based checks skip. Kept deliberately small — only the
quantities the verification suite needs. Nothing in the product fills `yplus` or `length_scale`:
only tests supply them, so the y+ and Reynolds checks say "not implemented" on every real case
(TRD VER-7c, hard-proofing audit, 21 Oct 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FluxSource:
    """Where `FieldData.patch_fluxes` came from: one `phi` file as the solver wrote it (TRD
    VER-18). Each sum is over the faces that bound the domain (no `empty` or coupled patch)."""

    file: str        # case-relative, as read: "2111/phi" (or "2111/phi.gz")
    kind: str        # "mass" (kg/s: a compressible solver's phi) or "volume" (m^3/s)
    inflow: float    # sum of -phi over every boundary face where phi < 0 (phi points out)
    outflow: float   # sum of phi over every boundary face where phi > 0
    largest: float   # the largest |phi| on any face in the file, internal faces included


@dataclass
class FieldData:
    time: float
    internal: dict[str, np.ndarray] = field(default_factory=dict)   # field -> values
    patch_fluxes: dict[str, float] = field(default_factory=dict)    # patch -> signed phi
    yplus: np.ndarray | None = None                                 # per wall-face y+
    length_scale: float | None = None                              # characteristic L
    nu: float | None = None                                        # kinematic viscosity
    inlet_speed: float | None = None                              # reference |U|
    #: The file `patch_fluxes` was summed from, when the reader filled it (VER-18).
    flux_source: FluxSource | None = None
    #: Why `patch_fluxes` is empty, when the reader was asked for it and could not fill it: a
    #: sentence for "not checked" (no `phi`, a binary one, a decomposed run, a moving mesh, ...).
    flux_note: str | None = None
    #: Which time was read, and what the reader saw beside it (TRD VER-7b): the time directory
    #: as named on disk; the fields whose file there exists and could not be read (on no rank);
    #: (name, ranks read, ranks) for a field of a decomposed case some ranks read and others did
    #: not, which `internal` holds for the ranks read only; a later time held by `processor*/`
    #: than the case root's it read; the time directories later than the log's last step it was
    #: told to stop at (`until`) and so did not read.
    source_time: str | None = None
    unread: tuple[str, ...] = ()
    partial: tuple[tuple[str, int, int], ...] = ()
    later_ranks: str | None = None
    after_log: tuple[str, ...] = ()

    def velocity_magnitude(self) -> np.ndarray | None:
        u = self.internal.get("U")
        if u is None:
            return None
        u = np.asarray(u)
        return np.linalg.norm(u, axis=1) if u.ndim == 2 else np.abs(u)

    @classmethod
    def from_case(cls, case_dir: str, time: str | None = None) -> "FieldData":  # pragma: no cover
        """Read solution fields from a solved case (live path). The boundary flux is not read
        here: only `courant verify --checks next` reads it (VER-18, GATE-13)."""
        from courant.verification._fluid_reader import read_fields

        return read_fields(case_dir, time)
