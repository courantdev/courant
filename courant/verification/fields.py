"""FieldData — a structured view of a run's solution fields (Tech Arch §7, §2.2).

The field-based verification checks (conservation, y+, regime, plausibility)
operate on this. It is populated live from a solved case (foamlib); offline
it is ``None`` and the field-based checks skip. Kept deliberately small — only the
quantities the verification suite needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FieldData:
    time: float
    internal: dict[str, np.ndarray] = field(default_factory=dict)   # field -> values
    patch_fluxes: dict[str, float] = field(default_factory=dict)    # patch -> signed phi
    yplus: np.ndarray | None = None                                 # per wall-face y+
    length_scale: float | None = None                              # characteristic L
    nu: float | None = None                                        # kinematic viscosity
    inlet_speed: float | None = None                              # reference |U|

    def velocity_magnitude(self) -> np.ndarray | None:
        u = self.internal.get("U")
        if u is None:
            return None
        u = np.asarray(u)
        return np.linalg.norm(u, axis=1) if u.ndim == 2 else np.abs(u)

    @classmethod
    def from_case(cls, case_dir: str, time: str | None = None) -> "FieldData":  # pragma: no cover
        """Read solution fields from a solved case (live path)."""
        from courant.verification._fluid_reader import read_fields

        return read_fields(case_dir, time)
