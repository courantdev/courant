"""MeshQuality — structured checkMesh metrics for diagnosis (Tech Arch §5.1).

The mesh checker "wraps checkMesh; extracts skewness, non-orthogonality, aspect
ratio, negative volumes into structured Findings." This model is that structured
extraction. Constructors: ``from_dict`` (replay case-facts) and ``from_checkmesh``
(parse real checkMesh stdout, live path).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict


class MeshQuality(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_cells: int | None = None
    max_non_orthogonality: float | None = None       # degrees
    average_non_orthogonality: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None
    n_negative_volume_cells: int = 0
    mesh_ok: bool | None = None                      # checkMesh "Mesh OK." verdict

    @classmethod
    def from_dict(cls, data: dict) -> "MeshQuality":
        return cls.model_validate(data)

    @classmethod
    def from_checkmesh(cls, text: str) -> "MeshQuality":
        """Parse the metrics out of checkMesh stdout."""

        # The first group that actually captured, not group(1).
        #
        # With an alternation, whichever branch matches at the *earliest*
        # position wins — and for `aspect ratio…|Max aspect ratio…` against real
        # output, that is the second branch, because "Max" starts before
        # "aspect". So group(1) was None and `float(None)` raised TypeError on
        # every real checkMesh log. `read_mesh_quality` catches and returns None,
        # so the symptom was silent: mesh metrics simply never arrived, on the
        # agent path as well as the live one, and all five MESH.* codes were
        # undiagnosable even when a checkMesh log was sitting right there.
        def _first(match: "re.Match | None") -> str | None:
            if match is None:
                return None
            return next((g for g in match.groups() if g is not None), None)

        def _f(pattern: str) -> float | None:
            captured = _first(re.search(pattern, text))
            return float(captured) if captured is not None else None

        def _i(pattern: str) -> int | None:
            captured = _first(re.search(pattern, text))
            return int(captured) if captured is not None else None

        non_ortho_max = _f(r"non-orthogonality Max:\s*([\d.eE+-]+)")
        non_ortho_avg = _f(r"non-orthogonality Max:\s*[\d.eE+-]+\s*average:\s*([\d.eE+-]+)")
        # `[=:]` — and this is the second time this line has been wrong.
        #
        # checkMesh prints the metric one way when the mesh is fine and another
        # way when it is not:
        #
        #     Max aspect ratio = 1 OK.
        #     ***High aspect ratio cells found, Max aspect ratio: 2000, ...
        #
        # The failing form uses a **colon**. With `=?` the value was readable
        # only on a healthy mesh and unreadable on exactly the meshes worth
        # reporting — so `MESH.ASPECT_RATIO` could never be diagnosed, and the
        # scenario for it came back with an empty diagnosis while `checkMesh`
        # was printing the number in as many words.
        #
        # A perfect inversion, and the same shape as the `CONV.MASS_IMBALANCE`
        # log pattern that fired on healthy runs and missed broken ones. The
        # lesson both times: the failing output is a *different string* from the
        # passing one, and a pattern derived from a healthy run has never seen
        # the case it exists for.
        skew = _f(r"[Mm]ax skewness\s*[=:]?\s*([\d.eE+-]+)")
        aspect = _f(r"Max aspect ratio\s*[=:]?\s*([\d.eE+-]+)|"
                    r"aspect ratio\s*[=:]?\s*([\d.eE+-]+)")
        n_cells = _i(r"cells:\s*(\d+)")
        neg = _i(r"(\d+)\s+cells with negative volume") or (
            0 if "negative volume" not in text.lower() else 1
        )
        mesh_ok = "Mesh OK." in text or "Mesh OK" in text

        return cls(
            n_cells=n_cells,
            max_non_orthogonality=non_ortho_max,
            average_non_orthogonality=non_ortho_avg,
            max_skewness=skew,
            max_aspect_ratio=aspect,
            n_negative_volume_cells=neg or 0,
            mesh_ok=mesh_ok,
        )


#: Where a checkMesh log conventionally lands. `log.checkMesh` is what
#: OpenFOAM's own `runApplication` wrapper writes and what `DockerRunner` writes
#: for a pre-command; `checkMesh.log` is the other common hand-rolled spelling.
CHECKMESH_LOG_NAMES = ("log.checkMesh", "checkMesh.log")


def read_mesh_quality(case_dir: str) -> "MeshQuality | None":
    """Mesh metrics from a checkMesh log already sitting in the case, or None.

    Shared by the two paths that need it so there is one answer to "where does
    the checkMesh log live". They differ in who *runs* checkMesh, not in where
    they look for it:

      * `courant watch` never runs it. That is the customer's cluster, their
        queue and possibly their 50M-cell mesh, and a watcher that silently
        starts a heavy job is a watcher nobody installs twice.
      * `run_live_case` does, because the sandbox is ours and the run is already
        happening inside it.

    Never raises. A partial or truncated checkMesh log is common, and losing
    mesh metrics must not cost the rest of the diagnosis.
    """
    from pathlib import Path

    from courant.case.reads import note

    for name in CHECKMESH_LOG_NAMES:
        candidate = Path(case_dir) / name
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(errors="replace")
            note(candidate)             # VER-14: read, so listed, whether or not it parses
            return MeshQuality.from_checkmesh(text)
        except Exception:  # noqa: BLE001 — a partial log is not a failure
            return None
    return None
