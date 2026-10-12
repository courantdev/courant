"""The task specification, `courant.spec/1` (PRD 2.0 P2-SPEC-1).

A specification names the values a task asked for and where each lives in the case:

    {
      "schema": "courant.spec/1",
      "task": "pitzDaily, inlet 10 m/s, k-epsilon",
      "items": [
        {"name": "inlet velocity", "file": "0/U", "patch": "inlet", "value": [10, 0, 0], "units": "m/s"},
        {"name": "nu", "file": "constant/transportProperties", "key": "nu", "value": 1e-05},
        {"name": "turbulence model", "file": "constant/turbulenceProperties",
         "path": ["RAS"], "key": "RASModel", "value": "kEpsilon", "modelling_choice": true},
        {"name": "geometry", "file": "constant/triSurface/body.stl", "sha256": "…"}
      ]
    }

Each item has exactly one locator: `patch` (a boundary value; `key` defaults to `value`),
`key` (an entry, inside the blocks named by `path`), `blocks: "cells"` (the cell counts of
every `hex` block in a blockMeshDict) or `sha256` (the whole file's hash). Values are
compared at a relative tolerance (default 1%; directions to 1°); cell counts and hashes
exactly. An item marked `modelling_choice` is binding like any other: if the specification
names it, the case must match it.

A specification is locked by its file's SHA-256 before the run (`courant spec hash`); a
verdict records the hash, and a specification that no longer matches the locked hash makes
the verdict `NOT_ASSESSABLE` (P2-SPEC-3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SPEC_SCHEMA = "courant.spec/1"
DEFAULT_REL_TOL = 0.01
DEFAULT_ANGLE_TOL_DEG = 1.0


class SpecError(ValueError):
    """The specification could not be read or is not valid `courant.spec/1`."""


class SpecItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    file: str = Field(min_length=1)
    patch: str | None = None
    key: str | None = None
    path: tuple[str, ...] = ()
    blocks: Literal["cells"] | None = None
    sha256: str | None = None
    value: float | tuple[float, ...] | str | None = None
    units: str | None = None
    #: OpenFOAM dimension set, 5 or 7 exponents; checked when the case writes one.
    dimensions: tuple[float, ...] | None = None
    rel_tol: float | None = Field(default=None, ge=0)
    abs_tol: float | None = Field(default=None, ge=0)
    angle_tol_deg: float | None = Field(default=None, ge=0)
    #: SPEC-12: a bound instead of a value, in the item's units (for `blocks: "cells"`, the
    #: total cell count). A value outside it is a nonconformance (VER-5.7). A bound that cannot
    #: be met (`min` above `max`) is not a schema error: it is an impossible specification
    #: (SPEC-9), judged `insufficient`.
    min: float | None = None
    max: float | None = None
    modelling_choice: bool = False

    @model_validator(mode="after")
    def _one_locator(self) -> SpecItem:
        kinds = [k for k, on in (("patch", self.patch is not None),
                                 ("key", self.key is not None and self.patch is None),
                                 ("blocks", self.blocks is not None),
                                 ("sha256", self.sha256 is not None)) if on]
        if len(kinds) != 1:
            raise ValueError(f"item '{self.name}': give exactly one of patch, key, blocks or "
                             f"sha256 (got {', '.join(kinds) or 'none'})")
        if self.path and self.patch is not None:
            raise ValueError(f"item '{self.name}': 'path' applies to a key, not a patch")
        if self.bounded:
            if self.value is not None:
                raise ValueError(f"item '{self.name}': give a value or a bound (min, max), "
                                 "not both")
            if self.sha256 is not None:
                raise ValueError(f"item '{self.name}': a sha256 item takes no bound")
        elif self.sha256 is None and self.value is None:
            raise ValueError(f"item '{self.name}': a value is required (or a bound: min, max)")
        if self.sha256 is not None and self.value is not None:
            raise ValueError(f"item '{self.name}': a sha256 item takes no value")
        if Path(self.file).is_absolute() or ".." in Path(self.file).parts:
            raise ValueError(f"item '{self.name}': file must be relative to the case and "
                             "inside it")
        return self

    @property
    def bounded(self) -> bool:
        """SPEC-12: the item states a bound instead of a value."""
        return self.min is not None or self.max is not None

    @property
    def exact(self) -> bool:
        return self.blocks is not None or self.sha256 is not None

    @property
    def location(self) -> str:
        if self.patch is not None:
            return f"boundaryField/{self.patch}/{self.key or 'value'}"
        if self.blocks is not None:
            return "blocks: total hex cells" if self.bounded else "blocks: hex cell counts"
        if self.sha256 is not None:
            return "file SHA-256"
        return "/".join([*self.path, self.key or ""])


class Provenance(BaseModel):
    """SPEC-7: where a specification came from. It lives inside the hashed content, so nobody
    can change it after the lock without changing the hash.

    A specification whose provenance names `extracted_by` without `confirmed_by` is never
    binding, whatever the lock says: it is judged on basis `validity+extracted-spec`, where a
    mismatch can only lower a verdict to `insufficient` (VER-5.12).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The tool that drafted it from a task in words, e.g. "courant spec extract 0.9.0".
    extracted_by: str | None = None
    #: The model id the door answered with, and the UTC date of the extraction.
    model: str | None = None
    date: str | None = None
    #: The person who reviewed it (`courant spec confirm --by NAME`), and when.
    confirmed_by: str | None = None
    confirmed_date: str | None = None

    @property
    def extracted_unconfirmed(self) -> bool:
        return bool(self.extracted_by) and not self.confirmed_by


class Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["courant.spec/1"] = Field(alias="schema")
    task: str | None = None
    #: SPEC-7. Absent on a specification written by hand or drafted from a reference case.
    provenance: Provenance | None = None
    rel_tol: float = Field(default=DEFAULT_REL_TOL, ge=0)
    angle_tol_deg: float = Field(default=DEFAULT_ANGLE_TOL_DEG, ge=0)
    items: tuple[SpecItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> Spec:
        names = [i.name for i in self.items]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"item names must be unique: {', '.join(dupes)}")
        return self

    def to_json(self) -> str:
        data = self.model_dump(mode="json", by_alias=True, exclude_none=True,
                               exclude_defaults=False)
        for item in data["items"]:
            for k in ("path",):
                if not item.get(k):
                    item.pop(k, None)
            if not item.get("modelling_choice"):
                item.pop("modelling_choice", None)
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def spec_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: str | Path) -> tuple[Spec, str]:
    """(spec, sha256 of the file). Raises SpecError with a sentence a user can act on."""
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"no such specification: {path}")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        spec = Spec.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise SpecError(f"{path.name} is not valid {SPEC_SCHEMA}: {where}: {first.get('msg')}") from exc
    return spec, hashlib.sha256(raw).hexdigest()
