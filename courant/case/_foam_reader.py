"""Read a real OpenFOAM case directory into a CaseConfig (live path).

Uses foamlib to parse the dictionaries that matter for diagnosis. Best-effort and
defensive: a missing file or key is skipped, never fatal — real cases vary. This is
what makes the live DockerRunner path work end to end (config_view over real files).
"""

from __future__ import annotations

import os

from courant.case.config import CaseConfig, LinearSolverSpec
from courant.case.reads import reading


def _scalar(v):
    """Extract a float from foamlib's varied return types (number, dimensioned, tuple)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "value"):
        try:
            return float(v.value)  # foamlib Dimensioned
        except Exception:  # noqa: BLE001
            return None
    if isinstance(v, (tuple, list)) and v:
        last = v[-1]
        return float(last) if isinstance(last, (int, float)) else None
    return None


def _tokens(v) -> str | None:
    """Render a scheme value (tuple of tokens or string) as a string."""
    if isinstance(v, (tuple, list)):
        return " ".join(str(t) for t in v)
    if v is None:
        return None
    return str(v)


def _get(f, *key):
    try:
        return f[key if len(key) > 1 else key[0]]
    except Exception:  # noqa: BLE001 — missing key, or an unreadable file
        return None


def _readable(f, path: str, rel: str, unreadable: dict[str, str]) -> bool:
    """Whether this file parses at all, recording the reason if not.

    Every `_get` below swallows its exception, which is right for a missing key
    and wrong for a file that cannot be parsed: the two are indistinguishable
    afterwards, so an unreadable `controlDict` produced a `CaseConfig` with no
    solver, no timestep and no write control — *silently*, exactly as though the
    case had chosen to omit them.

    That is reachable on real cases. `foamDictionary -set` rewrites a dictionary
    with its `#include` directives expanded, and OpenFOAM's own
    `pisoFoam/RAS/cavity` expands into `fields ( U { mean on; ... } );` — a list
    containing an inline dictionary, which foamlib cannot parse. Any case an
    engineer has edited with OpenFOAM's own tool can arrive in this state, and
    every controlDict-derived checker and KB predicate then reads nothing and
    says nothing about why.
    """
    try:
        _parse(f, path)
    except Exception as exc:            # noqa: BLE001
        unreadable[rel] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        return False
    return True


def _parse(f, path: str) -> None:
    """Parse a `FoamFile` once, noting `path` (the path the reader used, not foamlib's
    resolution of it) when the bytes were read (VER-14).

    foamlib reads lazily, and every `_get` swallows its exception, so this is the one place a
    read can be seen to succeed. A file that cannot be opened (`OSError`) is not noted; a file
    that was read and does not parse is noted, then the error propagates.
    """
    with reading(path):
        try:
            list(f.items())
        except KeyError:
            pass                        # parsed; simply has no entries we asked for


def _subdict_items(f, key):
    """Yield (name, value) for a subdict; empty if absent."""
    d = _get(f, key)
    if d is None:
        return []
    try:
        return list(d.items())
    except Exception:  # noqa: BLE001
        return []


def read_case_config(case_dir: str) -> CaseConfig:
    from foamlib import FoamFile

    data: dict = {}
    unreadable: dict[str, str] = {}

    def open_file(rel):
        p = os.path.join(case_dir, rel)
        if not os.path.exists(p):
            return None
        f = FoamFile(p)                 # VER-14: `_readable` notes it once it is read
        return f if _readable(f, p, rel, unreadable) else None

    # --- controlDict ---
    cd = open_file("system/controlDict")
    if cd is not None:
        data["solver"] = _get(cd, "application")
        data["delta_t"] = _scalar(_get(cd, "deltaT"))
        data["end_time"] = _scalar(_get(cd, "endTime"))
        adj = _get(cd, "adjustTimeStep")
        data["adjust_timestep"] = bool(adj) if adj is not None else None
        data["max_co"] = _scalar(_get(cd, "maxCo"))
        wc = _get(cd, "writeControl")
        data["write_control"] = str(wc) if wc is not None else None
        data["write_interval"] = _scalar(_get(cd, "writeInterval"))

    # --- fvSchemes ---
    fs = open_file("system/fvSchemes")
    if fs is not None:
        ddt = _get(fs, "ddtSchemes", "default")
        data["ddt_scheme"] = _tokens(ddt)
        div = {k: _tokens(v) for k, v in _subdict_items(fs, "divSchemes")}
        if div:
            data["div_schemes"] = {k: v for k, v in div.items() if v}

    # --- fvSolution ---
    fv = open_file("system/fvSolution")
    if fv is not None:
        relax: dict[str, float] = {}
        scopes: dict[str, str] = {}
        rf = _get(fv, "relaxationFactors")
        if rf is not None:
            # flatten nested fields/equations, or a flat mapping
            try:
                items = list(rf.items())
            except Exception:  # noqa: BLE001
                items = []
            for k, v in items:
                if k in ("fields", "equations"):
                    try:
                        for kk, vv in v.items():
                            s = _scalar(vv)
                            if s is not None:
                                relax[kk] = s
                                scopes[kk] = k
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    s = _scalar(v)
                    if s is not None:
                        relax[k] = s
                        scopes[k] = ""
        if relax:
            data["relaxation_factors"] = relax
            data["relaxation_scopes"] = scopes
        for loop in ("SIMPLE", "PIMPLE"):
            nno = _scalar(_get(fv, loop, "nNonOrthogonalCorrectors"))
            if nno is not None:
                data["n_non_orthogonal_correctors"] = int(nno)
            ncorr = _scalar(_get(fv, loop, "nCorrectors"))
            if ncorr is not None:
                data["n_correctors"] = int(ncorr)
        nouter = _scalar(_get(fv, "PIMPLE", "nOuterCorrectors"))
        if nouter is not None:
            data["n_outer_correctors"] = int(nouter)
        solvers: dict[str, LinearSolverSpec] = {}
        for field, spec in _subdict_items(fv, "solvers"):
            try:
                solvers[field] = LinearSolverSpec(
                    field=field, solver=_tokens(spec.get("solver")),
                    tolerance=_scalar(spec.get("tolerance")),
                    rel_tol=_scalar(spec.get("relTol")))
            except Exception:  # noqa: BLE001
                pass
        if solvers:
            data["linear_solvers"] = solvers

    # --- turbulence ---
    for rel in ("constant/momentumTransport", "constant/turbulenceProperties"):
        tf = open_file(rel)
        if tf is not None:
            sim = _get(tf, "simulationType")
            if sim is not None:
                data["turbulence_type"] = str(sim)
            # `RASModel` / `LESModel`, which is what OpenFOAM actually calls
            # them. The key here was `model`, so `turbulence_model` has been
            # `None` on every RAS case in the corpus — `pisoFoam/RAS/cavity` and
            # `simpleFoam/pitzDaily` included — while `RASModel kEpsilon;` sat in
            # the dictionary being read. Found independently by two solver
            # families at once, and the asymmetry is telling: the fault injector
            # already writes `RAS/RASModel` correctly, so the writer and the
            # reader disagreed about the same key.
            model = (_get(tf, "RAS", "RASModel") or _get(tf, "LES", "LESModel")
                     or _get(tf, "RAS", "model") or _get(tf, "LES", "model")
                     or _get(tf, "model"))
            if model is not None:
                data["turbulence_model"] = str(model)
            break

    # --- transport ---
    tp = open_file("constant/transportProperties")
    if tp is not None:
        transport = {}
        for name in ("nu", "rho", "Pr", "Prt"):
            val = _scalar(_get(tp, name))
            if val is not None:
                transport[name] = val
        if transport:
            data["transport"] = transport

    # --- boundary conditions (0/ field files) ---
    boundary: dict[str, dict[str, str]] = {}
    zero = os.path.join(case_dir, "0")
    if os.path.isdir(zero):
        for fname in os.listdir(zero):
            fpath = os.path.join(zero, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                ff = FoamFile(fpath)
                _parse(ff, fpath)       # VER-14: noted once read, parsed or not
                for patch, spec in _subdict_items(ff, "boundaryField"):
                    t = _tokens(spec.get("type")) if hasattr(spec, "get") else None
                    if t:
                        boundary.setdefault(patch, {})[fname] = t
            except Exception:  # noqa: BLE001
                continue
    if boundary:
        data["boundary"] = boundary

    if unreadable:
        data["unreadable"] = unreadable
    # --- thermophysicalProperties: the compressible and buoyant families ---
    #
    # `transport` was read only from `constant/transportProperties`, which
    # compressible and buoyant cases **do not have**. Their viscosity is `mu`,
    # not `nu`, and it lives at `mixture/transport/mu` in a differently shaped
    # file, with `Cp` under `mixture/thermodynamics`. Measured on real v2406
    # tutorials: `CaseConfig.from_foam(...).transport == {}` for both
    # `rhoPimpleFoam/RAS/cavity` and `buoyantPimpleFoam/hotRoom`.
    #
    # The consequence is a permanent blind spot rather than a false positive:
    # `nu_out_of_range` and the `physics.nu` finding cannot fire on any solver
    # in either family, so a units error in `mu` — the single most common
    # real mistake there is, and the exact fault `PHYS.TRANSPORT_UNITS` names —
    # was undetectable. Reported independently by both families.
    thermo = open_file("constant/thermophysicalProperties")
    if thermo is not None:
        properties: dict[str, float] = dict(data.get("transport") or {})
        for group, keys in (("transport", ("mu", "Pr")),
                            ("thermodynamics", ("Cp", "Hf")),
                            ("specie", ("molWeight",))):
            for key in keys:
                value = _scalar(_get(thermo, "mixture", group, key))
                if value is not None:
                    properties.setdefault(key, value)
        if properties:
            data["transport"] = properties

    return CaseConfig.from_dict({k: v for k, v in data.items() if v is not None})
