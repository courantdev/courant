"""Seeded generators for the gate's invariants at scale (research/notes/invariants-070).

`hypothesis` is not in `packaging/requirements-dev.lock`, and adding a dependency needs a
hash-locked lock and a reason, so every generated case here is a pure function of its index:
`log_case(k)`, `dir_case(k)` and `spec_for(k, ...)` rebuild case k from a `random.Random`
seeded by a string that names it, without generating the others. A failure reproduces from the
id it prints; the fast sample of each test is a prefix of its slow run.

Inputs are the committed real runs only:

- `certification/corpus/*.log.gz` with their `.case.json` sidecars (the 31 runs the gate's
  error rates are measured on), mutated by `tests/robust/logfuzz.py`'s operators and by changes
  to the sidecar's configuration view;
- the case directories under `tests/fixtures/` that hold a solver log, copied once per session
  with their logs unpacked (`find_log` ignores `log.*.gz`).
"""

from __future__ import annotations

import atexit
import copy
import gzip
import hashlib
import json
import random
import shutil
import tempfile
from functools import cache, lru_cache
from pathlib import Path

from tests.robust import logfuzz

ROOT = Path(__file__).resolve().parents[2]

#: accept > insufficient > reject: what "lower" means for every invariant here (PRD 2.0: a
#: specification, or a model-derived input, may only move a verdict down this order).
RANK = {"accept": 2, "insufficient": 1, "reject": 0}
RANK_V1 = {"TRUSTWORTHY": 2, "NOT_ASSESSABLE": 1, "NOT_TRUSTWORTHY": 0}

#: The corpus run that takes ~9 s a verdict; left out of the generated stream, which is about
#: breadth (it is still one of the 31 in `tests/unit/test_checks_next.py`).
_SLOW_RUNS = ("pimpleFoam_planarContraction__healthy",)


@lru_cache(maxsize=1)
def corpus() -> tuple[tuple[str, str], ...]:
    """(log, sidecar) relative paths of the certification corpus, sorted."""
    out = []
    for log in sorted((ROOT / "certification/corpus").glob("*.log.gz")):
        stem = log.name[: -len(".log.gz")]
        if stem in _SLOW_RUNS:
            continue
        side = log.with_name(stem + ".case.json")
        out.append((str(log.relative_to(ROOT)), str(side.relative_to(ROOT))))
    return tuple(out)


#: Case directories under tests/fixtures with a solver log, each verified in about 0.1 s.
CASE_DIRS = (
    "tests/fixtures/buoyant/hotRoom_laminar_044",
    "tests/fixtures/moving_mesh/mixer_moving",
    "tests/fixtures/moving_mesh/mixer_static",
    "tests/fixtures/real_openfoam/cavity_extreme_aspect_ratio",
    "tests/fixtures/real_openfoam/cht_plate/early",
    "tests/fixtures/real_openfoam/duct/converged",
    "tests/fixtures/real_openfoam/duct/early",
    "tests/fixtures/real_openfoam/heated_cavity/run",
    "tests/fixtures/real_openfoam/hot_box/early",
    "tests/fixtures/real_openfoam/wedge_pipe/energy",
)

_TMP: list[Path] = []


def scratch() -> Path:
    """One temporary directory for the session, removed at exit."""
    if not _TMP:
        _TMP.append(Path(tempfile.mkdtemp(prefix="courant-inv-")))
        atexit.register(shutil.rmtree, _TMP[0], True)
    return _TMP[0]


def unpack(src: Path, dst: Path) -> Path:
    """A copy of a fixture case with its solver logs unpacked, as a user's case holds them."""
    shutil.copytree(src, dst, symlinks=True)
    for lg in list(dst.glob("log.*.gz")):
        (dst / lg.name[:-3]).write_bytes(gzip.decompress(lg.read_bytes()))
        lg.unlink()
    return dst


@cache
def pristine(rel: str) -> Path:
    """The session's unpacked copy of a fixture case; never mutated (copy it first)."""
    name = rel.replace("/", "__")
    return unpack(ROOT / rel, scratch() / "pristine" / name / Path(rel).name)


# --- log cases -------------------------------------------------------------------------------

def _mutate_view(view, rng: random.Random, n: int):
    """`n` random edits of a sidecar's configuration view (a JSON tree): a number scaled by a
    power of ten or negated, a boolean flipped, a string swapped for another string of the view,
    a key dropped. Validation may then refuse the view, which the verifier must say."""
    view = copy.deepcopy(view)
    strings: list[str] = []

    def leaves(node, path):
        if isinstance(node, dict):
            for k in sorted(node):
                yield from leaves(node[k], (*path, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from leaves(v, (*path, i))
        else:
            if isinstance(node, str):
                strings.append(node)
            yield path, node

    found = list(leaves(view, ()))
    if not found:
        return view
    for _ in range(n):
        path, value = rng.choice(found)
        if not path:
            continue
        parent = view
        try:
            for p in path[:-1]:
                parent = parent[p]
            key = path[-1]
            if isinstance(value, bool):
                parent[key] = not value
            elif isinstance(value, (int, float)):
                parent[key] = rng.choice([value * 10 ** rng.randint(-6, 6), -value, 0])
            elif isinstance(value, str) and strings:
                parent[key] = rng.choice(strings)
            elif isinstance(parent, dict):
                del parent[key]
        except (KeyError, IndexError, TypeError):
            continue
    return view


def log_case(k: int) -> tuple[str, Path, Path]:
    """Case k of the log stream: (id, log path, sidecar path), written under `scratch()`.

    Two in five are an unmutated corpus run; the rest carry one to three of logfuzz's operators
    and, half of the time, one to three edits of the sidecar's configuration view."""
    rng = random.Random(f"invariants:log:{k}")
    log_rel, side_rel = rng.choice(corpus())
    data = gzip.decompress((ROOT / log_rel).read_bytes())
    ops: list[str] = []
    if rng.random() >= 0.4:
        for name in rng.sample(sorted(logfuzz.RANDOM_OPS), rng.randint(1, 3)):
            data = logfuzz.RANDOM_OPS[name](data, rng)
            ops.append(name)
    side = json.loads((ROOT / side_rel).read_text())
    if ops and rng.random() < 0.5 and isinstance(side.get("config"), dict):
        n = rng.randint(1, 3)
        side["config"] = _mutate_view(side["config"], rng, n)
        ops.append(f"view*{n}")
    stem = Path(log_rel).name[: -len(".log.gz")]
    where = scratch() / "logs" / str(k)
    where.mkdir(parents=True, exist_ok=True)
    log, sidecar = where / f"{stem}.log", where / f"{stem}.case.json"
    log.write_bytes(data)
    sidecar.write_text(json.dumps(side, sort_keys=True))
    return f"log[{k}]:{stem}+{','.join(ops) or 'as-run'}", log, sidecar


# --- case directories and specifications ------------------------------------------------------

def dir_case(k: int) -> tuple[str, Path]:
    """Case k of the directory stream: (id, a pristine unpacked fixture case)."""
    rng = random.Random(f"invariants:dir:{k}")
    rel = rng.choice(CASE_DIRS)
    return rel, pristine(rel)


@cache
def drafted(rel: str, checks: str) -> tuple[dict, ...]:
    """`courant spec init`'s items for a fixture case, as JSON data (deterministic code)."""
    from courant.spec.check import draft_spec

    spec = draft_spec(pristine(rel), checks=checks)
    return tuple(i.model_dump(mode="json", exclude_none=True) for i in spec.items)


def _perturb(value, rng: random.Random):
    """A value near or far from `value`, of the same kind where it can be."""
    scale = rng.choice([1.0, 1.0005, 1.02, 0.5, 10.0, -1.0])
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value * scale if value else rng.choice([0.0, 1e-3, 1.0])
    if isinstance(value, list):
        return [v * scale if isinstance(v, (int, float)) else v for v in value]
    if isinstance(value, str):
        return rng.choice([value, value + "X", "kOmegaSST", "laminar", "simpleFoam"])
    return value


#: What a generated specification says about where it came from (SPEC-7). `extracted` is a
#: model's draft nobody confirmed; `confirmed` one a person confirmed with `courant spec
#: confirm`; None a hand-written or `courant spec init` specification.
PROVENANCES = (None, "extracted", "confirmed")


def spec_for(k: int, rel: str, checks: str = "default", *,
             provenance: str | None | type = type) -> tuple[str, dict]:
    """Specification k for the fixture case `rel`: (what was done, spec as JSON data).

    Drawn from the case's own draft (`drafted`): a random subset of its items, each kept,
    perturbed (a little, within or beyond the 1% tolerance, or by a factor), pointed at a key or
    file the case does not hold (unreadable, VER-5.9), or made impossible (`min` above `max`,
    SPEC-9). `provenance` fixes SPEC-7's provenance; by default it is drawn."""
    rng = random.Random(f"invariants:spec:{k}:{rel}:{checks}")
    base = list(drafted(rel, checks))
    what: list[str] = []
    items = []
    if base:
        for item in rng.sample(base, rng.randint(1, len(base))):
            item = dict(item)
            r = rng.random()
            if r < 0.45:
                pass
            elif r < 0.75 and "value" in item:
                item["value"] = _perturb(item["value"], rng)
                what.append(f"perturb:{item['name']}")
            elif r < 0.85 and "key" in item:
                item["key"] = item["key"] + "Missing"
                what.append(f"unreadable:{item['name']}")
            elif r < 0.92:
                item["file"] = "constant/noSuchDict"
                what.append(f"nofile:{item['name']}")
            elif isinstance(item.get("value"), (int, float)) and not item.get("sha256"):
                item.pop("value")
                item["min"], item["max"] = 2.0, 1.0
                what.append(f"impossible:{item['name']}")
            items.append(item)
    if not items:
        items = [{"name": "nu", "file": "constant/transportProperties", "key": "nu",
                  "value": 1e-5}]
        what.append("fallback")
    data: dict = {"schema": "courant.spec/1", "task": f"generated {k}", "items": items}
    prov = rng.choice(PROVENANCES) if provenance is type else provenance
    if prov == "extracted":
        data["provenance"] = {"extracted_by": "courant spec extract 0.7.0",
                              "model": "example/model:free", "date": "2026-11-04"}
    elif prov == "confirmed":
        data["provenance"] = {"extracted_by": "courant spec extract 0.7.0",
                              "model": "example/model:free", "date": "2026-11-04",
                              "confirmed_by": "S", "confirmed_date": "2026-11-04"}
    what.append(f"prov:{prov}")
    return ";".join(what), data


def write_spec(data: dict, k: int) -> tuple[Path, str]:
    """The specification written to a file as `courant spec` writes it; (path, sha256)."""
    from courant.spec.model import Spec

    text = Spec.model_validate(data).to_json()
    where = scratch() / "specs"
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"spec-{k}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.json"
    path.write_text(text)
    return path, hashlib.sha256(text.encode()).hexdigest()


def lock_for(k: int, sha: str) -> str | None:
    """The lock a generated run was given: none, the specification's own hash, or another's."""
    r = random.Random(f"invariants:lock:{k}").random()
    if r < 0.3:
        return None
    if r < 0.85:
        return sha
    return hashlib.sha256(f"another spec {k}".encode()).hexdigest()


def ids(n: int) -> list[int]:
    return list(range(n))
