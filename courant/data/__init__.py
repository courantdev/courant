"""Data that ships inside the package, so an installed copy works outside this repository (TRD PKG-5).

- `certification.json` is a byte-for-byte copy of the measured result RM-01 (the scorecard on
  the real-run corpus). It is what `courant verify --about` and the MCP server's
  `verifier_certification` print.
- `certified_codes.json` lists the failure modes named on real solver output (tier 1 of
  `docs/COVERAGE.md`).
- `demo/` holds the runs `courant demo` verifies: two corpus logs, and one monitored steady
  case as a deterministic tarball.
- `schemas/` holds the JSON Schemas (draft 2020-12) of what the gate writes and reads: the
  verdicts, the task specification, preflight, convergence, the audit, the batch cache, the
  record, the support bundle, the error rates and the release signature.

`scripts/package_data.py` writes all of it, and `--check` fails if any packaged file differs
from what its source would produce. A packaged number cannot drift from the measured one.
"""

from __future__ import annotations

import json
import tarfile
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent
DEMO = DATA / "demo"
DEMO_CASE = "pitzDaily_fixed_budget"


@lru_cache(maxsize=1)
def certification() -> dict | None:
    p = DATA / "certification.json"
    return json.loads(p.read_text()) if p.exists() else None


@lru_cache(maxsize=1)
def certified_codes() -> frozenset[str]:
    p = DATA / "certified_codes.json"
    return frozenset(json.loads(p.read_text())["codes"]) if p.exists() else frozenset()


def extract_demo_case(dest: Path) -> Path:
    """Unpack the monitored demo case into `dest` and return its directory."""
    with tarfile.open(DEMO / f"{DEMO_CASE}.tar.gz", "r:gz") as tar:
        tar.extractall(dest, filter="data")
    return dest / DEMO_CASE
