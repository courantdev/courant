"""The evidence pack, v0: reproducible to the byte (TRD REP-4, REP-5, RRD RM-12).

    pack/
      verdict.json      the canonical verdict
      report.md         the evidence report
      inputs.manifest   sha256, bytes and path of every input, then how it was produced

Given the same inputs, a pack re-verifies to identical bytes. `check_pack`
proves it, or names the first thing that changed. That is what lets a
reviewer re-derive a verdict next year without trusting us.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from courant.verify import about
from courant.verify.render import render_markdown
from courant.verify.schema import canonical_json
from courant.verify.verdict2 import SCHEMA_V1, AnyVerdict, in_schema

VERDICT, REPORT, MANIFEST = "verdict.json", "report.md", "inputs.manifest"


SPEC = "spec.json"


def write_pack(verdict: AnyVerdict, out_dir: str | Path, target_name: str,
               case_json_name: str | None = None, *, spec_path: str | Path | None = None,
               spec_locked_sha256: str | None = None) -> Path:
    """Write the pack. A task specification travels inside it, so the pack re-derives on its
    own; its hash is already bound into the verdict's conformance block."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / VERDICT).write_text(verdict.to_json())
    (out / REPORT).write_text(render_markdown(verdict))
    lines = [f"{f.sha256}  {f.bytes}  {f.path}" for f in verdict.run.inputs]
    meta = {"tool": verdict.tool.name, "version": verdict.tool.version, "schema": verdict.schema_,
            "source": verdict.run.source, "target": target_name, "case_json": case_json_name,
            "error_rates": about.error_rates()}                 # GATE-12
    if spec_path is not None:
        (out / SPEC).write_bytes(Path(spec_path).read_bytes())
        meta["spec"] = SPEC
        if spec_locked_sha256:
            meta["spec_locked_sha256"] = spec_locked_sha256
    (out / MANIFEST).write_text("\n".join(lines) + "\n# meta " + canonical_json(meta))
    return out


def read_manifest(pack: Path) -> tuple[list[tuple[str, int, str]], dict]:
    entries, meta = [], {}
    for line in (pack / MANIFEST).read_text().splitlines():
        if line.startswith("# meta "):
            meta = json.loads(line[len("# meta "):])
        elif line.strip():
            sha, size, path = line.split("  ", 2)
            entries.append((sha, int(size), path))
    return entries, meta


def check_pack(pack_dir: str | Path, root: str | Path) -> tuple[bool, str]:
    """Re-verify a pack from its inputs under `root`. (identical, message).

    The verdict is re-derived in the schema the pack was written in (its manifest's `schema`),
    so a `courant.verdict/1` pack still checks byte for byte (ARC-16)."""
    from courant.verify import verify_path

    pack, root = Path(pack_dir), Path(root)
    entries, meta = read_manifest(pack)
    for sha, size, rel in entries:
        p = root / rel
        if not p.is_file():
            return False, f"input missing: {rel}"
        data = p.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            return False, f"input changed: {rel} (sha256 {hashlib.sha256(data).hexdigest()[:12]} != {sha[:12]})"
    # OPS-8: a verdict carries the version that made it, so another version cannot re-derive its
    # bytes. Name the one to install (every release stays installable) instead of a byte offset.
    from courant import DISTRIBUTION, __version__

    made_by = meta.get("version")
    if made_by and made_by != __version__:
        return False, (f"the pack was made by courant {made_by} and this is {__version__}: "
                       f"re-derive it with that version (pip install {DISTRIBUTION}=={made_by})")
    spec = pack / meta["spec"] if meta.get("spec") else None
    if spec is not None and not spec.is_file():
        return False, f"the pack names a specification that is missing: {meta['spec']}"
    locked = meta.get("spec_locked_sha256")
    if meta.get("source") == "case_dir":
        verdict = verify_path(root, spec=spec, spec_locked_sha256=locked)
    else:
        cj = root / meta["case_json"] if meta.get("case_json") else None
        verdict = verify_path(root / meta["target"], cj, spec=spec, spec_locked_sha256=locked)
    verdict = in_schema(verdict, 1 if meta.get("schema") == SCHEMA_V1 else 2)
    fresh = verdict.to_json()
    stored = (pack / VERDICT).read_text()
    if fresh != stored:
        for i, (a, b) in enumerate(zip(fresh, stored)):
            if a != b:
                return False, f"verdict differs at byte {i}: ...{stored[max(0, i - 40):i + 40]!r}"
        return False, "verdict differs in length"
    if render_markdown(verdict) != (pack / REPORT).read_text():
        return False, "report.md does not match the verdict"
    return True, f"identical: {len(entries)} inputs re-hashed, verdict re-derived byte for byte"
