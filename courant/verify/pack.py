"""The evidence pack, v0: reproducible to the byte (TRD REP-4, REP-5, RRD RM-12).

    pack/
      verdict.json      the canonical verdict
      report.md         the evidence report
      inputs.manifest   sha256, bytes and path of every input, then how it was produced

Given the same inputs, a pack re-verifies to identical bytes. `check_pack`
proves it, or names the first thing that changed. That is what lets a
reviewer re-derive a verdict next year without trusting us.

Nor trusting the pack: its manifest is a claim by whoever handed it over
(REP-5). The reviewer names the run (`--root`), and the root decides how it is
judged wherever it can: a case directory is judged as one, so a pack of its
log alone is refused when `courant verify` reads anything else there; a log
goes with the sidecar the root holds for it (`sidecar_of`), or with none;
every listed file resolves under the root; and the listed lines must be the
re-derived verdict's inputs, line for line, so a pack that lists none is
refused. The reviewer may give the hash the specification was locked at
(`--spec-sha256`), and the pack is held to it.

What the pack still decides, and the OK line says: in a directory of runs (the
corpus), which log it judged, by a file name in printable ASCII directly under
the root; and, with no hash from the reviewer, whether it was judged against a
specification and at which locked hash.
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
    lines = [f"{sha}  {size}  {path}" for sha, size, path in _entries(verdict)]
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


def _entries(verdict: AnyVerdict) -> list[tuple[str, int, str]]:
    """The manifest's lines for `verdict`: its inputs, in its order (one implementation, for
    the writer and the check)."""
    return [(f.sha256, f.bytes, f.path) for f in verdict.run.inputs]


def read_manifest(pack: Path) -> tuple[list[tuple[str, int, str]], dict]:
    entries, meta = [], {}
    for line in (pack / MANIFEST).read_text().splitlines():
        if line.startswith("# meta "):
            meta = json.loads(line[len("# meta "):])
        elif line.strip():
            sha, size, path = line.split("  ", 2)
            entries.append((sha, int(size), path))
    return entries, meta


def _file_name(name: object) -> bool:
    """A bare file name a reviewer reads as it is on disk: printable ASCII with no space at either
    end (no zero-width or control character), no directory part, not `.` or `..`, not absolute,
    on any platform."""
    return (isinstance(name, str) and name not in ("", ".", "..") and name.isascii()
            and name.isprintable() and name == name.strip()
            and not any(sep in name for sep in ("/", "\\")))


def _sidecar_problem(root: Path, log: str, named: str | None) -> str | None:
    """Why a log pack's sidecar is not the one the root holds for its log, or None. The root's is
    `sidecar_of(log)` when it is there, else none: a pack neither leaves it out nor names another."""
    from courant.verify.ingest import sidecar_of

    side = sidecar_of(root / log)
    held = side.name if side.is_file() else None
    if named == held:
        return None
    if named is None:
        return (f"the pack judged {log} without its sidecar: the root holds {held}, the sidecar "
                f"of {log}, and a log is checked with the sidecar the root holds for it")
    if held is None:
        return (f"the pack names its sidecar as {named!r}, and the root holds no {side.name}, the "
                f"only sidecar {log} is checked with")
    return f"the pack names its sidecar as {named!r}; the root's sidecar for {log} is {held}"


def _case_reads(root: Path) -> list[str] | None:
    """The files `courant verify` reads when it judges `root` as a case directory, or None when
    it cannot judge it as one (no solver log in it: a directory of runs, like the corpus)."""
    from courant.verify.ingest import InputError, from_case_dir

    try:
        return [f.path for f in from_case_dir(root).inputs]
    except InputError:
        return None


def _under(root: Path, rel: str) -> bool:
    """`rel` names a file under `root` with every link followed: relative, no `..`, and not
    leaving the root through a link (the same resolution as MCP-3's confinement)."""
    from courant.case.reads import Confinement, OutsideRoot

    if "\0" in rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return False
    try:
        Confinement([root]).check(root / rel)
    except (OutsideRoot, OSError):
        return False
    return True


def _difference(listed: list[tuple[str, int, str]], read: list[tuple[str, int, str]]) -> str:
    """How the manifest's lines differ from the inputs the re-derived verdict read."""
    by_listed, by_read = {e[2]: e for e in listed}, {e[2]: e for e in read}
    for path in sorted(by_listed.keys() | by_read.keys()):
        if path not in by_listed:
            return f"{path} was read and is not listed"
        if path not in by_read:
            return f"{path} is listed and was not read"
        if by_listed[path] != by_read[path]:
            return f"{path} is listed with other bytes than were read"
    return "its lines are not the verdict's inputs line for line"


def check_pack(pack_dir: str | Path, root: str | Path, *,
               spec_sha256: str | None = None) -> tuple[bool, str]:
    """Re-verify a pack from its inputs under `root`. (identical, message).

    `root` is the run the reviewer names, and `spec_sha256`, when given, the hash the reviewer
    locked the specification at; the pack is held to both (REP-5), and the OK line says how the
    run was judged. The verdict is re-derived in the schema the pack was written in (its
    manifest's `schema`), so a `courant.verdict/1` pack still checks byte for byte (ARC-16)."""
    from courant.spec import spec_sha256 as sha_of
    from courant.verify import verify_path

    pack, root = Path(pack_dir), Path(root)
    if not root.is_dir():
        return False, f"the root is not a directory: {root}"
    entries, meta = read_manifest(pack)
    case_dir = meta.get("source") == "case_dir"
    target, sidecar = meta.get("target"), meta.get("case_json") or None
    named = []                                          # the files a log pack verifies
    if not case_dir:                                    # a case pack verifies the root itself
        for name, what in ((target, "its log"), (sidecar, "its sidecar")):
            if name is None and what == "its sidecar":
                continue
            if not _file_name(name):
                return False, (f"the pack names {what} as {name!r}: a pack names it by a file "
                               "name in printable ASCII, directly under the root, and only that "
                               "file is verified")
            named.append(name)
        problem = _sidecar_problem(root, target, sidecar)
        if problem:
            return False, problem
    if meta.get("spec") and not _file_name(meta["spec"]):
        return False, (f"the pack names its specification as {meta['spec']!r}: a pack names it "
                       "by a file name in printable ASCII, inside the pack")
    if not entries:
        return False, ("the manifest lists no inputs: nothing ties the pack's verdict to a run "
                       "under the root")
    for rel in [*named, *(rel for _sha, _size, rel in entries)]:
        if not _under(root, rel):
            return False, (f"input outside the root: {rel} does not resolve to a file under the "
                           "root; a pack is checked only on files under it")
    if not case_dir and not (root / target).is_file():  # the files hashed are the files read
        missing = not (root / target).exists()
        return False, (f"the pack names its log as {target!r}, which is "
                       f"{'missing' if missing else 'not a file'} under the root")
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
    carried = sha_of(spec) if spec is not None else None
    locked = (meta.get("spec_locked_sha256") or "").strip().lower() or None
    given = spec_sha256.strip().lower() if spec_sha256 else None
    if given:                                           # the reviewer's lock, not the pack's
        if carried is None:
            return False, (f"the pack carries no specification, and the specification was locked "
                           f"at {given[:12]}: its verdict was not judged against the task")
        if carried != given:
            return False, (f"the pack's specification is not the one locked: its sha256 is "
                           f"{carried[:12]}, and the specification was locked at {given[:12]}")
        if locked != given:
            return False, ("the pack's verdict was judged against its specification "
                           + (f"locked at {locked[:12]}" if locked else "not locked")
                           + f", and the specification was locked at {given[:12]}")
    if not case_dir:                                    # a case directory is judged as one
        extra = sorted(set(_case_reads(root) or ()) - {rel for _sha, _size, rel in entries})
        if extra:
            more = f" and {len(extra) - 3} more" if len(extra) > 3 else ""
            return False, (f"the root is a case directory: `courant verify` reads "
                           f"{', '.join(extra[:3])}{more} there, which this log pack's verdict "
                           "did not; a case directory is re-verified from its own pack "
                           "(courant verify CASE --pack DIR)")
    if case_dir:
        verdict = verify_path(root, spec=spec, spec_locked_sha256=locked)
        what, scope = f"the case directory {root}", []
    else:
        cj = root / sidecar if sidecar else None
        verdict = verify_path(root / target, cj, spec=spec, spec_locked_sha256=locked)
        what = f"{target} under {root}"
        scope = [f"a log with its sidecar {sidecar}" if sidecar else "a log, no sidecar"]
    scope.append("no specification" if carried is None else (
        f"specification {carried[:12]}, " + ("locked at the hash you gave" if given
                                             else "locked at the hash the pack carries"
                                             if locked else "not locked")))
    verdict = in_schema(verdict, 1 if meta.get("schema") == SCHEMA_V1 else 2)
    if _entries(verdict) != entries:
        return False, ("the manifest is not the inputs the verdict read: "
                       + _difference(entries, _entries(verdict)))
    fresh = verdict.to_json()
    stored = (pack / VERDICT).read_text()
    if fresh != stored:
        for i, (a, b) in enumerate(zip(fresh, stored)):
            if a != b:
                return False, f"verdict differs at byte {i}: ...{stored[max(0, i - 40):i + 40]!r}"
        return False, "verdict differs in length"
    if render_markdown(verdict) != (pack / REPORT).read_text():
        return False, "report.md does not match the verdict"
    return True, (f"identical: {what} re-verified ({', '.join(scope)}), {len(entries)} inputs "
                  "re-hashed, verdict re-derived byte for byte")
