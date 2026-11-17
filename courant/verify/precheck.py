"""`courant precheck DIR`: the pre-check intake (TRD GATE-14).

The prospect runs it on their own machine. Every run under DIR is verified (`batch.runs_under`,
`batch.verify_batch`, no cache) and one deterministic zip is built:

    summary.json               courant.precheck/1: the version, the published error rates
                               (GATE-12), and per run its place, verdict, basis, rule and reasons
    summary.md                 rendered from summary.json alone (`render_summary_md`)
    packs/run-NN/              the evidence pack `courant verify --pack` writes (`pack_files`)
    logs/run-NN/<log>          only with --include-logs: the solver log, byte for byte
    MANIFEST.sha256            `sha256sum` lines for every other member, sorted

A run that cannot be read (`InputError`) is an `insufficient` row with no pack, no basis, no
rule and no reasons, and the sentence that says why (D1). No pack, summary or manifest holds
DIR's absolute path; a log `--include-logs` adds is copied byte for byte (D3, D5). `redact=True` replaces each run's place and own names with a label (`run-01`, ...), field
by field and never by substitution over JSON (D2); the label map is returned to be kept apart.

`check_archive` is Courant's side: it reads the archive in memory, never extracts it, and checks
the layout, the manifest both ways, the summary, every verdict, each row against its pack, the
logs, and that summary.md is the summary's rendering.

Nothing here opens a network connection or writes inside a run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import lzma
import os
import re
import zipfile
import zlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

SCHEMA = "courant.precheck/1"
SUMMARY_JSON, SUMMARY_MD, MANIFEST = "summary.json", "summary.md", "MANIFEST.sha256"
PACK_FILES = ("verdict.json", "report.md", "inputs.manifest")

#: D4: every member is stored, dated 1980-01-01, mode 0644, made on one system.
DATE_TIME = (1980, 1, 1, 0, 0, 0)
EXTERNAL_ATTR = (0o100644 & 0xFFFF) << 16
CREATE_SYSTEM = 3

#: `--check` reads members in memory: no member above this, no more than `MAX_KEPT_BYTES` kept
#: in all, no archive above the total. A solver log (`logs/`) is hashed as it streams and never
#: kept, so it may be larger.
MAX_MEMBER_BYTES = 16 * 2**20
MAX_KEPT_BYTES = 256 * 2**20
MAX_LOG_BYTES = 512 * 2**20
MAX_TOTAL_BYTES = 2 * 2**30

_LABEL = r"run-\d{2,}"
_PACK_MEMBER = re.compile(rf"packs/({_LABEL})/(verdict\.json|report\.md|inputs\.manifest)")
_LOG_MEMBER = re.compile(rf"logs/({_LABEL})/([^/\\\x00]+)")
_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")


# ------------------------------------------------------------------ the summary
class _AsGiven(BaseModel):
    """A closed model of a document whose optional keys are absent, not null, when not given:
    it dumps back to the keys it was given, so its canonical bytes do not change."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_serializer(mode="wrap")
    def _as_given(self, handler):
        out = handler(self)
        unset = {k for k in type(self).model_fields if k not in self.model_fields_set}
        unset |= {f.alias for k, f in type(self).model_fields.items() if k in unset and f.alias}
        return {k: v for k, v in out.items() if k not in unset}


class ErrorRate(_AsGiven):
    """One rate as `about.error_rates` states it: n of `of`, and its 95% interval when known.
    An interval bound is absent or a finite number, never null, NaN or infinite."""

    model_config = ConfigDict(allow_inf_nan=False)

    n: int
    of: int
    lo: int | float = None  # absent when not known; null is refused
    hi: int | float = None


class ErrorRatesSource(_AsGiven):
    """`about.error_rates()["measured_on"]`: which measured result the rates are read from."""

    verifier_version: str | None
    tag: str | None
    date: str | None
    corpus: str | None
    policy: str | None
    source: str


class ErrorRates(_AsGiven):
    """`courant.error-rates/1` as `about.error_rates` writes it (GATE-12). An archive is not
    trusted to hold this shape, and `about.sentences` reads it, so `--check` holds it here."""

    schema_: Literal["courant.error-rates/1"] = Field(alias="schema")
    courant_version: str
    measured: bool
    non_accept_rate_healthy: ErrorRate | None
    non_accept_rate_note: str
    note: str | None = None
    measured_on: ErrorRatesSource | None = None
    false_accepts: ErrorRate | None = None
    false_alarms: ErrorRate | None = None
    abstentions: ErrorRate | None = None

    @model_validator(mode="after")
    def _measured_rates_are_stated(self):
        if self.measured and None in (self.measured_on, self.false_accepts, self.false_alarms):
            raise ValueError("measured rates name where they were measured, the false accepts "
                             "and the false alarms")
        return self


class PrecheckReason(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: str
    summary: str


class PrecheckRow(BaseModel):
    """One run found. `pack` is set exactly when the run was read; `unreadable` exactly when not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    place: str                       # its path under DIR, or its label under --redact-paths
    verdict: Literal["accept", "reject", "insufficient"]
    basis: str | None
    rule: str | None
    reasons: list[PrecheckReason]
    pack: str | None                 # packs/run-NN
    unreadable: str | None = None    # why the run could not be read (D1)
    log: str | None = None           # logs/run-NN/<name>, with --include-logs
    log_note: str | None = None      # why its log was left out, with --include-logs


class PrecheckSummary(BaseModel):
    """`courant.precheck/1` (GATE-14)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["courant.precheck/1"] = Field(default=SCHEMA, alias="schema")
    courant_version: str
    error_rates: ErrorRates
    include_logs: bool
    redacted: bool
    runs: list[PrecheckRow]

    def to_json(self) -> str:
        from courant.verify.schema import canonical_json

        return canonical_json(self.model_dump(mode="json", by_alias=True))


def render_summary_md(doc: dict) -> str:
    """summary.md, from summary.json's document alone."""
    from courant.verify import about
    from courant.verify.render import _md_name

    rows = doc["runs"]
    n = len(rows)
    count = {w: sum(1 for r in rows if r["verdict"] == w)
             for w in ("accept", "reject", "insufficient")}
    unreadable = sum(1 for r in rows if r.get("unreadable") is not None)
    line = (f"{n} runs: {count['accept']} of {n} accept, {count['reject']} of {n} reject, "
            f"{count['insufficient']} of {n} insufficient")
    if unreadable:
        line += f" ({unreadable} of {n} could not be read)"
    out = ["# Courant pre-check", "",
           f"courant {_md_name(doc['courant_version'])}. {line}.", "",
           (f"Paths redacted: {'yes' if doc['redacted'] else 'no'}. "
            f"Solver logs included: {'yes' if doc['include_logs'] else 'no'}."), "",
           "## Runs", "",
           "| # | Run | Verdict | Basis | Rule | Pack | Reasons |",
           "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        if r.get("unreadable") is not None:
            why = "could not be read: " + _md_name(r["unreadable"])
        elif r["reasons"]:
            why = "; ".join(f"`{_md_name(x['code'])}` {_md_name(x['summary'])}"
                            for x in r["reasons"])
        else:
            why = "—"
        if r.get("log"):
            why += f" (log: {_md_name(r['log'])})"
        elif r.get("log_note"):
            why += f" (log left out: {_md_name(r['log_note'])})"
        out.append(f"| {i} | {_md_name(r['place'])} | {r['verdict']} | {r['basis'] or '—'} "
                   f"| {r['rule'] or '—'} | {r['pack'] or '—'} | {why} |")
    out += ["", "## Error rate", ""]
    out += [f"- {s}" for s in about.sentences(doc["error_rates"])]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ redaction (D2, D3)
# A verdict is redacted field by field: `run.id` becomes the label, `run.inputs[].path` is mapped
# exactly (a log run's log and sidecar names; a case's own paths, `0/U`, are kept), and the
# manifest meta's `target` and `case_json` are given the redacted names. Its place, its stem and
# its `run.id` are never searched for in its text: a run named `0`, `case` or `icoFoam` would
# rewrite `0/U`, a fixed sentence or the `solver` field. The one name a verdict's text can carry
# is a log run's own file name (a reason's `input`, `sentences.ENDED_SHORT`'s `{log}`), so only
# the log's and the sidecar's file names, each with its suffix, are replaced where they stand
# as whole names (`_rename_files`). An unreadable run's sentence is an `InputError`, not
# Courant's verdict text, and has its place and names replaced as well.
def _rename_files(node, pairs):
    """A verdict's document with the run's own file names (`pairs`, from the `.log`, `.log.gz`
    and `.case.json` names only) replaced in its string values; `run` is left to the caller."""
    if not pairs:
        return node
    if isinstance(node, dict):
        return {k: (v if k == "run" else _rename_files(v, pairs)) for k, v in node.items()}
    if isinstance(node, list):
        return [_rename_files(v, pairs) for v in node]
    if isinstance(node, str):
        return _scrub(node, pairs)
    return node


def _pattern(name: str) -> re.Pattern:
    """`name` where it stands as a whole name, not inside a longer one."""
    return re.compile(r"(?<![\w.\-])" + re.escape(name) + r"(?![\w\-])")


def _pairs(mapping: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    """Replacements, longest name first, so a path is replaced before a name inside it."""
    names = sorted((k for k in mapping if k and k not in (".", "..")), key=lambda k: (-len(k), k))
    return [(_pattern(k), mapping[k]) for k in names]


def _scrub(text: str, pairs: list[tuple[re.Pattern, str]]) -> str:
    for pat, new in pairs:
        text = pat.sub(lambda _m, new=new: new, text)
    return text


def _absolute_forms(p: Path) -> set[str]:
    out = {os.path.abspath(p), str(p.absolute())}
    try:
        out.add(str(p.resolve()))
    except OSError:
        pass
    return {f for f in out if os.path.isabs(f) and f != os.sep}


def _log_suffix(name: str) -> str:
    for suffix in (".log.gz", ".log"):
        if name.endswith(suffix):
            return suffix
    return ""


def _own_name(p: Path) -> str:
    """A path's own name, `caseA` for `caseA` and also for `.` or `..` run from inside it."""
    return p.name if p.name not in ("", ".", "..") else p.resolve().name


# ------------------------------------------------------------------ building
def label_of(i: int, n: int) -> str:
    """The pack label of the i-th run (1-based) of n, numbered in discovery order (D4)."""
    return f"run-{i:0{max(2, len(str(n)))}d}"


def build(dir: str | Path, *, jobs: int = 1, include_logs: bool = False, redact: bool = False,
          runs: list[tuple[Path, Path | None]] | None = None
          ) -> tuple[dict[str, bytes], dict[str, str], list[PrecheckRow]]:
    """(members, labels, rows) for the runs under `dir`: the archive's members by name, the label
    map {label: place} (empty unless `redact`), and the summary's rows in discovery order."""
    from courant import __version__
    from courant.cli.tail import find_log
    from courant.verify import about
    from courant.verify.batch import runs_under, verify_batch
    from courant.verify.pack import pack_files
    from courant.verify.verdict2 import as_v2, load_verdict

    if include_logs and redact:
        raise ValueError("--include-logs cannot go with --redact-paths: a solver log carries the "
                         "case's absolute path, and Courant does not edit a run's bytes")
    # GATE-14, same inputs, same bytes: DIR and each run are made absolute (lexically, links kept)
    # before anything reads them, so `caseA`, `caseA/`, `./caseA`, `/abs/caseA` and `.` from
    # inside it give one archive (a run's id is its directory's name, and `.` has none).
    root = Path(os.path.abspath(dir))
    runs = runs_under(root) if runs is None else runs
    runs = [(Path(os.path.abspath(t)), None if c is None else Path(os.path.abspath(c)))
            for t, c in runs]
    done = verify_batch(root, jobs=jobs, cache=None, runs=runs).results
    n = len(runs)
    root_is_run = len(runs) == 1 and Path(runs[0][0]) == root
    root_name = _own_name(root)
    members: dict[str, bytes] = {}
    labels: dict[str, str] = {}
    rows: list[PrecheckRow] = []
    for i, ((target, sidecar), (_t, verdict, err)) in enumerate(zip(runs, done, strict=True), 1):
        target = Path(target)
        label = label_of(i, n)
        place = root_name if root_is_run else target.relative_to(root).as_posix()
        own = _own_name(target)
        suffix = _log_suffix(own) if target.is_file() else ""
        sidecar_name = None if sidecar is None else Path(sidecar).name
        if verdict is None:                                    # D1, with D3 and D2 on its sentence
            names: dict[str, str] = {}
            for form in _absolute_forms(root):
                names[form] = "." if redact else root_name
            for form in _absolute_forms(target) | {str(target)}:
                names[form] = label if redact else place
            if redact:
                labels[label] = place
                names[place] = label
                names[own] = label + suffix
                if sidecar_name is not None:
                    names[sidecar_name] = label + ".case.json"
            rows.append(PrecheckRow(place=label if redact else place, verdict="insufficient",
                                    basis=None, rule=None, reasons=[], pack=None,
                                    unreadable=_scrub(str(err), _pairs(names))))
            continue
        verdict = as_v2(verdict)
        target_name = own
        if redact:                                             # D2: field by field, nothing else
            labels[label] = place
            exact = {own: label + suffix} if target.is_file() else {}
            if sidecar_name is not None:
                exact[sidecar_name] = label + ".case.json"
            data = verdict.model_dump(mode="json", by_alias=True)
            data["run"]["id"] = label
            for f in data["run"]["inputs"]:
                f["path"] = exact.get(f["path"], f["path"])
            data = _rename_files(data, _pairs(exact))
            verdict = load_verdict(data)
            target_name = label + suffix
            sidecar_name = None if sidecar_name is None else label + ".case.json"
        pack = f"packs/{label}"
        for name, body in pack_files(verdict, target_name, sidecar_name).items():
            members[f"{pack}/{name}"] = body
        log_member, log_note = None, None
        if include_logs:
            log_member, log_note, body = _solver_log(target, verdict, find_log)
            if log_member is not None:
                log_member = f"logs/{label}/{log_member}"
                members[log_member] = body
        rows.append(PrecheckRow(
            place=label if redact else place, verdict=verdict.verdict, basis=verdict.basis,
            rule=verdict.rule, pack=pack, log=log_member, log_note=log_note,
            reasons=[PrecheckReason(code=r.code, severity=r.severity, summary=r.summary)
                     for r in verdict.reasons]))
    summary = PrecheckSummary(courant_version=__version__, error_rates=about.error_rates(),
                              include_logs=include_logs, redacted=redact, runs=rows)
    members[SUMMARY_JSON] = summary.to_json().encode()
    members[SUMMARY_MD] = render_summary_md(json.loads(members[SUMMARY_JSON])).encode()
    members[MANIFEST] = manifest_sha256(members)
    return members, labels, rows


def _solver_log(target: Path, verdict, find_log) -> tuple[str | None, str | None, bytes]:
    """(file name, None, bytes) of the run's solver log when its bytes are the ones the verdict
    read (D5), else (None, why, b"")."""
    if target.is_dir():
        log = find_log(target, finished=True)
        if log is None:
            return None, "no solver log was found in the case", b""
        rel = log.relative_to(target).as_posix()
    else:
        log, rel = target, target.name
    listed = {f.path: f.sha256 for f in verdict.run.inputs}
    body = log.read_bytes()
    if listed.get(rel) != hashlib.sha256(body).hexdigest():
        return None, (f"the solver log {log.name} changed after it was read, or is not among "
                      "the files the verdict read; it was left out"), b""
    return log.name, None, body


# ------------------------------------------------------------------ the archive (D4)
def manifest_sha256(members: dict[str, bytes]) -> bytes:
    """`sha256sum` lines for every member but the manifest itself, sorted by name."""
    lines = [f"{hashlib.sha256(b).hexdigest()}  {n}" for n, b in sorted(members.items())
             if n != MANIFEST]
    return ("\n".join(lines) + "\n").encode()


def utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


def default_name(members: dict[str, bytes], today: dt.date) -> str:
    """`courant-precheck-<UTC date>-<12 hex digits of the manifest's SHA-256>.zip`."""
    digest = hashlib.sha256(members[MANIFEST]).hexdigest()[:12]
    return f"courant-precheck-{today.isoformat()}-{digest}.zip"


def zip_bytes(members: dict[str, bytes]) -> bytes:
    """The archive: members stored, sorted, with fixed dates, modes and system (D4)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = EXTERNAL_ATTR
            info.create_system = CREATE_SYSTEM
            zf.writestr(info, members[name])
    return buf.getvalue()


def write_zip(members: dict[str, bytes], path: str | Path) -> bytes:
    """Write the archive at `path` (whole, or not at all) and return its bytes."""
    data = zip_bytes(members)
    path = Path(path)
    tmp = path.with_name(f".{path.name}.partial")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return data


# ------------------------------------------------------------------ --check
class ArchiveError(ValueError):
    """The file is not a zip archive that can be read."""


def _read_members(path: Path) -> tuple[dict[str, bytes], dict[str, str], str | None]:
    """(bytes of every non-log member, SHA-256 of every member, the first problem or None).
    The member names are checked against the layout before any member's bytes are read."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError, EOFError) as exc:
        raise ArchiveError(f"{path} is not a readable zip archive ({type(exc).__name__})") from exc
    data: dict[str, bytes] = {}
    sha: dict[str, str] = {}
    total = kept = 0
    with zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                return data, sha, f"{name} is in the archive twice"
            seen.add(name)
        problem = _layout_problem(sorted(names))
        if problem:
            return data, sha, problem
        for info in infos:
            name = info.filename
            keep = not name.startswith("logs/")
            cap = MAX_MEMBER_BYTES if keep else MAX_LOG_BYTES
            if info.file_size > cap:
                return data, sha, f"{name} is larger than {cap} bytes"
            h, size = hashlib.sha256(), 0
            chunks = []
            try:
                with zf.open(info) as fh:
                    while chunk := fh.read(1 << 20):
                        size += len(chunk)
                        total += len(chunk)
                        kept += len(chunk) if keep else 0
                        if size > cap or total > MAX_TOTAL_BYTES or kept > MAX_KEPT_BYTES:
                            return data, sha, f"{name} goes past the size {path.name} may have"
                        h.update(chunk)
                        if keep:
                            chunks.append(chunk)
            # RuntimeError: an encrypted member, which zipfile will not open without a password;
            # EOFError: a member shorter than its stated size; zlib/lzma: a corrupt stream
            except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError, RuntimeError,
                    EOFError, zlib.error, lzma.LZMAError) as exc:
                return data, sha, f"{name} cannot be read ({type(exc).__name__})"
            sha[name] = h.hexdigest()
            if keep:
                data[name] = b"".join(chunks)
    return data, sha, None


def _layout_problem(names: list[str]) -> str | None:
    for name in names:
        if name in (SUMMARY_JSON, SUMMARY_MD, MANIFEST) or _PACK_MEMBER.fullmatch(name):
            continue
        m = _LOG_MEMBER.fullmatch(name)
        if m and m.group(2) not in (".", ".."):
            continue
        return f"{name!r} is not a member a pre-check archive holds"
    for top in (SUMMARY_JSON, SUMMARY_MD, MANIFEST):
        if top not in names:
            return f"{top} is missing"
    return None


def _manifest_problem(data: dict[str, bytes], sha: dict[str, str]) -> str | None:
    try:
        text = data[MANIFEST].decode("utf-8")
    except UnicodeDecodeError:
        return f"{MANIFEST} is not text"
    if not text.endswith("\n"):
        return f"{MANIFEST} does not end with a line break"
    listed: dict[str, str] = {}
    order = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _MANIFEST_LINE.fullmatch(line)
        if not m:
            return f"{MANIFEST} line {i} is not a sha256sum line"
        if m.group(2) in listed:
            return f"{m.group(2)} is listed twice in {MANIFEST}"
        listed[m.group(2)] = m.group(1)
        order.append(m.group(2))
    for name in sorted(sha):
        if name != MANIFEST and name not in listed:
            return f"{name} is in the archive and not listed in {MANIFEST}"
    for name in order:
        if name not in sha or name == MANIFEST:
            return f"{name} is listed in {MANIFEST} and is not in the archive"
        if sha[name] != listed[name]:
            return f"{name} does not match its SHA-256 in {MANIFEST}"
    if order != sorted(order):
        return f"{MANIFEST} is not sorted by name"
    return None


#: The manifest meta's keys, as `pack.pack_files` writes them for a pack with no specification.
_META_KEYS = {"tool", "version", "schema", "source", "target", "case_json", "error_rates"}


def _pack_problem(pack: str, verdict, doc: PrecheckSummary, data: dict[str, bytes],
                  rendered: bool) -> str | None:
    """Why a pack is not one `pack_files` wrote for its verdict, by pack.py's own checks: its
    manifest parses (`parse_manifest`), lists the verdict's inputs line for line (`_entries`),
    and its meta names this tool, the verdict's version, schema and source, file names a pack
    may carry (`_file_name`), and the summary's error rates. With `rendered` (the archive is this
    version's), its three files are also the bytes `pack_files` gives for the verdict."""
    from courant.verify.pack import (
        ManifestError,
        _difference,
        _entries,
        _file_name,
        pack_files,
        parse_manifest,
    )
    from courant.verify.schema import ToolInfo

    mname = f"{pack}/inputs.manifest"
    vname = f"{pack}/verdict.json"
    try:
        entries, meta = parse_manifest(data[mname].decode("utf-8"))
    except (UnicodeDecodeError, ManifestError) as exc:
        return f"{mname} cannot be read: {exc}"
    if _entries(verdict) != entries:
        return (f"{mname} is not the inputs {vname} read: "
                + _difference(entries, _entries(verdict)))
    if set(meta) != _META_KEYS:
        return f"{mname}'s meta does not hold the keys a pre-check pack's meta holds"
    if meta["tool"] != ToolInfo.model_fields["name"].default:
        return f"{mname} names the tool {meta['tool']!r}, not courant"
    if (meta["version"], meta["schema"], meta["source"]) != \
            (verdict.tool.version, verdict.schema_, verdict.run.source):
        return f"{mname}'s meta disagrees with {vname} on the version, schema or source"
    if not _file_name(meta["target"]) or not (meta["case_json"] is None
                                              or _file_name(meta["case_json"])):
        return f"{mname} names its run or its sidecar by something other than a file name"
    if meta["error_rates"] != doc.error_rates.model_dump(mode="json", by_alias=True):
        return f"{mname}'s error rates are not the ones {SUMMARY_JSON} states"
    if rendered:
        made = pack_files(verdict, meta["target"], meta["case_json"])
        for f in PACK_FILES:
            if made[f] != data[f"{pack}/{f}"]:
                return f"{pack}/{f} is not what courant {doc.courant_version} writes for {vname}"
    return None


def _row_problem(i: int, row: PrecheckRow, n: int, doc: PrecheckSummary,
                 data: dict[str, bytes], sha: dict[str, str], rendered: bool) -> str | None:
    from courant.verify.verdict2 import SCHEMA_V2, load_verdict

    label = label_of(i, n)
    where = f"{SUMMARY_JSON} row {i} ({label})"
    if doc.redacted and row.place != label:
        return f"{where} is redacted and its place is not its label"
    if row.unreadable is not None:
        if row.pack is not None or row.verdict != "insufficient" or row.basis is not None \
                or row.rule is not None or row.reasons or row.log is not None:
            return f"{where} could not be read and still carries a verdict, a pack or a log"
        return None
    if row.pack != f"packs/{label}":
        return f"{where} names its pack {row.pack!r}, not packs/{label}"
    for f in PACK_FILES:
        if f"{row.pack}/{f}" not in data:
            return f"{row.pack}/{f} is missing"
    vname = f"{row.pack}/verdict.json"
    try:
        raw = json.loads(data[vname])
    except (ValueError, RecursionError) as exc:
        return f"{vname} is not a courant.verdict/2 verdict ({type(exc).__name__})"
    # The bytes' own schema, before `load_verdict`, which reads a courant.verdict/1 as /2.
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_V2:
        return f"{vname} is not a courant.verdict/2 verdict"
    try:
        verdict = load_verdict(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        return f"{vname} is not a courant.verdict/2 verdict ({type(exc).__name__})"
    # `VerdictV2` ignores a key it does not know and coerces "123" to 123; the bytes are held to
    # the schema as written, whatever the archive's version: every key one the model has, every
    # value the one it holds, of the type it holds.
    off = _not_as_written(raw, verdict.model_dump(mode="json", by_alias=True))
    if off is not None:
        return (f"{vname} is not a courant.verdict/2 verdict: {off} is not a key of its schema or "
                "not of the type its schema gives")
    if verdict.tool.version != doc.courant_version:
        return (f"{vname} was made by courant {verdict.tool.version}, and {SUMMARY_JSON} says "
                f"courant {doc.courant_version}")
    problem = _pack_problem(row.pack, verdict, doc, data, rendered)
    if problem:
        return problem
    mine =(row.verdict, row.basis, row.rule, [(r.code, r.severity, r.summary) for r in row.reasons])
    theirs = (verdict.verdict, verdict.basis, verdict.rule,
              [(r.code, r.severity, r.summary) for r in verdict.reasons])
    if mine != theirs:
        return f"{where} disagrees with {vname}"
    if doc.include_logs and row.log is not None:
        m = _LOG_MEMBER.fullmatch(row.log)
        if not m or m.group(1) != label or row.log not in sha:
            return f"{where} names its log {row.log!r}, which is not in the archive"
        listed = {Path(f.path).name: f.sha256 for f in verdict.run.inputs}
        if listed.get(m.group(2)) != sha[row.log]:
            return f"{row.log} is not the log {vname} read"
    elif row.log is not None:
        return f"{where} names a log, and the summary says no log was included"
    return None


def check_archive(path: str | Path) -> tuple[bool, str]:
    """(intact, message) for a pre-check archive, read in memory and never extracted. Raises
    `ArchiveError` when the file is not a zip archive that can be read."""
    path = Path(path)
    data, sha, problem = _read_members(path)
    if problem:
        return False, problem
    names = sorted(sha)                               # the layout was checked on reading them
    problem = _manifest_problem(data, sha)
    if problem:
        return False, problem
    from courant import __version__

    try:
        raw = json.loads(data[SUMMARY_JSON])
    except (ValueError, RecursionError) as exc:
        return False, f"{SUMMARY_JSON} is not a {SCHEMA} summary ({type(exc).__name__})"
    try:
        doc = PrecheckSummary.model_validate(raw)     # error_rates included: about's shape
    except (ValueError, TypeError, RecursionError) as exc:
        where = ("its error_rates are not courant.error-rates/1 as courant states them"
                 if "error_rates" in _bad_fields(exc) else type(exc).__name__)
        return False, f"{SUMMARY_JSON} is not a {SCHEMA} summary ({where})"
    # `PrecheckSummary` coerces 0 to False and "0" or 1.0 to an integer, and another version's
    # summary is not re-rendered: its bytes are held to the schema as written, as each verdict's
    # are (T-PRECHECK-2 C4).
    # A summary courant writes states every key of its schema (`to_json` drops none), so one
    # left out is not taken as its default: `schema` itself included (round 3).
    off = _not_as_written(raw, doc.model_dump(mode="json", by_alias=True), complete=True)
    if off is not None:
        return False, (f"{SUMMARY_JSON} is not a {SCHEMA} summary: {off} is missing, not a key "
                       "of its schema, or not of the type its schema gives")
    if not doc.runs:
        return False, f"{SUMMARY_JSON} lists no run, and an archive judges at least one run"
    if doc.error_rates.courant_version != doc.courant_version:
        return False, (f"{SUMMARY_JSON} states error rates for courant "
                       f"{doc.error_rates.courant_version} and says it is courant "
                       f"{doc.courant_version}")
    # The renderings (canonical JSON, report.md, inputs.manifest, summary.md) are this build's
    # code; an archive another version made is held to them only when the versions match.
    rendered = doc.courant_version == __version__
    if rendered and doc.to_json().encode() != data[SUMMARY_JSON]:
        return False, f"{SUMMARY_JSON} is not in the form courant writes it"
    if rendered:                       # this build's rates, whether or not a pack restates them
        from courant.verify import about

        if doc.error_rates.model_dump(mode="json", by_alias=True) != about.error_rates():
            return False, (f"{SUMMARY_JSON}'s error rates are not the ones courant {__version__} "
                           "states")
    n = len(doc.runs)
    for i, row in enumerate(doc.runs, 1):
        problem = _row_problem(i, row, n, doc, data, sha, rendered)
        if problem:
            return False, problem
    packs = {n_.rsplit("/", 1)[0] for n_ in names if n_.startswith("packs/")}
    named = {r.pack for r in doc.runs if r.pack is not None}
    for extra in sorted(packs - named):
        return False, f"{extra} is in the archive and no row of {SUMMARY_JSON} names it"
    logs = {n_ for n_ in names if n_.startswith("logs/")}
    for extra in sorted(logs - {r.log for r in doc.runs if r.log is not None}):
        return False, f"{extra} is in the archive and no row of {SUMMARY_JSON} names it"
    if rendered and render_summary_md(raw).encode() != data[SUMMARY_MD]:
        return False, f"{SUMMARY_MD} is not the rendering of {SUMMARY_JSON}"
    readable = len(named)
    message = (f"intact: {len(names)} members match {MANIFEST}; {n} rows, {readable} verdicts "
               f"valid and in agreement with {SUMMARY_JSON}")
    if not rendered:
        message += (f"; the rendering check was not applied: the archive was made by courant "
                    f"{doc.courant_version} and this is courant {__version__}, so report.md, "
                    f"summary.md and the canonical bytes were not re-rendered")
    return True, message


def _not_as_written(raw, made, at: str = "the document", complete: bool = False) -> str | None:
    """Where a JSON document `raw` is not what its model gave back (`made`, its JSON dump): a key
    the model does not have, or a value the model coerced (a string for a number, a number for a
    boolean, 1.0 for an integer). A key `raw` leaves out is the model's default, and allowed,
    unless `complete`: then every key of `made` must be in `raw`, at every depth.
    None when every key and value in `raw` stands as the model holds it."""
    if isinstance(made, dict):
        if not isinstance(raw, dict):
            return repr(at)
        if complete:
            for key in sorted(made):
                if key not in raw:
                    return repr(key if at == "the document" else f"{at}.{key}")
        for key, value in raw.items():
            where = key if at == "the document" else f"{at}.{key}"
            if key not in made:
                return repr(where)
            off = _not_as_written(value, made[key], where, complete)
            if off is not None:
                return off
        return None
    if isinstance(made, list):
        if not isinstance(raw, list) or len(raw) != len(made):
            return repr(at)
        for i, (r, m) in enumerate(zip(raw, made, strict=True)):
            off = _not_as_written(r, m, f"{at}[{i}]", complete)
            if off is not None:
                return off
        return None
    if isinstance(made, float) and not isinstance(raw, bool) and isinstance(raw, int | float):
        return None if raw == made else repr(at)          # JSON writes 1 and 1.0 for one number
    return None if type(raw) is type(made) and raw == made else repr(at)


def _bad_fields(exc: BaseException) -> set[str]:
    """The top-level fields a pydantic validation error names, or none."""
    errors = getattr(exc, "errors", None)
    try:
        return {str(e["loc"][0]) for e in errors() if e.get("loc")} if callable(errors) else set()
    except (TypeError, KeyError, IndexError):
        return set()
