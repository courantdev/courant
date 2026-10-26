"""`courant.support/1`: what to send when something is wrong (TRD OPS-6, SEC-6).

Versions, the platform, the command line and, when one is given, the verdict JSON. No case data:
no log, no dictionary, no field. The verdict still carries the customer's own words: the run id,
file names, patch names, the names of specification items, and the values its reasons quote. With
`pseudonymise=True` (`courant doctor --bundle F --pseudonymise`) those become stable pseudonyms
(`run1`, `patch1`, `file1`, `item1`), everything free-form goes through the trust boundary's
`redact_text` (`courant/diagnosis/redact.py`: paths, hosts, credentials), and the command line
keeps only the command words and flags. The command writes the bundle to a file the user names
(this package reads and never writes, ARC lint) and Courant never sends it anywhere.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path

SCHEMA = "courant.support/1"

#: Case-relative names that are OpenFOAM's own vocabulary, not the customer's. Anything else in a
#: file path is a customer word and is pseudonymised. A name is looked up without a `.gz`
#: suffix: OpenFOAM reads `phi.gz` where `phi` is absent, and the name is its own either way.
_KNOWN_PATH_WORDS = frozenset({
    "0", "0.orig", "constant", "system", "postProcessing", "polyMesh", "boundary", "points",
    "faces", "owner", "neighbour", "controlDict", "fvSchemes", "fvSolution", "blockMeshDict",
    "snappyHexMeshDict", "decomposeParDict", "setFieldsDict", "transportProperties",
    "turbulenceProperties", "thermophysicalProperties", "momentumTransport", "g", "U", "p",
    "p_rgh", "k", "epsilon", "omega", "nut", "nuTilda", "T", "alpha.water", "courantMonitors",
    "force.dat", "moment.dat", "surfaceFieldValue.dat", "forceCoeffs.dat", "probes",
    "phi", "dynamicMeshDict", "fvModels", "fvOptions",   # read for the boundary flux (VER-18)
})
_KNOWN_PATH = re.compile(r"(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|log\.\w+|log\.\w+\.gz)", re.IGNORECASE)
_COMMANDS = frozenset({"courant", "verify", "doctor", "convergence", "preflight", "audit", "watch",
                       "spec", "init", "hash", "check", "demo", "mcp", "test", "run", "version"})


def build(argv: list[str], verdict_path: str | Path | None = None, *,
          pseudonymise: bool = False) -> dict:
    """The bundle as data. A verdict file must be a Courant verdict of either major; anything
    else is refused, so a bundle can never carry an arbitrary file."""
    from courant import __version__
    from courant.verify.verdict2 import load_verdict

    verdict = None
    if verdict_path is not None:
        try:
            verdict = json.loads(load_verdict(Path(verdict_path).read_text()).to_json())
        except (OSError, ValueError) as exc:
            raise ValueError(f"{verdict_path} is not a Courant verdict: {exc}") from exc
    doc = {
        "schema": SCHEMA,
        "courant": {"version": __version__, "packaged": bool(getattr(sys, "frozen", False))},
        "platform": {"system": platform.system().lower(), "release": platform.release(),
                     "machine": platform.machine(), "python": platform.python_version()},
        "command_line": list(argv),
        "verdict": verdict,
        "pseudonymised": pseudonymise,
    }
    return _pseudonymise(doc) if pseudonymise else doc


# ---------------------------------------------------------------- SEC-6
def _names(verdict: dict) -> dict[str, str]:
    """Every customer word the verdict carries, with its pseudonym, longest first."""
    mapping: dict[str, str] = {}
    counters = {"patch": 0, "file": 0, "item": 0, "phase": 0}

    def add(word: str | None, kind: str) -> None:
        if word and len(word) > 1 and word not in mapping:
            counters[kind] += 1
            mapping[word] = f"{kind}{counters[kind]}"

    run = verdict.get("run") or {}
    if run.get("id"):
        mapping[run["id"]] = "run1"
    paths = [f.get("path", "") for f in run.get("inputs") or []]
    conf = verdict.get("conformance") or {}
    paths += [i.get("file", "") for i in conf.get("items") or []]
    for path in paths:
        for seg in path.replace("\\", "/").split("/"):
            if seg and seg.removesuffix(".gz") not in _KNOWN_PATH_WORDS \
                    and not _KNOWN_PATH.fullmatch(seg):
                add(seg, "file")
    for item in conf.get("items") or []:
        m = re.match(r"boundaryField/([^/]+)/", item.get("location") or "")
        if m:
            add(m.group(1), "patch")
    for item in conf.get("items") or []:
        name = item.get("name")
        add(name, "item")
        if name:                                         # the reason code is conformance.<slug>
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            if slug and slug not in mapping:
                mapping[slug] = mapping.get(name, "item")
    for reason in verdict.get("reasons") or []:           # patch names quoted by other reasons
        for m in re.finditer(r"boundaryField/([^/\s)]+)/", reason.get("input") or ""):
            add(m.group(1), "patch")
    for said in _said(verdict):                          # PRE-7 names a phase by its name
        for m in _PHASES_SAID.finditer(said):
            for phase in m.group(1).split(", "):
                add(phase.strip(), "phase")
    if conf.get("task"):
        mapping[conf["task"]] = "<task>"
    return dict(sorted(mapping.items(), key=lambda kv: -len(kv[0])))


_PATH_START = re.compile(r"""^[("'\[]*(?:/|~|[A-Za-z]:[\\/])""")
#: PRE-7's sentence names each phase whose `nu` the default checks did not read
#: (`preflight.setup_not_checked`): the case author's words, as a patch name is.
_PHASES_SAID = re.compile(r"the nu of phases (.+?) in constant/transportProperties")
#: The lists of a verdict or preflight whose sentences may quote what the case wrote.
_SAID = ("not_assessed", "not_checked")


def _said(verdict: dict) -> list[str]:
    return [s for key in _SAID for s in verdict.get(key) or [] if isinstance(s, str)]


def _unquoted(verdict: dict) -> dict:
    """The verdict with the sentence that quotes the values the case reader did not read (TRD
    VER-3b, `checkers.unread_values_sentence`: a macro's name, an include's path, an
    expression) said without them: pattern scrubbing cannot recognise the customer's text, so
    a pseudonymised bundle carries that such values were not read, and not what they were."""
    from courant.diagnosis.checkers import UNREAD_VALUES_TAIL, UNREAD_VALUES_TITLE

    head = UNREAD_VALUES_TITLE + ": "
    out = dict(verdict)
    for key in _SAID:
        if isinstance(out.get(key), list):
            out[key] = [UNREAD_VALUES_TITLE + " (what the case wrote is left out of a "
                        "pseudonymised bundle)" + UNREAD_VALUES_TAIL
                        if isinstance(s, str) and s.startswith(head) else s for s in out[key]]
    return out


def _redact(text: str) -> str:
    """The trust boundary's `redact_text` over free text, without its habit of collapsing a
    case-relative path (`boundaryField/patch1/value`, which is not a disclosure) to its last
    segment: a token that starts like an absolute path goes through whole, any other token goes
    through one `/`-separated piece at a time."""
    from courant.diagnosis.redact import redact_text

    out = []
    for token in re.split(r"(\s+)", text):
        if _PATH_START.match(token):
            out.append(redact_text(token))
        else:
            out.append("/".join(redact_text(seg) for seg in token.split("/")))
    return "".join(out)


def _pseudonymise(doc: dict) -> dict:
    if doc.get("verdict"):
        doc = {**doc, "verdict": _unquoted(doc["verdict"])}
    mapping = _names(doc.get("verdict") or {})
    pattern = (re.compile("|".join(rf"(?<!\w){re.escape(w)}(?!\w)" for w in mapping))
               if mapping else None)

    def clean(value):
        if isinstance(value, str):
            if pattern is not None:
                value = pattern.sub(lambda m: mapping[m.group(0)], value)
            return _redact(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    out = dict(doc)
    out["verdict"] = clean(doc["verdict"]) if doc.get("verdict") else None
    out["command_line"] = [a if (a.startswith("-") or a in _COMMANDS) else "<arg>"
                           for a in doc["command_line"]]
    return out
