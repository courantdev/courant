"""JUnit XML for batch verification, so any CI system shows each run as a test (TRD REP-7).

One `testcase` per run, in the order `verify_all` returns them, which is sorted paths:
- `reject` (NOT_TRUSTWORTHY in version 1) is a `failure`, carrying its first reason;
- `insufficient` (NOT_ASSESSABLE) is `skipped`, carrying its `insufficient.*` reason;
- an input error is an `error`.
Each verdict is written in its own schema's words, so `--schema 1` keeps version 1's bytes.

There are no timestamps and no durations, so the bytes depend only on the verdicts.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape as _escape
from xml.sax.saxutils import quoteattr as _quoteattr

# XML 1.0 forbids these even escaped; one in a run's name made the whole file unparseable
# (security review 0.7.0, SEC070-7). Each is written as \\xNN instead.
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff\ud800-\udfff]")


def _xml_text(text: str) -> str:
    return _NOT_XML.sub(lambda m: f"\\x{ord(m.group()):02x}", str(text))


def escape(text: str) -> str:
    return _escape(_xml_text(text))


def quoteattr(text: str) -> str:
    return _quoteattr(_xml_text(text))


def _insufficient_why(verdict) -> str:
    """Why a run is insufficient, for its `skipped` message (VPA-CLI-2): the verdict's own
    `insufficient.*` reason, code and sentence. A version 1 verdict has none, so its first
    not-assessed sentence stands in, as before; failing both, the rule."""
    for r in verdict.reasons:
        if r.kind == "insufficient" or r.code.startswith("insufficient."):
            return f"{r.code}: {r.summary}"
    return verdict.not_assessed[0] if verdict.not_assessed else verdict.rule


def render_junit(results, root: str | Path) -> str:
    root = Path(root)
    cases, fails, errors, skipped = [], 0, 0, 0
    for target, verdict, err in results:
        name = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
        head = f'    <testcase classname="courant.verify" name={quoteattr(name)}'
        if err:
            errors = errors + 1
            cases.append(f'{head}>\n      <error type="InputError" message={quoteattr(err)}/>\n    </testcase>')
        elif verdict.verdict in ("NOT_TRUSTWORTHY", "reject"):
            fails = fails + 1
            first = verdict.reasons[0] if verdict.reasons else None
            msg = f"{first.code}: {first.summary}" if first else verdict.rule
            body = "\n".join(f"{r.code} [{r.severity}] {r.summary} (read {r.input}; {r.method})"
                             for r in verdict.reasons)
            cases.append(f'{head}>\n      <failure type="{verdict.verdict}" message={quoteattr(msg)}>'
                         f"{escape(f'rule {verdict.rule}, grade {verdict.grade or chr(8212)}' + chr(10) + body)}"
                         f"</failure>\n    </testcase>")
        elif verdict.verdict in ("NOT_ASSESSABLE", "insufficient"):
            skipped = skipped + 1
            why = _insufficient_why(verdict)
            cases.append(f'{head}>\n      <skipped message={quoteattr(verdict.verdict + ": " + why)}/>\n    </testcase>')
        else:
            cases.append(f"{head}/>")
    n = len(cases)
    attrs = f'tests="{n}" failures="{fails}" errors="{errors}" skipped="{skipped}"'
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            f'<testsuites name="courant verify" {attrs}>\n'
            f'  <testsuite name="courant verify" {attrs}>\n'
            + "\n".join(cases) + ("\n" if cases else "")
            + "  </testsuite>\n</testsuites>\n")
