"""JUnit XML for batch verification, so any CI system shows each run as a test (TRD REP-7).

One `testcase` per run, in the order `verify_all` returns them, which is sorted paths:
- `reject` (NOT_TRUSTWORTHY in version 1) is a `failure`, carrying its first reason;
- `insufficient` (NOT_ASSESSABLE) is `skipped`;
- an input error is an `error`.
Each verdict is written in its own schema's words, so `--schema 1` keeps version 1's bytes.

There are no timestamps and no durations, so the bytes depend only on the verdicts.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


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
            why = verdict.not_assessed[0] if verdict.not_assessed else verdict.rule
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
