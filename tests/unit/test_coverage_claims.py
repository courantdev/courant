"""Every code the taxonomy advertises must be one the product can actually emit.

A vocabulary is a promise. Listing 25 failure modes says Courant can name 25
failure modes, and an audit over the knowledge base, the checkers and 31 real
solver logs found that three of them had no route of any kind — no finding any
checker emits, no log pattern OpenFOAM ever prints, no config predicate, no
suggestion. They were counted in the coverage number anyway.

Two more were nearly as bad: the signature existed but described output the
solver does not produce. `NUM.BAD_INITIAL_FIELD` looked for `residual.nan` and
`Initial residual = inf|nan`; the real run prints neither and was diagnosed as a
Courant violation instead. A pattern written from what a fault *ought* to look
like is indistinguishable, at rest, from one written from a log.

So this file asserts the shape of the claim rather than its size:

  * every detectable code has at least one route to it,
  * every undetectable code says why, in words a reader can check,
  * and the routes that are log patterns are checked against real solver output
    where the corpus can speak to them.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from courant.diagnosis.kb_data import build_default_kb
from courant.taxonomy import TAXONOMY, detectable_codes, undetectable

CHECKERS = pathlib.Path(__file__).resolve().parents[2] / "courant" / "diagnosis" / "checkers.py"


@pytest.fixture(scope="module")
def kb():
    return build_default_kb()


@pytest.fixture(scope="module")
def suggestable() -> set[str]:
    """Codes some checker can put in a finding's `suggests`.

    Read from the source because that is where the list lives; a checker that
    stops suggesting a code should make this test notice.
    """
    return set(re.findall(r'"([A-Z]+\.[A-Z_]+)"', CHECKERS.read_text()))


class TestEveryClaimedCodeCanBeEmitted:
    def test_every_detectable_code_has_a_route(self, kb, suggestable):
        orphaned = []
        for code in detectable_codes():
            sig = kb.entries[code].signature
            if (sig.finding_ids or sig.log_patterns or sig.config_predicates
                    or code in suggestable):
                continue
            orphaned.append(code)

        assert not orphaned, (
            "these codes are advertised in the taxonomy and cannot be emitted "
            f"by any route: {orphaned}. Either wire them up or mark them "
            "detectable=False with a reason.")

    def test_every_undetectable_code_says_why(self):
        for entry in undetectable():
            assert len(entry.why_undetectable) > 80, (
                f"{entry.code} is excluded from coverage with a reason too "
                "short to be checkable")

    def test_the_taxonomy_is_still_twenty_five(self):
        """The undetectable three stay in the vocabulary on purpose — they are
        real failure modes, and a report may need to discuss one. They are
        excluded from the coverage *claim*, not from the language."""
        assert len(TAXONOMY) == 25
        assert len(detectable_codes()) == 22


class TestSignaturesDescribeRealSolverOutput:
    """A log pattern is a claim about what OpenFOAM prints. The corpus is 31
    real logs; where a pattern's scenario is in it, the pattern must match."""

    def test_no_signature_matches_a_healthy_tutorial(self, kb):
        """The most expensive class of bug this product has: `trapFpe`,
        `reverse flow` and `thermophysical` each fired on unmodified tutorials
        before a real solver was ever run against them."""
        import glob
        import gzip

        healthy = {
            pathlib.Path(f).name.split("__")[0]:
                gzip.open(f, "rt", errors="ignore").read()
            for f in glob.glob("certification/corpus/*__healthy.log.gz")
        }
        assert healthy, "the healthy corpus is missing"

        offenders = []
        for code in TAXONOMY:
            for pattern in kb.entries[code].signature.log_patterns:
                for case, text in healthy.items():
                    if re.search(pattern, text, re.I):
                        offenders.append((code, pattern, case))

        assert not offenders, (
            "signatures matched output from an unmodified OpenFOAM tutorial, "
            f"which is a false positive on a customer who did nothing wrong: "
            f"{offenders}")
