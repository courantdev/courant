"""Which OpenFOAM discretisation schemes are bounded, in one place.

This module exists because the question "is this scheme bounded?" was answered
twice in the codebase, and the two answers disagreed.

The original rule was `"bounded" not in scheme and "upwind" not in scheme`, which
is wrong on the most common turbulence scheme OpenFOAM ships:
`div(phi,k) Gauss limitedLinear 1` contains neither word and `limitedLinear` is
*definitionally* a limited scheme. Both of OpenFOAM's turbulent tutorials use it.
The certification run of 30 Jul 2026 found this and it was fixed — in
`courant/diagnosis/checkers.py`. The copy in the knowledge base's
`unbounded_div` config predicate was missed, so the KB kept scoring
`NUM.UNBOUNDED_SCHEME` +2.0 on every RAS case for a scheme that is bounded, and
the next full matrix had that code in the wrong answer of eight of twelve
failing scenarios.

A fact about OpenFOAM with two implementations is a fact with two truth values.
There is now one.
"""

from __future__ import annotations

import re

#: OpenFOAM's limited (bounded) surface-interpolation schemes, read from
#: `src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes` in v2406
#: rather than recalled. Lower-cased for matching.
BOUNDED_SCHEMES = frozenset({
    "limitedlinear", "limitedcubic", "limitedlinear01", "limitedcubic01",
    "vanleer", "vanalbada", "minmod", "muscl", "superbee", "ospre", "umist",
    "quick", "sfcd", "gamma", "limitwith", "limited", "limited01",
    "filteredlinear", "filteredlinear2", "filteredlinear3",
    "upwind", "linearupwind",
    # Courant- and region-blended schemes fall back to upwind where it matters,
    # which is the whole point of them.
    "coblended", "cellcoblended", "localblended", "limiterblended",
    "fixedblended", "blended", "outletstabilised", "lust",
})

#: The explicit keyword, still honoured — `bounded Gauss limitedLinear 1` is how
#: OpenFOAM's own steady tutorials write it.
BOUNDED_KEYWORD = "bounded"

#: `div(phi,<field>)` is convection of a transported quantity. Matching any
#: field rather than a hardcoded three: pisoFoam/RAS/cavity also transports
#: `omega`, `R` and `nuTilda`, and an unbounded scheme on those is just as real.
_RE_CONVECTION = re.compile(r"^div\(\s*phi\s*,", re.I)

_RE_WORDS = re.compile(r"[^a-z0-9]+")


def is_convection_term(term: str) -> bool:
    """Whether this `divSchemes` key discretises transport of a field."""
    return bool(_RE_CONVECTION.match(term.strip()))


def is_bounded_scheme(scheme: str) -> bool:
    """Whether this discretisation is bounded/limited.

    Word-wise rather than substring: `linear` is a substring of `limitedLinear`,
    so a naive `in` test would call the genuinely unbounded `Gauss linear`
    bounded and miss the real thing.
    """
    lowered = scheme.lower()
    if BOUNDED_KEYWORD in lowered:
        return True
    return any(word in BOUNDED_SCHEMES for word in _RE_WORDS.split(lowered) if word)


def resolve_macros(div_schemes: dict[str, str], depth: int = 8) -> dict[str, str]:
    """Expand `$name` references, the way OpenFOAM reads the dictionary.

    `simpleFoam/T3A` writes `turbulence bounded Gauss linearUpwind grad;` once, and every
    turbulence term as `div(phi,k) $turbulence;`. Unexpanded, `$turbulence` read as a scheme
    with no bounding in it, and an unmodified tutorial was told its schemes were unbounded.
    That was the first fresh case run, and nothing in the corpus wrote a macro. Only
    same-dictionary references are resolved. An unresolved one is left as written, and
    `is_bounded_scheme` then treats it as not known to be bounded, as before.
    """
    out = dict(div_schemes)
    for _ in range(depth):
        changed = False
        for term, scheme in out.items():
            s = str(scheme).strip()
            if s.startswith("$"):
                ref = s[1:].strip().rstrip(";").strip()
                if ref in out and str(out[ref]).strip() != s:
                    out[term] = out[ref]
                    changed = True
        if not changed:
            break
    return out


def unbounded_convection_schemes(div_schemes: dict[str, str]) -> dict[str, str]:
    """The convection terms in this case discretised without bounding (macros expanded)."""
    resolved = resolve_macros(div_schemes)
    return {term: scheme for term, scheme in resolved.items()
            if is_convection_term(term) and not is_bounded_scheme(scheme)}
