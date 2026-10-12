"""The error rate travels with the verdict (TRD GATE-12).

`--about`, the HTML footer, the pack manifest and the Action's summary name the Courant version
and the rates published for it: false accepts and false alarms (RRD RM-01's definitions), the
abstentions apart, each with its denominator and 95% interval, and RM-21's non-accept rate where
it exists. Every figure is read from the packaged results (`courant/data/certification.json`,
a byte copy of the measured result RM-01; INV-7, CERT-3). None is typed here, and none
is written into the verdict, so a verdict's bytes do not change when a results file does.

A version with no measured rates says so: the packaged results name the verifier version they
were measured on, and a different version is not given that version's rates.
"""

from __future__ import annotations

SCHEMA = "courant.error-rates/1"

#: RM-21 (the non-accept rate on healthy runs, both forks) has no packaged result yet.
_NO_RM21 = ("not measured; RRD RM-21 is pre-registered and has no result yet, so the share of "
            "healthy runs this gate does not accept is unknown")


def _rate(card: dict, key: str) -> dict | None:
    r = card.get(key)
    if not r:
        return None
    out = {k: r[k] for k in ("n", "of", "lo", "hi") if k in r}
    return out if {"n", "of"} <= set(out) else None


def error_rates() -> dict:
    """The rates published for this build, as data (`courant.error-rates/1`)."""
    from courant import __version__
    from courant.data import certification

    data = certification()
    doc: dict = {"schema": SCHEMA, "courant_version": __version__, "measured": False,
                 "non_accept_rate_healthy": None, "non_accept_rate_note": _NO_RM21}
    if data is None:
        doc["note"] = "This build carries no certification results."
        return doc
    measured_on = data.get("verifier_version")
    doc["measured_on"] = {"verifier_version": measured_on, "tag": data.get("tag"),
                          "date": data.get("date"), "corpus": data.get("corpus"),
                          "policy": data.get("policy"),
                          # what this build read: the packaged copy of RM-01's result
                          "source": "courant/data/certification.json"}
    if measured_on != __version__:
        doc["note"] = (f"No error rates are measured for courant {__version__}; the published "
                       f"results are for courant {measured_on}.")
        return doc
    card = (data.get("scorecard") or {}).get("all") or {}
    far, frr, abstain = _rate(card, "far"), _rate(card, "frr"), _rate(card, "abstain")
    if far is None or frr is None:
        doc["note"] = "The packaged results do not state this build's error rates."
        return doc
    doc.update(measured=True, false_accepts=far, false_alarms=frr, abstentions=abstain)
    return doc


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ratio(r: dict) -> str:
    s = f"{r['n']} of {r['of']}"
    if "lo" in r and "hi" in r:
        s += f" (95% interval {_pct(r['lo'])}–{_pct(r['hi'])})"
    return s


def sentences(doc: dict | None = None) -> list[str]:
    """The rates as plain sentences, one fact each, for `--about` and the Markdown summary."""
    doc = doc or error_rates()
    ver = doc["courant_version"]
    if not doc["measured"]:
        return [f"courant {ver}: " + doc.get("note", "no error rates are published for this version.")]
    m = doc["measured_on"]
    out = [f"courant {ver}, measured on {m['corpus']} ({m['date']}, policy {m['policy']}).",
           f"Wrong results accepted: false accepts {_ratio(doc['false_accepts'])}.",
           f"Good results rejected: false alarms {_ratio(doc['false_alarms'])}."]
    if doc.get("abstentions"):
        out.append(f"Abstentions, reported apart: insufficient on {_ratio(doc['abstentions'])}.")
    out.append(f"Non-accept rate on healthy runs (RM-21): {doc['non_accept_rate_note']}.")
    return out


def render_text(doc: dict | None = None) -> str:
    return "\n".join(sentences(doc)) + "\n"


def render_markdown(doc: dict | None = None) -> str:
    """For a CI job summary (GATE-7): a heading and one bullet per fact."""
    doc = doc or error_rates()
    return "### Error rate\n\n" + "\n".join(f"- {s}" for s in sentences(doc)) + "\n"


def footer(doc: dict | None = None) -> str:
    """The rates as plain text for the HTML report's footer."""
    return "Error rate. " + " ".join(sentences(doc))
