"""LLM diagnosis — Layer 2 of the Diagnosis Engine (Tech Arch §5.2).

The LLM receives the Evidence Bundle and produces a ranked list of root-cause
hypotheses, each citing real deterministic Findings. It is a reasoning subroutine
inside a deterministic control plane — it never decides what is *true* or *safe*:

  * output is constrained to a JSON schema whose root_cause_code is an ENUM of the
    25 taxonomy codes (the model cannot emit an invalid code),
  * every hypothesis is re-validated against the taxonomy after the call,
  * evidence_refs are filtered to Finding ids that actually exist in the bundle
    (hallucinated citations are dropped), and
  * any API/parse/validation failure falls back to the deterministic diagnoser —
    the blast radius of a bad model output is a discarded response, logged.

Which *model* answers is the model door's decision (``courant/llm/``, ARC-5) and
deliberately not this module's business. An Evidence Bundle is customer-derived
data, so the door sends it only to an allowlisted free model that is
zero-data-retention and measured on diagnosis (ARC-9, ARC-12); with none, the
quota spent, or an answer that fails the schema, this falls back to the
deterministic engine (ARC-11). Every guarantee above holds identically whichever
model replies, because none of them is trusted with any of it.

The one other way to run it is as a named, pre-registered measurement over public
data (``data="public", measurement="RM-nn"``, ARC-12, ARC-19): there the
deterministic answer is never returned, since it would be scored as the model's.
A stop (``courant.llm.stops_batch``) is raised so the batch stops, and a model's
failure on one log is an empty diagnosis with ``last_error`` saying why. A model
that answers an empty list of hypotheses has answered (nothing is wrong): that is
an empty diagnosis with no ``last_error`` (ARC-20). ``batch_identity`` is what such
a batch is: the measurement, the models the door would ask, and the prompt's
version; each item is keyed by ``diagnosis_request(bundle)``, the request it sends.

Single-shot over the bundle (which already carries det_findings + kb_matches);
the read_file/query_kb/run_probe tool loop (§5.3) is a live-mode enhancement
layered on later.
"""

from __future__ import annotations

import logging

from courant.diagnosis.bundle import EvidenceBundle
from courant.diagnosis.schema import DiagnosisResponse, Hypothesis
from courant.llm import LLMError, default_door, request_of, stops_batch
from courant.llm.policy import DATA
from courant.taxonomy import TAXONOMY, all_codes

log = logging.getLogger("courant.diagnosis.llm")

#: A diagnosis, as the model door is asked it: customer-derived data (ARC-9) wanting a
#: structured answer. One definition, for the diagnoser and for every "can it run" question
#: (`default_diagnoser`, `Settings.capabilities`, the CLI and `courant doctor`).
DIAGNOSIS_CALL = {"task": "diagnosis", "data": "customer", "capability": "structured"}

#: The diagnosis prompt's version. Change it with the prompt: a measured batch is keyed by it
#: (``LLMDiagnoser.batch_identity``) and each item by its request (ARC-20).
PROMPT_VERSION = "diagnosis/1"


def build_system_prompt() -> str:
    lines = [
        "You are Courant, an expert CFD reliability engineer. You diagnose why an "
        "OpenFOAM simulation run failed or produced questionable results.",
        "",
        "How to reason:",
        "- Deterministic checkers have already established hard facts — the "
        "`det_findings` in the bundle. Treat them as ground truth; never contradict "
        "or invent findings.",
        "- `kb_matches` are candidate root causes surfaced by signature matching. "
        "They are hints, not conclusions — you rank the true root cause.",
        "- Separate SYMPTOM from ROOT CAUSE. A NaN residual, a diverging pressure, or "
        "a high Courant number is usually a symptom. Ask what upstream setup or mesh "
        "problem produced it (e.g. reversed outlet flow, over-relaxation, a laminar "
        "model at a turbulent Reynolds number, wrong viscosity units).",
        "- Prefer the most specific correct cause. If two are plausible, list both, "
        "most likely first, with calibrated confidence.",
        "",
        "Rules:",
        "- Use ONLY these taxonomy codes for root_cause_code:",
    ]
    for code in all_codes():
        e = TAXONOMY[code]
        lines.append(f"    {code} — {e.title}: {e.summary}")
    lines += [
        "- `evidence_refs` MUST be ids drawn from `det_findings` in the bundle. Cite "
        "at least one for your top hypothesis. Do not cite ids that are not present.",
        "- confidence is 0..1 and calibrated (not all 0.9).",
        "- Return 1-3 hypotheses, most likely first. Explanations are terse and "
        "engineering-precise.",
    ]
    return "\n".join(lines)


def build_output_schema() -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "root_cause_code": {"type": "string", "enum": all_codes()},
                            "confidence": {"type": "number"},
                            "evidence_refs": {"type": "array",
                                              "items": {"type": "string"}},
                            "explanation": {"type": "string"},
                        },
                        "required": ["root_cause_code", "confidence",
                                     "evidence_refs", "explanation"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["hypotheses"],
            "additionalProperties": False,
        },
    }


def build_user_prompt(bundle: EvidenceBundle) -> str:
    finding_ids = [f.id for f in bundle.det_findings]
    return (
        "Diagnose this run. Valid finding ids you may cite: "
        f"{finding_ids}\n\nEVIDENCE BUNDLE:\n{bundle.to_llm_json()}"
    )


def diagnosis_request(bundle: EvidenceBundle) -> dict:
    """What a diagnosis of ``bundle`` asks the model (``courant.llm.request_of``): the one
    definition the diagnoser sends and a measured batch keys each item by (ARC-20), so a
    change to the parser, a checker, the knowledge base or the bundle is a different item."""
    return request_of(system=build_system_prompt(), user=build_user_prompt(bundle),
                      schema=build_output_schema()["schema"])


class LLMDiagnoser:
    """Layer-2 diagnoser. Emits the same DiagnosisResponse contract as the
    deterministic baseline, so it drops into the same harness.

    ``data`` is what a bundle is (ARC-9): customer-derived on the product path. A
    ``measurement`` names the pre-registered run the calls belong to (ARC-12); a public
    diagnosis exists only as such a measurement over the public corpus (ARC-19).
    """

    name = "llm"

    def __init__(self, door=None, *, data: str = DIAGNOSIS_CALL["data"],
                 measurement: str | None = None, max_tokens: int | None = None, fallback=None):
        from courant.config import get_settings

        if data not in DATA:
            raise ValueError(f"data must be one of {DATA}, not {data!r}")
        if measurement is not None and (not isinstance(measurement, str)
                                        or not measurement.strip()):
            raise ValueError("measurement names a pre-registered run, such as 'RM-12'")
        if data == "public" and measurement is None:
            raise ValueError("a public diagnosis exists only as a measurement over the public "
                             "corpus: name the pre-registered run (measurement='RM-nn'; "
                             "ARC-12, ARC-19)")
        self.data, self.measurement = data, measurement
        self.door = door if door is not None else default_door()
        self.max_tokens = max_tokens or get_settings().llm_max_tokens
        self.last_usage: dict | None = None      # token accounting for the last call
        self.last_provider: str | None = None
        self.last_call = None                    # the CallRecord of the last request (ARC-10)
        self.last_records: tuple = ()            # every CallRecord the last diagnosis sent
        #: Why the last diagnosis holds no model answer, or None when it holds one.
        self.last_error: str | None = None
        # The deterministic safety net. Not optional: if no model may answer, or none does,
        # the customer still gets a diagnosis rather than an error.
        from courant.diagnosis.deterministic import DeterministicDiagnoser

        self._fallback = fallback or DeterministicDiagnoser()

    def batch_identity(self) -> dict:
        """What a measured batch of this diagnoser's calls is (ARC-20): the measurement, the
        data class, the models the door would ask for it in order, and the prompt's version.
        Each item's own content is its request (``diagnosis_request``). Asks the door
        nothing over the network."""
        models = self.door.eligible(task=DIAGNOSIS_CALL["task"], data=self.data,
                                    capability=DIAGNOSIS_CALL["capability"],
                                    measurement=self.measurement)
        return {"measurement": self.measurement, "data": self.data,
                "models": [f"{c.provider}:{c.id}" for c in models],
                "prompt": PROMPT_VERSION}

    @property
    def model(self) -> str | None:
        """Which model actually answered last — not which one we intended.

        Recorded on the run, so a report always names the engine that produced
        it. With several allowlisted candidates, "the configured model" and "the
        model that replied" are different questions.
        """
        return self.last_call.model if self.last_call is not None else None

    def diagnose(self, bundle: EvidenceBundle) -> DiagnosisResponse:
        """The model's diagnosis of ``bundle``, after the control plane's checks.

        Without a measurement, any failure is the deterministic engine's answer (ARC-11).
        Under one, the deterministic answer is never returned: a stop (``stops_batch``) is
        raised, and any other failure gives an empty diagnosis, ``last_error`` saying why.
        An answer of no hypotheses under a measurement is the model's answer, nothing is
        wrong: an empty diagnosis and no ``last_error`` (ARC-20).
        """
        self.last_error, self.last_records = None, ()
        try:
            completion = self.door.complete(
                task=DIAGNOSIS_CALL["task"], data=self.data, **diagnosis_request(bundle),
                max_tokens=self.max_tokens, measurement=self.measurement,
            )
            self.last_call = completion.record
            self.last_records = completion.records
            self.last_usage = completion.usage
            self.last_provider = completion.provider

            diagnosis = self._sanitize(completion.data, bundle)
            if diagnosis is not None:
                return diagnosis
            if self.measurement is not None and completion.data.get("hypotheses") == []:
                return DiagnosisResponse(hypotheses=[])     # its answer: nothing is wrong
            self.last_error = f"{completion.model} named no valid hypothesis"
        except LLMError as exc:
            self.last_call = exc.records[-1] if exc.records else None
            self.last_records = exc.records
            if self.measurement is not None and stops_batch(exc):
                raise                                   # not the model's answer: stop (C12)
            self.last_error = exc.message
        except Exception as exc:  # noqa: BLE001 — a model must never break the loop
            self.last_error = f"LLM diagnosis failed ({type(exc).__name__})"

        if self.measurement is not None:
            log.warning("run %s: no usable model answer under %s (%s); the diagnosis is "
                        "empty", bundle.run_id, self.measurement, self.last_error)
            return DiagnosisResponse(hypotheses=[])
        log.warning("run %s: no model diagnosed (%s); using the deterministic engine",
                    bundle.run_id, self.last_error)
        return self._fallback.diagnose(bundle)

    @staticmethod
    def _sanitize(data: dict, bundle: EvidenceBundle) -> DiagnosisResponse | None:
        """Validate + defang the model output in the control plane."""
        valid_ids = {f.id for f in bundle.det_findings}
        hypotheses: list[Hypothesis] = []
        for h in data.get("hypotheses", []):
            code = h.get("root_cause_code")
            if code not in TAXONOMY:            # enum should prevent this; belt + braces
                continue
            refs = [r for r in h.get("evidence_refs", []) if r in valid_ids]
            conf = float(h.get("confidence", 0.0))
            conf = min(1.0, max(0.0, conf))
            hypotheses.append(Hypothesis(
                root_cause_code=code, confidence=conf,
                evidence_refs=refs, explanation=str(h.get("explanation", ""))[:500],
            ))
        if not hypotheses:
            return None
        return DiagnosisResponse(hypotheses=hypotheses)


def default_diagnoser(kb=None):
    """The diagnoser this deployment should use.

    Auto-selected everywhere — CLI, API, orchestrator — with no flags: the LLM
    engine when the model door has a model that may take a diagnosis (an
    allowlisted free model, zero-data-retention, measured on diagnosis, with its
    provider's key configured), the deterministic baseline otherwise. Until a
    pre-registered measurement clears a model for diagnosis (ARC-12), that is
    the deterministic baseline, whatever keys are set.

    Note what does *not* change: the product still watches, diagnoses, fixes,
    verifies, and reports, and the deterministic engine is what every offline
    customer runs on anyway.
    """
    door = default_door()
    chosen = door.eligible(**DIAGNOSIS_CALL)
    if chosen:
        import sys

        from courant.llm import contact_notice

        # Chosen with no flag, so said before any request (docs/NETWORK.md).
        print(f"courant: diagnosis by {chosen[0].provider}:{chosen[0].id}; "
              f"{contact_notice(chosen[0].provider)}; the run's evidence bundle is sent",
              file=sys.stderr)
        return LLMDiagnoser(door=door)

    from courant.diagnosis.deterministic import DeterministicDiagnoser

    return DeterministicDiagnoser(kb)


def unavailable_reason(door=None) -> str | None:
    """Why the LLM diagnoser cannot run on this installation, in the door's words, or None
    when it can: for example that no allowlisted model is measured for diagnosis (ARC-12)."""
    return (door if door is not None else default_door()).why_not(**DIAGNOSIS_CALL)
