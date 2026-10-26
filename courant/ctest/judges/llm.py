"""A language model as a judge (TRD CT-6). It reaches a model only through the model door.

The prompt is fixed and versioned, the same for every model. The answer must
be JSON with the three JudgeAnswer fields. Anything else is recorded as an
error and scored as a wrong answer, never repaired into a guess.

Courant Test tasks are generated from public tutorials, so a judge's call is
public data (ARC-9). The door lets it through only to an allowlisted free model,
or to a paid one S allowed with ``COURANT_ALLOW_PAID_LLM=1``, and only for a
model measured on judging or a call naming the pre-registered measurement it
belongs to (ARC-12). Every request leaves a record in ``LLMJudge.calls``, which
a Courant Test run reports (ARC-10). Tests inject a fake client, or a door over
a fake network.

A model that fails on a task (no usable answer, a failed exchange) abstains on
that task. A stop is not the model's answer (``courant.llm.stops_batch``: a
quota stop, a paid response, a refusal, a rejected key or a model that is not
served, an unwritable ledger): it is raised, so the run stops and nothing is
scored (ARC-19).

``judge/2`` (RRD RM-36, ``courant/ctest/judges/llm_spec.py``) is this prompt with the task's
locked specification after it; it sends through ``LLMJudge._ask``, and ``judge/1`` is unchanged.
"""

from __future__ import annotations

import json
import re

from courant.ctest.schema import JudgeAnswer, Task

PROMPT_VERSION = "judge/1"
MAX_LOG_CHARS = 60_000
#: The most output tokens a judge's request asks for (thinking included): RM-02.H2 v2's.
MAX_OUTPUT_TOKENS = 8000

PROMPT = """You are judging whether a completed OpenFOAM simulation can be trusted.

You get the solver log, the case configuration as parsed, and mesh-quality metrics when
available. There is no reference solution. Decide from the evidence alone.

Answer with JSON only, exactly this shape:
{{"verdict": "TRUSTWORTHY" | "NOT_TRUSTWORTHY" | "NOT_ASSESSABLE",
  "codes": [up to three failure-mode codes from the list below, most likely first; empty if trustworthy],
  "rationale": "one or two sentences"}}

Failure-mode codes:
{taxonomy}

Solver: {solver}
Case configuration:
{config}

Mesh metrics:
{mesh}

Solver log ({log_name}{truncated}):
{log}
"""


def _truncate(text: str) -> tuple[str, str]:
    if len(text) <= MAX_LOG_CHARS:
        return text, ""
    half = MAX_LOG_CHARS // 2
    return text[:half] + "\n[... middle of log truncated ...]\n" + text[-half:], ", truncated in the middle"


def render_prompt(task: Task) -> str:
    from courant.taxonomy import TAXONOMY

    log, note = _truncate(task.log_text)
    taxonomy = "\n".join(f"- {code}: {entry.title}" for code, entry in sorted(TAXONOMY.items()))
    return PROMPT.format(taxonomy=taxonomy, solver=task.solver,
                         config=json.dumps(task.config_view or {}, sort_keys=True, indent=1)[:20_000],
                         mesh=json.dumps(task.mesh_view or {}, sort_keys=True),
                         log_name=task.log_name, truncated=note, log=log)


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(task_id: str, text: str) -> JudgeAnswer:
    m = _JSON.search(text or "")
    if not m:
        return JudgeAnswer(task_id=task_id, verdict="NOT_ASSESSABLE", error="no JSON in answer")
    try:
        data = json.loads(m.group(0))
        return JudgeAnswer(task_id=task_id, verdict=data["verdict"],
                           codes=tuple(data.get("codes") or ())[:3],
                           rationale=str(data.get("rationale", ""))[:500])
    except Exception as exc:                            # noqa: BLE001
        return JudgeAnswer(task_id=task_id, verdict="NOT_ASSESSABLE",
                           error=f"unparseable answer: {type(exc).__name__}")


SYSTEM = ("You judge completed engineering simulations from their evidence alone. "
          "Answer only with the requested JSON.")

def judge_request(task: Task) -> dict:
    """What an LLM judge asks the model about ``task`` (``courant.llm.request_of``): the one
    definition the judge sends and its batch keys each task by (ARC-20)."""
    from courant.llm import request_of

    return request_of(system=SYSTEM, user=render_prompt(task), schema=ANSWER_SCHEMA)


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["TRUSTWORTHY", "NOT_TRUSTWORTHY", "NOT_ASSESSABLE"]},
        "codes": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "codes", "rationale"],
    "additionalProperties": False,
}


class LLMJudge:
    """A model behind the product's one model door (courant/llm/, ARC-5)."""

    def __init__(self, provider: str, model: str, client=None, measurement: str | None = None,
                 door=None):
        self.provider, self.model = provider, model
        self.name = f"llm:{provider}:{model}"
        self.measurement = measurement
        self._client = client
        self._door = door
        #: One record per request the door sent, failed ones included (ARC-10).
        self.calls: list[dict] = []

    @property
    def door(self):
        """The model door this judge calls through: the one it was given, else the
        installation's, made when first needed. A run batches its tasks on the same door."""
        if self._door is None:
            from courant.llm import default_door

            self._door = default_door()
        return self._door

    def _complete(self, task: Task) -> str:
        if self._client is not None:
            return self._client(render_prompt(task))
        return self._ask(judge_request(task))

    def _ask(self, request: dict) -> str:
        """One request through the door, recorded in ``calls`` whether it answered or not;
        the answer the door validated against ``ANSWER_SCHEMA``, as JSON text."""
        from courant.llm import LLMError

        try:
            completion = self.door.complete(
                task="judge", data="public", **request,
                max_tokens=MAX_OUTPUT_TOKENS, model=self.model, provider=self.provider,
                measurement=self.measurement)
        except LLMError as exc:
            self.calls += [record.to_dict() for record in exc.records]
            raise
        self.calls += [record.to_dict() for record in completion.records]
        # The answer the door validated against ANSWER_SCHEMA, not a re-reading of its text.
        return json.dumps(completion.data)

    def judge(self, task: Task) -> JudgeAnswer:
        """The model's answer on one task, or its abstention when it gave none it could use.
        A stop (``stops_batch``) is raised instead: it is not this model's answer (ARC-19)."""
        try:
            return parse_answer(task.task_id, self._complete(task))
        except Exception as exc:
            from courant.llm.errors import stops_batch

            if stops_batch(exc):
                raise
            return JudgeAnswer(task_id=task.task_id, verdict="NOT_ASSESSABLE",
                               error=f"judge failed: {type(exc).__name__}")
