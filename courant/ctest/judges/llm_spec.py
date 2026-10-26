"""A language model as a judge with the task's specification: prompt ``judge/2`` (RRD RM-36,
pre-registered 15 Oct 2026).

``judge/2`` is RM-02.H2 v2's ``judge/1`` (``courant/ctest/judges/llm.py``) byte for byte, the
solver log middle-truncated at 60,000 characters, the case configuration and the mesh metrics as
``judge/1`` gives them, and then the task's locked specification (``courant.spec/1``) and the
question whether the run is a valid result of the specified problem. The system prompt, the
answer schema, the door, the 8,000 output tokens and what stops a batch are ``judge/1``'s
(``LLMJudge``), so nothing but the specification and the question differs.
"""

from __future__ import annotations

import json

from courant.ctest.judges.llm import ANSWER_SCHEMA, SYSTEM, LLMJudge, render_prompt
from courant.ctest.schema import Task

PROMPT_VERSION = "judge/2"

#: What judge/2 adds after judge/1's whole prompt: the specification and the question.
SPEC_PROMPT = """
The task this run was given, as its locked specification (courant.spec/1): the problem it was
asked to solve, the values it names and their tolerances.
{spec}

Decide whether this run is a valid result of the problem this specification states:
TRUSTWORTHY if the run can be trusted and solves the specified problem; NOT_TRUSTWORTHY if it
cannot be trusted or solves a different problem; NOT_ASSESSABLE if the evidence cannot decide.
Answer with JSON only, in the shape above, the most probable failure-mode codes first.
"""


class SpecTask(Task):
    """A judge/2 task: everything a judge/1 task carries, and the locked specification of the
    task the run was given (``courant.spec/1``). Never the label, never the fault's name."""

    spec: dict


def render_prompt_2(task: SpecTask) -> str:
    """judge/2: judge/1's prompt for ``task``, byte for byte (its log middle-truncated at
    60,000 characters), then the specification and the question (``SPEC_PROMPT``)."""
    return render_prompt(task) + SPEC_PROMPT.format(
        spec=json.dumps(task.spec, sort_keys=True, indent=1))


def spec_judge_request(task: SpecTask) -> dict:
    """What a judge/2 judge asks the model about ``task``: judge/1's system prompt and answer
    schema with judge/2's user prompt (``courant.llm.request_of``): the one definition the judge
    sends and its batch keys each run by (ARC-20)."""
    from courant.llm import request_of

    return request_of(system=SYSTEM, user=render_prompt_2(task), schema=ANSWER_SCHEMA)


class SpecJudge(LLMJudge):
    """A model judging with judge/2: judge/1's door, answer, records and stops (``LLMJudge``),
    asked whether the run is a valid result of the specified problem."""

    def _complete(self, task: SpecTask) -> str:
        if self._client is not None:
            return self._client(render_prompt_2(task))
        return self._ask(spec_judge_request(task))
