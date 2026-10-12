"""Run a judge over the Courant Test (TRD CT-8)."""

from __future__ import annotations

from courant.ctest.build import load_labels, load_tasks
from courant.ctest.judges import make
from courant.ctest.judges.llm import PROMPT_VERSION, LLMJudge, judge_request
from courant.ctest.schema import JudgeAnswer
from courant.ctest.score import misses, score
from courant.ctest.splits import HELD_OUT_FAMILIES_V0, split_of


def run(judge_spec: str, split: str = "all", client=None, measurement: str | None = None,
        door=None) -> dict:
    """Score a judge. An ``llm:`` judge's result also carries ``calls``, a record of every
    request its model door sent (ARC-10); every other judge's result is as it always was.

    An ``llm:`` judge reached through the model door runs its tasks as one batch through
    ``Door.batch`` (ARC-19): the day's allowance is checked before the first request, every
    finished task is checkpointed with its answer, and a stop (a quota stop, a paid response,
    a refusal) propagates from here with no result, so it is never scored. Run again with the
    same arguments, it sends only the tasks that are not finished. The batch is keyed by the
    measurement, the judge and the prompt's version, and each task by the request it is sent
    (ARC-20): a changed prompt or task body is judged again. Such a result also carries
    ``finished``, the ledger stamp of its last task's answer, which a finished batch run again
    keeps, so it is never a new dated result (ARC-20).
    """
    tasks = load_tasks()
    labels = load_labels()
    if split != "all":
        tasks = [t for t in tasks if split_of(t.family) == split.replace("-", "_")]
    judge = make(judge_spec, client=client, measurement=measurement, door=door)
    finished = None
    if isinstance(judge, LLMJudge) and client is None:
        answers, calls, finished = _batched(judge, tasks, measurement, split)
    else:
        answers = [judge.judge(t) for t in tasks]
        calls = getattr(judge, "calls", None)
    result = {
        "benchmark": "courant-test/judge-v0",
        "judge": judge.name,
        "split": split,
        "held_out_families": list(HELD_OUT_FAMILIES_V0),
        "scorecard": score(tasks, labels, answers),
        "misses": misses(tasks, labels, answers),
        "answers": [a.model_dump(mode="json") for a in answers],
    }
    if calls is not None:
        result["calls"] = list(calls)
    if isinstance(judge, LLMJudge) and client is None:
        result["finished"] = finished
    return result


def _batched(judge: LLMJudge, tasks, measurement: str | None,
             split: str) -> tuple[list[JudgeAnswer], list[dict], str | None]:
    """Every task's answer, every finished task's request records in task order, and when the
    last task was finished, from one public batch on the judge's own door (ARC-19 C10,
    ARC-20)."""

    def judge_one(task) -> dict:
        before = len(judge.calls)
        answer = judge.judge(task)
        return {"answer": answer.model_dump(mode="json"), "calls": judge.calls[before:]}

    held = judge.door.batch(f"ctest-{measurement or 'unmeasured'}-{judge.name}-{split}",
                            {t.task_id: t for t in tasks}, judge_one, data="public",
                            provider=judge.provider, model=judge.model,
                            identity={"measurement": measurement, "judge": judge.name,
                                      "prompt": PROMPT_VERSION},
                            content=judge_request, task="judge", measurement=judge.measurement)
    answers = [JudgeAnswer.model_validate(held[t.task_id]["answer"]) for t in tasks]
    calls = [call for t in tasks for call in held[t.task_id]["calls"]]
    return answers, calls, max(held.finished.values(), default=None)
