"""A baseline judge. Every scorecard reports it so no number stands alone."""

from __future__ import annotations

from typing import cast

from courant.ctest.schema import JudgeAnswer, Task, Verdict

_VALID = ("TRUSTWORTHY", "NOT_TRUSTWORTHY", "NOT_ASSESSABLE")


class ConstantJudge:
    def __init__(self, verdict: str):
        if verdict not in _VALID:
            raise ValueError(f"constant judge verdict must be one of {_VALID}")
        self.verdict: Verdict = cast(Verdict, verdict)      # checked against _VALID above
        self.name = f"constant:{verdict}"

    def judge(self, task: Task) -> JudgeAnswer:
        return JudgeAnswer(task_id=task.task_id, verdict=self.verdict)
