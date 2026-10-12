"""A baseline judge. Every scorecard reports it so no number stands alone."""

from __future__ import annotations

from courant.ctest.schema import JudgeAnswer, Task

_VALID = ("TRUSTWORTHY", "NOT_TRUSTWORTHY", "NOT_ASSESSABLE")


class ConstantJudge:
    def __init__(self, verdict: str):
        if verdict not in _VALID:
            raise ValueError(f"constant judge verdict must be one of {_VALID}")
        self.verdict = verdict
        self.name = f"constant:{verdict}"

    def judge(self, task: Task) -> JudgeAnswer:
        return JudgeAnswer(task_id=task.task_id, verdict=self.verdict)
