"""The verifier as a judge, through exactly the function users call (ARC-4)."""

from __future__ import annotations

from courant.ctest.schema import JudgeAnswer, Task
from courant.verify import RunInputs, verify_inputs
from courant.verify.schema import InputFile


class CourantJudge:
    name = "courant"

    def judge(self, task: Task) -> JudgeAnswer:
        verdict = verify_inputs(RunInputs(
            run_id=task.task_id, log_text=task.log_text, log_name=task.log_name,
            config_view=task.config_view, mesh_view=task.mesh_view,
            inputs=(InputFile.of_text(task.log_text, task.log_name),),
            source="corpus", solver_hint=task.solver))
        verdict = verdict.to_v1()                 # GATE-3: the locked labels are version 1's
        return JudgeAnswer(
            task_id=task.task_id, verdict=verdict.verdict,
            codes=tuple(c.code for c in verdict.probable_causes),
            rationale=f"{verdict.rule}: " + "; ".join(r.code for r in verdict.reasons[:4]))
