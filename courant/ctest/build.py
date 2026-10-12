"""Build Courant Test judge-v0 from the real corpus (TRD CT-3).

Task IDs are a hash of the log, so neither the ID nor the task text names the
injected fault. The build refuses to write a task whose payload contains its
own fault identifier. A benchmark that leaks its labels measures reading, not
judgement.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from courant.ctest.schema import Label, Task
from courant.verify.schema import canonical_json

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
DATA = Path(__file__).resolve().parent / "data"
TASKS = DATA / "judge-v0.tasks.jsonl.gz"
LABELS = DATA / "judge-v0.labels.jsonl"


class LeakError(ValueError):
    pass


#: Labels revision 2 (TRD CT-11). Decided by S on 30 Sep 2026: the live certification matrix
#: retired this scenario as a fault on 16 Aug, because the adaptive time-step controller
#: recovers it (sustained Co 1.085 against 1.083 untouched). The fault id stays, for
#: provenance. The expected code is cleared, because a healthy run has no cause to name.
RELABELLED_HEALTHY = {"interFoam_damBreak::cfl_blowup"}


def build_judge_v0(corpus: Path = CORPUS) -> tuple[list[Task], list[Label]]:
    tasks: list[Task] = []
    labels: list[Label] = []
    for log_path in sorted(corpus.glob("*.log.gz")):
        meta_path = log_path.with_name(log_path.name.replace(".log.gz", ".case.json"))
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        with gzip.open(log_path, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        task_id = "ct.judge.v0." + hashlib.sha256(text.encode()).hexdigest()[:12]
        task = Task(task_id=task_id, family=meta["case_id"], solver=meta["solver"],
                    log_name=f"log.{meta['solver']}", log_text=text,
                    config_view=meta.get("config"), mesh_view=meta.get("mesh"))
        fault = meta.get("fault_id")
        if fault:
            payload = task.model_dump_json()
            if fault in payload or fault.replace("_", " ") in payload:
                raise LeakError(f"{meta['scenario_id']}: the task payload names its own fault")
        tasks.append(task)
        healthy = fault is None or meta.get("scenario_id") in RELABELLED_HEALTHY
        labels.append(Label(task_id=task_id,
                            truth="TRUSTWORTHY" if healthy else "NOT_TRUSTWORTHY",
                            expects_code=None if healthy else meta.get("expects_code"),
                            fault_id=fault))
    order = sorted(range(len(tasks)), key=lambda i: tasks[i].task_id)
    return [tasks[i] for i in order], [labels[i] for i in order]


def write(tasks: list[Task], labels: list[Label]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    body = "".join(canonical_json(t.model_dump(mode="json")) for t in tasks).encode()
    # mtime=0 keeps the gzip bytes identical across builds.
    with gzip.GzipFile(TASKS, "wb", mtime=0) as fh:
        fh.write(body)
    LABELS.write_text("".join(canonical_json(lab.model_dump(mode="json")) for lab in labels))


def load_tasks(path: Path = TASKS) -> list[Task]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [Task.model_validate_json(line) for line in fh if line.strip()]


def load_labels(path: Path = LABELS) -> dict[str, Label]:
    return {lab.task_id: lab for lab in (Label.model_validate_json(line)
                                   for line in path.read_text().splitlines() if line.strip())}
