"""Invariant 6: every reader refuses or reads, never crashes (and never hangs), on mutated real
dictionaries and field files taken from tests/fixtures (`readerfuzz.mutant`).

The mutants run in a child process (`python -m tests.invariants.readerfuzz`), which prints when
it starts and ends each one; a mutant that has not ended within `BUDGET_S` is a hang: the child
is killed and the run resumes at the next mutant. A hang is a failure like a traceback: a gate
that never answers is a gate the run goes around.

The child runs every reader with `case.foam_guard` on, in both modes (`readerfuzz._with_the_guard`):
under the default foamlib still hangs on a list cut mid-write (A6-1, until T-FOAMGUARD-2).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

import pytest

from tests.invariants import gen

BUDGET_S = 20.0
N_READERS_FAST, N_READERS_SLOW = 150, 5000
N_VERIFY_FAST, N_VERIFY_SLOW = 40, 2000


def _pump(proc, lines: queue.Queue) -> None:
    for ln in proc.stdout:
        lines.put(ln)
    lines.put(None)


def run_lane(lane: str, start: int, stop: int, budget: float = BUDGET_S):
    """{k: "ok" | "crash: ..." | "hang"} for mutants start..stop of a lane."""
    out: dict[int, str] = {}
    k = start
    while k < stop:
        proc = subprocess.Popen([sys.executable, "-m", "tests.invariants.readerfuzz", lane,
                                 str(k), str(stop)], cwd=gen.ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True,
                                env={**os.environ, "PYTHONHASHSEED": "0"})
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=_pump, args=(proc, lines), daemon=True).start()
        current = None
        while True:
            try:
                ln = lines.get(timeout=budget)
            except queue.Empty:                       # the current mutant did not end in time
                proc.kill()
                proc.wait()
                out[current] = "hang"
                k = current + 1
                break
            if ln is None:                            # the child ended
                proc.wait()
                if current is None:
                    raise RuntimeError(f"the fuzz child ended before mutant {k} "
                                       f"({proc.returncode})")
                if current not in out:                # it died inside a mutant
                    out[current] = f"crash: the process ended ({proc.returncode})"
                k = current + 1
                break
            rec = json.loads(ln)
            if "start" in rec:
                current = rec["k"]
            elif rec.get("ok"):
                out[rec["k"]] = "ok"
            else:
                out[rec["k"]] = f"crash: {rec['crash']} at {rec['where']}"
    return out


def _failures(out):
    from tests.invariants.readerfuzz import mutant

    return [f"{mutant(k)[0]}: {why}" for k, why in sorted(out.items()) if why != "ok"]


def _check(lane: str, n: int):
    out = run_lane(lane, 0, n)
    assert len(out) == n, f"{n - len(out)} mutants were never run"
    bad = _failures(out)
    assert not bad, f"{len(bad)} of {n} mutants crashed or hung:\n" + "\n".join(bad[:20])


def test_the_readers_refuse_or_read():
    _check("readers", N_READERS_FAST)


@pytest.mark.slow
def test_the_readers_refuse_or_read_at_scale():
    _check("readers", N_READERS_SLOW)


def test_the_gate_gives_a_verdict_or_an_input_error():
    _check("verify", N_VERIFY_FAST)


@pytest.mark.slow
def test_the_gate_gives_a_verdict_or_an_input_error_at_scale():
    _check("verify", N_VERIFY_SLOW)
