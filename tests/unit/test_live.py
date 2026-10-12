"""The live advisory: chunking, prefixes, monotone FAILING, read-only watch (TRD LIVE-1 to LIVE-6)."""

from __future__ import annotations

import gzip
import json
import random
import threading
import time
from pathlib import Path

import pytest

from courant.cli.tail import Chunk, LogTail
from courant.live import FAILING, RUNNING, LiveEngine, state_of_prefix
from courant.live.watch import run_watch

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
MONITORED = ROOT / "certification" / "monitored"


def corpus(name: str) -> str:
    return gzip.open(CORPUS / f"{name}.log.gz", "rt").read()


HEALTHY = "icoFoam_cavity__healthy"
CRASH = "icoFoam_cavity__extreme_aspect_ratio"           # crashes at step 4 (SIGFPE)


def _summary(state):
    return (state.state, state.steps, state.alarm.code if state.alarm else None,
            state.alarm.step if state.alarm else None)


def test_feeding_in_pieces_equals_feeding_at_once():
    text = corpus(CRASH)
    whole = state_of_prefix(text)
    rng = random.Random(3)
    eng = LiveEngine()
    i = 0
    while i < len(text):
        j = min(len(text), i + rng.randint(1, 4000))
        eng.feed(text[i:j])
        i = j
    assert _summary(eng.state()) == _summary(whole)
    once = LiveEngine()
    once.feed(text)
    assert [(e.kind, e.log_offset) for e in eng.events] == [(e.kind, e.log_offset) for e in once.events]


@pytest.mark.parametrize("name", [HEALTHY, CRASH])
def test_the_state_at_an_offset_is_the_state_of_the_prefix(name):
    text = corpus(name)
    cuts = sorted({int(len(text) * k / 10) for k in range(1, 11)})
    eng = LiveEngine()
    done = 0
    for cut in cuts:
        eng.feed(text[done:cut])
        done = cut
        assert _summary(eng.state()) == _summary(state_of_prefix(text[:cut])), cut


def test_failing_is_monotone():
    text = corpus(CRASH)
    states = [state_of_prefix(text[:int(len(text) * k / 20)]).state for k in range(1, 21)]
    first = states.index(FAILING)
    assert all(s == FAILING for s in states[first:])


def test_an_unfinished_healthy_log_is_running_not_a_crash():
    text = corpus(HEALTHY)
    st = state_of_prefix(text[: len(text) // 2])
    assert st.state == RUNNING and st.alarm is None


@pytest.mark.parametrize("name", [HEALTHY, "simpleFoam_pitzDaily__healthy", "pisoFoam_RAS_cavity__healthy"])
def test_healthy_runs_never_fail(name):
    assert state_of_prefix(corpus(name)).state == RUNNING     # monotone: never on any prefix


# ------------------------------------------------------------------ run_watch
def _watch(tmp_path, text, **kw):
    from courant.verify import verify_path

    log = tmp_path / "log.icoFoam"
    log.write_text(text)
    lines: list[str] = []
    out = run_watch([Chunk(text=text, offset=0)], case_dir=None, log_path=log, emit=lines.append,
                    clock=iter(range(10 ** 6)).__next__, verify_fn=verify_path, **kw)
    return out, lines


def test_exit_on_fail_returns_at_the_alarm(tmp_path):
    out, lines = _watch(tmp_path, corpus(CRASH), exit_on_fail=True)
    assert out.exit_code == 2 and out.state.state == FAILING
    assert any("!! FAILING at step" in ln and "Courant does not act" in ln for ln in lines)
    assert out.verdict is None


def test_a_crashed_run_ends_with_its_verdict(tmp_path):
    out, lines = _watch(tmp_path, corpus(CRASH))
    assert out.verdict is not None and out.verdict.verdict == "reject"
    assert out.exit_code == 2


def test_a_finished_healthy_run_ends_with_trustworthy(tmp_path):
    out, lines = _watch(tmp_path, corpus(HEALTHY))
    assert out.exit_code == 0 and out.verdict.verdict == "accept"
    assert any(ln.startswith("VERDICT   accept") for ln in lines)


def test_json_mode_is_one_object_per_line(tmp_path):
    out, lines = _watch(tmp_path, corpus(CRASH), as_json=True)
    docs = [json.loads(ln) for ln in lines]
    assert docs[-1]["state"] == "FINISHED" and docs[-1]["verdict"]["verdict"] == "reject"
    assert any(d.get("alarm") for d in docs[:-1])


def test_the_watcher_bounds_its_own_overhead(tmp_path):
    from courant.verify import verify_path

    text = corpus(HEALTHY)
    log = tmp_path / "log.icoFoam"
    log.write_text(text)
    pieces = [Chunk(text=text[i:i + 200], offset=i) for i in range(0, len(text), 200)]
    ticks = iter(range(0, 10 ** 7))                   # one second per chunk
    out = run_watch(pieces, case_dir=None, log_path=log, emit=lambda s: None,
                    clock=ticks.__next__, interval=10.0, verify_fn=verify_path, as_json=True)
    assert out.evaluations <= len(pieces) // 10 + 2


def test_following_a_growing_file(tmp_path):
    text = corpus(HEALTHY)
    log = tmp_path / "log.icoFoam"
    log.write_text("")

    def writer():
        for i in range(0, len(text), 4000):
            with log.open("a") as fh:
                fh.write(text[i:i + 4000])
            time.sleep(0.005)

    t = threading.Thread(target=writer)
    t.start()
    lines: list[str] = []
    out = run_watch(LogTail(log, poll_interval=0.01, idle_timeout=2.0, start_timeout=2.0).follow(),
                    case_dir=None, log_path=log, emit=lines.append, interval=0.0)
    t.join()
    assert out.verdict is not None and out.verdict.verdict == "accept"
    assert out.exit_code == 0


def test_watch_is_read_only(tmp_path):
    import shutil

    case = tmp_path / "case"
    shutil.copytree(MONITORED / "simpleFoam_pitzDaily__fixed_budget", case)
    before = {p: p.read_bytes() for p in case.rglob("*") if p.is_file()}
    from typer.testing import CliRunner

    from courant.cli.main import app

    res = CliRunner().invoke(app, ["watch", str(case), "--once"])
    assert res.exit_code == 0, res.output
    assert {p: p.read_bytes() for p in case.rglob("*") if p.is_file()} == before
