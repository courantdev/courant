"""Following a live solver log.

The cases that matter here are the ones a cluster actually produces: a log that
does not exist yet because the scheduler has not started the job, and a log that
is replaced underneath us because the engineer re-ran after a fix. Both are
normal, and both would silently corrupt the event stream if mishandled.

Time is injected so these run instantly rather than sleeping through timeouts.
"""

from __future__ import annotations

from courant.cli.tail import Chunk, LogTail, find_log, wait_for_log


class FakeClock:
    """A monotonic clock that only advances when someone sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def tail(path, **kw) -> LogTail:
    clock = FakeClock()
    return LogTail(path, clock=clock.time, sleep=clock.sleep,
                   poll_interval=0.1, **kw)


class TestReadingFromTheStart:
    def test_a_finished_log_is_read_whole(self, tmp_path):
        log = tmp_path / "log.simpleFoam"
        log.write_text("Time = 0.1\nTime = 0.2\n")

        assert tail(log).read_all() == "Time = 0.1\nTime = 0.2\n"

    def test_offsets_are_absolute_file_offsets(self, tmp_path):
        """The server dedupes uploaded events on these. If they were relative
        to where we attached, a re-upload would double-count."""
        log = tmp_path / "log.simpleFoam"
        log.write_text("abc\n")

        follower = tail(log, idle_timeout=0.0)
        chunks = list(follower.follow())

        assert chunks[0].offset == 0
        assert follower.position == 4

    def test_attaching_late_still_reads_the_history(self, tmp_path):
        """A solve 4,000 iterations in has its evidence in those 4,000
        iterations. Starting at the tail would diagnose without it."""
        log = tmp_path / "log.simpleFoam"
        log.write_text("Time = 0.1\n" * 500)

        chunks = list(tail(log, idle_timeout=0.0).follow())
        assert "".join(c.text for c in chunks).count("Time") == 500


class TestGrowth:
    def test_only_new_bytes_come_back(self, tmp_path):
        log = tmp_path / "log.simpleFoam"
        log.write_text("first\n")

        follower = tail(log)
        assert follower.read_all() == "first\n"

        with log.open("a") as handle:
            handle.write("second\n")
        assert follower.read_all() == "second\n"

    def test_a_quiet_log_ends_the_stream(self, tmp_path):
        """A solver that stops writing has finished or wedged. Either way the
        agent must move on to verifying, not block forever."""
        log = tmp_path / "log.simpleFoam"
        log.write_text("Time = 0.1\n")

        chunks = list(tail(log, idle_timeout=1.0).follow())
        assert "".join(c.text for c in chunks) == "Time = 0.1\n"

    def test_stop_ends_the_stream_promptly(self, tmp_path):
        log = tmp_path / "log.simpleFoam"
        log.write_text("Time = 0.1\n")

        follower = tail(log, idle_timeout=None)
        produced = []
        for chunk in follower.follow():
            produced.append(chunk)
            follower.stop()          # e.g. Ctrl-C

        assert len(produced) == 1


class TestRestart:
    def test_truncation_is_reported_as_a_restart(self, tmp_path):
        """`simpleFoam > log.simpleFoam` after a fix resets the file. That is
        not corruption — it is the re-run this whole product exists to verify."""
        log = tmp_path / "log.simpleFoam"
        log.write_text("Time = 0.1\nTime = 0.2\n")

        follower = tail(log, idle_timeout=None)
        stream = follower.follow()
        assert next(stream).restarted is False

        log.write_text("Time = 0.1\n")          # shorter: the re-run began
        restarted = next(stream)

        assert restarted.restarted is True
        assert restarted.text == "Time = 0.1\n"
        assert restarted.offset == 0

    def test_replacement_is_detected_even_at_the_same_size(self, tmp_path):
        """A new inode at an identical size is invisible to a size check, and
        would make us skip the whole re-run."""
        log = tmp_path / "log.simpleFoam"
        log.write_text("aaaaaaaa\n")

        follower = tail(log, idle_timeout=None)
        stream = follower.follow()
        next(stream)

        log.unlink()
        log.write_text("bbbbbbbb\n")            # same length, new inode
        restarted = next(stream)

        assert restarted.restarted is True
        assert restarted.text == "bbbbbbbb\n"

    def test_an_empty_restart_is_announced_immediately(self, tmp_path):
        """The re-run has started but written nothing. The display should say
        so now, not after the first iteration lands."""
        log = tmp_path / "log.simpleFoam"
        log.write_text("Time = 0.1\n")

        follower = tail(log, idle_timeout=None)
        stream = follower.follow()
        next(stream)

        log.write_text("")
        restarted = next(stream)

        assert restarted == Chunk(text="", offset=0, restarted=True)


class TestWaitingForTheSolver:
    def test_a_log_that_appears_later_is_picked_up(self, tmp_path):
        """`courant watch` is typed before the solver starts, because that is
        the natural order and because a scheduler decides when a job runs."""
        log = tmp_path / "log.simpleFoam"
        clock = FakeClock()

        writes = {"done": False}
        original_sleep = clock.sleep

        def sleep_and_start_solver(seconds: float) -> None:
            original_sleep(seconds)
            if clock.now >= 1.0 and not writes["done"]:
                log.write_text("Time = 0.1\n")
                writes["done"] = True

        follower = LogTail(log, clock=clock.time, sleep=sleep_and_start_solver,
                           poll_interval=0.5, start_timeout=30.0, idle_timeout=1.0)
        chunks = list(follower.follow())

        assert "".join(c.text for c in chunks) == "Time = 0.1\n"

    def test_a_log_that_never_appears_gives_up(self, tmp_path):
        """A typo in the path must not hang a terminal all night."""
        follower = tail(tmp_path / "nope.log", start_timeout=5.0)
        assert list(follower.follow()) == []


class TestDecoding:
    def test_a_chunk_boundary_never_splits_a_character(self, tmp_path):
        """Decoding with errors='replace' would change the byte length and
        desynchronise every offset after it."""
        log = tmp_path / "log.simpleFoam"
        log.write_bytes("Δt = 0.001\n".encode())

        follower = tail(log)
        text = follower.read_all()

        assert text == "Δt = 0.001\n"
        assert follower.position == len(text.encode())


class TestWaitingForALogWhoseNameWeDoNotKnow:
    """`courant watch .` before the solver starts is the natural order — two
    panes, watcher in the first. Requiring the log to already exist would make
    the tool demand an ordering the engineer did not choose."""

    def test_a_log_that_appears_later_is_returned(self, tmp_path):
        clock = FakeClock()
        log = tmp_path / "log.simpleFoam"

        def sleep_then_start_solver(seconds: float) -> None:
            clock.sleep(seconds)
            if clock.now >= 2.0 and not log.exists():
                log.write_text("Time = 0.1\n")

        found = wait_for_log(tmp_path, timeout=30.0, poll_interval=1.0,
                             clock=clock.time, sleep=sleep_then_start_solver)
        assert found is not None and found.name == "log.simpleFoam"

    def test_an_existing_log_returns_without_waiting(self, tmp_path):
        clock = FakeClock()
        (tmp_path / "log.icoFoam").write_text("x")

        assert wait_for_log(tmp_path, clock=clock.time, sleep=clock.sleep)
        assert clock.now == 0.0, "it waited when it did not need to"

    def test_giving_up_is_reported_not_raised(self, tmp_path):
        clock = FakeClock()
        assert wait_for_log(tmp_path, timeout=5.0, poll_interval=1.0,
                            clock=clock.time, sleep=clock.sleep) is None

    def test_the_wait_is_announced_once(self, tmp_path):
        """Silence for ten minutes looks like a hang. One line is enough."""
        clock = FakeClock()
        announcements = []

        wait_for_log(tmp_path, timeout=5.0, poll_interval=1.0,
                     on_wait=lambda: announcements.append(1),
                     clock=clock.time, sleep=clock.sleep)

        assert len(announcements) == 1


class TestFindLog:
    def test_the_openfoam_convention_is_found(self, tmp_path):
        (tmp_path / "log.simpleFoam").write_text("x")
        assert find_log(tmp_path).name == "log.simpleFoam"

    def test_the_newest_wins(self, tmp_path):
        import os

        old = tmp_path / "log.icoFoam"
        new = tmp_path / "log.simpleFoam"
        old.write_text("x")
        new.write_text("x")
        os.utime(old, (1, 1))

        assert find_log(tmp_path).name == "log.simpleFoam"

    def test_archives_are_ignored(self, tmp_path):
        (tmp_path / "log.simpleFoam.gz").write_text("x")
        assert find_log(tmp_path) is None

    def test_nothing_to_find_is_not_an_error(self, tmp_path):
        assert find_log(tmp_path) is None
