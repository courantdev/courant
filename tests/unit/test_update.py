"""Self-update (Tech Bible §16, Phase B.4).

Two things here are worth real tests. The first is platform detection, because
getting it wrong means downloading an ARM binary onto an x86 login node and
telling the customer their machine is broken. The second is the atomic swap,
because a failed update that leaves a truncated executable on PATH is worse than
never updating at all.
"""

from __future__ import annotations

import stat

import pytest

from courant.cli.commands.update import _replace, target_triple


class TestPlatformDetection:
    @pytest.mark.parametrize("system,machine,expected", [
        ("Darwin", "arm64", "darwin-arm64"),
        ("Darwin", "x86_64", "darwin-x86_64"),
        ("Linux", "x86_64", "linux-x86_64"),
        ("Linux", "amd64", "linux-x86_64"),      # some kernels report this
        ("Linux", "aarch64", "linux-arm64"),     # what most ARM Linux reports
        ("Linux", "arm64", "linux-arm64"),
    ])
    def test_the_right_asset_is_chosen(self, monkeypatch, system, machine, expected):
        monkeypatch.setattr("platform.system", lambda: system)
        monkeypatch.setattr("platform.machine", lambda: machine)
        assert target_triple() == expected

    @pytest.mark.parametrize("system,machine", [
        ("Windows", "AMD64"),        # no binary published yet
        ("Linux", "ppc64le"),        # real, and on some HPC systems
        ("FreeBSD", "x86_64"),
    ])
    def test_an_unshipped_platform_says_so_rather_than_guessing(
        self, monkeypatch, system, machine
    ):
        """Guessing here downloads a binary that cannot run. The pip path is the
        honest answer."""
        monkeypatch.setattr("platform.system", lambda: system)
        monkeypatch.setattr("platform.machine", lambda: machine)
        assert target_triple() is None


class TestTheSwap:
    def test_the_binary_is_replaced_and_stays_executable(self, tmp_path):
        binary = tmp_path / "courant"
        binary.write_bytes(b"old")
        binary.chmod(0o755)

        _replace(binary, b"new")

        assert binary.read_bytes() == b"new"
        assert binary.stat().st_mode & stat.S_IXUSR

    def test_no_backup_is_left_behind_on_success(self, tmp_path):
        binary = tmp_path / "courant"
        binary.write_bytes(b"old")
        binary.chmod(0o755)

        _replace(binary, b"new")

        assert not (tmp_path / "courant.old").exists()
        assert [p.name for p in tmp_path.iterdir()] == ["courant"]

    def test_a_failed_swap_leaves_the_working_binary_alone(self, tmp_path, monkeypatch):
        """The disk fills mid-update. The machine must still have a working
        agent — a truncated executable on PATH is worse than a stale one."""
        binary = tmp_path / "courant"
        binary.write_bytes(b"old")
        binary.chmod(0o755)

        def explode(*_a, **_kw):
            raise OSError("no space left on device")

        monkeypatch.setattr("os.replace", explode)

        with pytest.raises(OSError):
            _replace(binary, b"new")

        assert binary.read_bytes() == b"old"

    def test_no_staging_file_survives_a_failure(self, tmp_path, monkeypatch):
        binary = tmp_path / "courant"
        binary.write_bytes(b"old")
        binary.chmod(0o755)

        monkeypatch.setattr("os.replace", lambda *_a, **_kw: (_ for _ in ()).throw(
            OSError("boom")
        ))

        with pytest.raises(OSError):
            _replace(binary, b"new")

        leftovers = [p.name for p in tmp_path.iterdir()
                     if p.name.startswith(".courant-update-")]
        assert leftovers == [], f"staging files left behind: {leftovers}"
