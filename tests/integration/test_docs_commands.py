"""Docs are tests (TRD PUB-3; T-PUB-2).

Every fenced `bash` block in the README and `docs/GETTING-STARTED.md` runs, command by command,
on the bundled demo data, in a scratch directory, and each command's exit code matches the
document. A command with no note exits 0; `# exit N` after a command says it exits N. A block
that cannot run in CI (an installer, a scheduler, a server that waits for a client) is marked
by `<!-- docs-skip: why -->` on the line before its fence, and every skip is listed here.

The commands name the placeholders the documents use (`./my_case`, `./runs`, `spec.json`); the
scratch directory has them, made from the demo data that ships inside Courant. `<hash>` is the
output of the last `courant spec hash`.
"""

from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app

ROOT = Path(__file__).resolve().parents[2]
#: The public repository's README (public/README.md here, README.md there) is run too.
DOCS = [p for p in (ROOT / "README.md", ROOT / "public" / "README.md",
                    ROOT / "docs" / "GETTING-STARTED.md") if p.exists()]

_FENCE = re.compile(r"(?P<pre><!-- docs-skip: (?P<why>[^>]*?) -->\n)?```bash\n(?P<body>.*?)```",
                    re.DOTALL)
_EXIT = re.compile(r"#\s*exit\s+(\d+)\b")


@dataclass(frozen=True)
class Block:
    doc: str
    line: int
    commands: tuple[tuple[str, int], ...]      # (command line, documented exit code)
    skip: str | None


def blocks(path: Path) -> list[Block]:
    text = path.read_text()
    out = []
    for m in _FENCE.finditer(text):
        cmds = []
        for raw in m.group("body").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            note = _EXIT.search(line)
            code = int(note.group(1)) if note else 0
            cmd = re.sub(r"\s+#.*$", "", line)          # a trailing comment is not a command
            cmds.append((cmd, code))
        out.append(Block(path.name, text[:m.start()].count("\n") + 1, tuple(cmds),
                         m.group("why").strip() if m.group("why") else None))
    return out


ALL = [b for doc in DOCS for b in blocks(doc)]


def _id(b: Block) -> str:
    return f"{b.doc}:{b.line}"


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """The placeholders the documents use, made from the demo data inside Courant."""
    from courant.data import DEMO, extract_demo_case

    work = tmp_path_factory.mktemp("docs-work")
    case = extract_demo_case(work / "_demo")
    for name in ("my_case", "reference_case", "agent_case"):
        shutil.copytree(case, work / name)
    shutil.copy(case / "log.simpleFoam", work / "log.simpleFoam")
    runs = work / "runs"
    runs.mkdir()
    for sid in ("icoFoam_cavity__healthy", "icoFoam_cavity__cfl_blowup",
                "icoFoam_cavity__transport_units"):
        shutil.copy(DEMO / f"{sid}.log.gz", runs / f"{sid}.log.gz")
        shutil.copy(DEMO / f"{sid}.case.json", runs / f"{sid}.case.json")
    return work


def test_the_documents_have_command_blocks_and_every_skip_says_why():
    assert len(ALL) >= 10, "the parser found too few blocks"
    assert any(b.doc == "README.md" for b in ALL) and any(b.doc != "README.md" for b in ALL)
    skipped = [b for b in ALL if b.skip is not None]
    assert all(b.skip for b in skipped), "a docs-skip must give its reason"
    assert len(skipped) <= len(ALL) // 2, "most blocks must run, not be skipped"
    for b in skipped:
        print(f"skipped {_id(b)}: {b.skip}")


def test_every_runnable_command_is_a_courant_command():
    for b in ALL:
        if b.skip is not None:
            continue
        for cmd, _ in b.commands:
            assert shlex.split(cmd)[0] == "courant", (
                f"{_id(b)}: {cmd!r} is not a courant command; mark the block with "
                "<!-- docs-skip: why --> if it cannot run in CI")


@pytest.mark.parametrize("block", [b for b in ALL if b.skip is None], ids=_id)
def test_a_documented_block_runs_with_its_documented_exit_codes(block, workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    runner = CliRunner()
    lock = None
    for cmd, expected in block.commands:
        args = shlex.split(cmd)[1:]
        if "<hash>" in args:
            assert lock, f"{_id(block)}: <hash> before any `courant spec hash`"
            args = [lock if a == "<hash>" else a for a in args]
        result = runner.invoke(app, args)
        assert not isinstance(result.exception, Exception) or isinstance(
            result.exception, SystemExit), f"{cmd!r} raised {result.exception!r}"
        assert "Traceback" not in result.output, cmd
        assert result.exit_code == expected, (
            f"{_id(block)}: `{cmd}` exited {result.exit_code}, the document says {expected}\n"
            f"{result.output[-1500:]}")
        if args[:2] == ["spec", "hash"]:
            lock = result.output.strip().split()[-1]
            assert re.fullmatch(r"[0-9a-f]{64}", lock)


def test_the_batch_example_writes_its_junit_file(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    for b in ALL:
        if any("--junit" in c for c, _ in b.commands) and b.skip is None:
            assert (workdir / "courant.xml").exists()
            return
    pytest.fail("no documented block writes a JUnit file")


def test_the_documented_verdict_words_are_the_current_ones():
    text = (ROOT / "docs" / "GETTING-STARTED.md").read_text()
    for word in ("accept", "reject", "insufficient"):
        assert f"`{word}`" in text
    assert "insufficient." in text                   # the closed vocabulary is named
