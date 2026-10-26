"""courant-mcp: stdio MCP server (TRD MCP-1). Needs the `mcp` extra (mcp>=2.2).

    courant-mcp --root ~/cases --root /scratch/runs

Every tool is read-only and confined to the given roots (default: the
current directory, unless that is /, the home directory or a directory that
holds it): every file the tools read, a file reached through a link included,
and every link in the case.
"""

from __future__ import annotations

import argparse
import contextvars
import threading
from pathlib import Path

from courant.mcp import tools

INSTRUCTIONS = (
    "Courant verifies engineering simulations. After running an OpenFOAM case, call "
    "verify_run on the case directory before reporting success, passing the task "
    "specification as `spec` when you were given one. If the verdict is reject, "
    "read the reasons, and either correct the setup or stop and tell the human. Never "
    "present a reject or insufficient result as a finished answer, and never edit "
    "the specification to make a case conform. Before you tell the user a run worked, call "
    "score_agent_output with the task specification and `reported_success`: a silent error "
    "(you would report success on a run the gate rejects) is the failure it exists to stop."
)


def unconfined_default(roots: list[str]) -> str | None:
    """Why the server will not start, or None (MCP-3). With no --root the root is the working
    directory, and MCP hosts often start their servers in / or in the home directory, where that
    root would let the tools read every file the user can: so in /, in the home directory, and
    in any directory that holds it (/Users, /home), whose root would hold every user's home
    directory, the roots must be named."""
    if roots:
        return None
    cwd = Path.cwd().resolve()
    try:
        home = Path.home().resolve()
    except RuntimeError:                                    # no home directory to compare
        home = None
    if cwd != Path(cwd.anchor) and (home is None or not home.is_relative_to(cwd)):
        return None
    return (f"no --root given, and the working directory is {cwd}, which would let the tools "
            "read every file under it: name each directory they may read with --root DIR")


#: MCP-5: the wall-clock limit on one tool call, in seconds. A call that has not answered by then
#: is answered with an error, so an agent's call returns when a case file blocks the reader that
#: opens it (a FIFO). It cannot answer a call stuck in C code that holds the interpreter's lock
#: (`within`); the readers bound the one such work known, matching a case-written keyword
#: pattern (`courant.case.wordre`, TRD SPEC-2).
TIMEOUT = 600.0


class TimedOut(Exception):
    pass


def within(seconds: float | None, call):
    """`call()`, or `TimedOut` once it has run `seconds` without returning (None or 0: no limit).

    It runs on a daemon thread, in a copy of the caller's context. A call that times out is left
    running, since a thread cannot be stopped, and the answer it may give later is dropped. The
    caller takes the interpreter's lock back to time out only when the call releases it: a call
    blocked on I/O or running Python code is answered at `seconds`; one inside C code that holds
    the lock (Python's `re` backtracking on a pattern, before the keyword matcher was bounded)
    is not answered until that code returns (the final review of 0.6.0, 21 Oct 2026)."""
    if not seconds:
        return call()
    box: dict = {}
    context = contextvars.copy_context()

    def run():
        try:
            box["value"] = context.run(call)
        except BaseException as exc:                    # noqa: BLE001 - raised in the caller
            box["error"] = exc

    worker = threading.Thread(target=run, name="courant-mcp-call", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimedOut(seconds)
    if "error" in box:
        raise box["error"]
    return box["value"]


def build(roots: list[str] | None = None, timeout: float | None = TIMEOUT):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    from courant import __version__

    confine = tools.Confinement(roots or [])
    server = MCPServer(name="courant", instructions=INSTRUCTIONS, version=__version__)
    ro = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                         idempotent_hint=True, open_world_hint=False)

    from mcp.server.mcpserver.exceptions import ToolError

    def _guard(tool: str, call):
        """GATE-10: an expected refusal is reported in words; an unexpected exception is one
        line naming the tool and the version, never a traceback."""
        try:
            return within(timeout, call)
        except TimedOut as exc:
            raise ToolError(f"courant {tool}: no answer within {timeout:g} s, so the call was "
                            "abandoned (MCP-5). A case file that blocks its reader, or a very "
                            "large case, can do this.") from exc
        except tools.OutsideRoot as exc:
            raise ToolError(str(exc)) from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"courant {tool}: internal error ({type(exc).__name__}) in courant "
                            f"{__version__}. Run the same check with `courant verify --debug` "
                            "for the traceback.") from exc

    def verify_run(path: str, case_json: str | None = None, spec: str | None = None,
                   spec_sha256: str | None = None, schema: int = 2) -> dict:
        """Verify a completed simulation: a case directory or a solver log.

        Returns courant.verdict/2 (or courant.verdict/1 with schema=1): accept, reject or
        insufficient, the basis it was judged on, cited reasons, what was not checked, and a
        suggested next action that is never applied. Pass `spec`, the task specification
        (courant.spec/1) you were given, and `spec_sha256`, the hash it was locked at, to have
        every value it names checked against the case: a run that solved a different problem
        is rejected. A log still being written (no `End` yet and no fatal error) is
        insufficient (`insufficient.truncated_log`), not a crash: verify again once the solver
        has finished.
        """
        return _guard("verify_run", lambda: tools.verify_run(
            path, case_json, spec, spec_sha256, schema, confine=confine))

    server.add_tool(verify_run, annotations=ro)

    def score_agent_output(case: str, spec: str, spec_sha256: str | None = None,
                           reported_success: bool | None = None, schema: int = 2) -> dict:
        """Judge the case you finished against the task you were given, before you report.

        Advisory, never a certification: it reads your own log. `spec` is the task
        specification (courant.spec/1) and `spec_sha256` the hash it was locked at.
        `reported_success` is whether you are about to tell the user the run worked. Returns
        the verdict (courant.verdict/2, or /1 with schema=1), `task_class` (the outcome:
        valid_conformant, invalid, nonconformant, insufficient or did_not_execute),
        `silent_error` (it executed, you would report success, and it is rejected) and
        `unverified_success` (the same, with insufficient).
        """
        return _guard("score_agent_output", lambda: tools.score_agent_output(
            case, spec, spec_sha256, reported_success, schema, confine=confine))

    server.add_tool(score_agent_output, annotations=ro)

    def check_convergence(path: str, transient_average: bool = False) -> dict:
        """Where a run's monitored answer stopped changing, and what the rest of the run cost.

        Read-only. `path` is a case directory with function-object output under
        postProcessing/ (forces, forceCoeffs, surfaceFieldValue, probes), or a solver log.
        Set transient_average for a transient run whose answer is a time-average.
        """
        return _guard("check_convergence", lambda: tools.check_convergence(
            path, transient_average, confine=confine))

    server.add_tool(check_convergence, annotations=ro)
    server.add_tool(tools.explain_code, annotations=ro)
    server.add_tool(tools.list_failure_modes, annotations=ro)
    server.add_tool(tools.verifier_certification, annotations=ro)
    return server


def main() -> None:
    ap = argparse.ArgumentParser(prog="courant-mcp", description=__doc__)
    ap.add_argument("--root", action="append", default=[],
                    help="a directory the tools may read (repeatable; default: cwd, unless "
                         "that is /, the home directory or a directory that holds it)")
    ap.add_argument("--timeout", type=float, default=TIMEOUT,
                    help=f"seconds a tool call blocked on I/O or running Python code may run "
                         f"before it is answered with an error (default {TIMEOUT:g}; 0: no "
                         "limit); a call inside C code that holds the interpreter's lock is "
                         "answered when that code returns (MCP-5)")
    args = ap.parse_args()
    if (why := unconfined_default(args.root)) is not None:
        ap.error(why)
    build(args.root, timeout=args.timeout).run("stdio")


if __name__ == "__main__":
    main()
