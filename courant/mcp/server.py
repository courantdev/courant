"""courant-mcp: stdio MCP server (TRD MCP-1). Needs the `mcp` extra (mcp>=2.2).

    courant-mcp --root ~/cases --root /scratch/runs

Every tool is read-only and confined to the given roots (default: the
current directory).
"""

from __future__ import annotations

import argparse

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


def build(roots: list[str] | None = None):
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
            return call()
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
        is rejected.
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
                    help="a directory the tools may read (repeatable; default: cwd)")
    args = ap.parse_args()
    build(args.root).run("stdio")


if __name__ == "__main__":
    main()
