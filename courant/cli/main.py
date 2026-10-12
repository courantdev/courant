"""Courant CLI: verify engineering simulation results, on the machine that ran them.

The product:
  verify       a verdict on a completed run (accept, reject, insufficient)
  gate         the CI check: verify the runs a repository names (the GitHub Action)
  convergence  where the engineering answer stopped changing, and what came after
  preflight    a broken setup, caught before the solver spends anything
  audit        what a folder of runs cost, and how much of it bought nothing
  watch        a read-only advisory while the solver runs
  test         the Courant Test: score a judge, including an AI model
  demo         three real runs, offline, in seconds
  mcp          the same tools for an AI agent, over MCP

Every command is local and read-only, and none opens a network connection unless that is its
documented purpose and the user asked for it: `update` (github.com), `doctor --models`, and a
model judge or extractor the user names (the model door's provider). Each names what it
contacts before it does (docs/NETWORK.md). There is no account and no Courant service to sign
in to.
"""

from __future__ import annotations

import typer
from rich.console import Console

from courant.cli.commands import audit as audit_cmds
from courant.cli.commands import convergence as convergence_cmds
from courant.cli.commands import doctor as doctor_cmds
from courant.cli.commands import gate as gate_cmds
from courant.cli.commands import update as update_cmds
from courant.cli.commands import verify as verify_cmds
from courant.cli.commands import ctest as ctest_cmds
from courant.cli.commands import live as live_cmds
from courant.cli.commands import mcp as mcp_cmds
from courant.cli.commands import preflight as preflight_cmds
from courant.cli.commands import spec as spec_cmds
from courant.cli.commands import tour as tour_cmds


class _Guarded(typer.Typer):
    """GATE-10: an unexpected exception in any command ends in exit 4 and one line, never a
    traceback; `--debug` (before the command) or COURANT_DEBUG=1 prints it."""

    def __call__(self, *args, **kwargs):
        import sys

        from courant.cli import guard

        try:
            return super().__call__(*args, **kwargs)
        except Exception as exc:
            guard.report(sys.argv[1] if len(sys.argv) > 1 else "command", exc)
            raise SystemExit(4) from exc


app = _Guarded(
    pretty_exceptions_enable=False,
    add_completion=False,
    help=("Courant: verify engineering simulation results. Is this run trustworthy, did its "
          "answer converge, and if not, why. Local and read-only."),
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root(debug: bool = typer.Option(False, "--debug", help="Print the traceback of an internal "
                                     "error instead of one line.")) -> None:
    if debug:
        import os

        os.environ["COURANT_DEBUG"] = "1"


# --- the product ----------------------------------------------------------
app.command()(verify_cmds.verify)
app.command()(gate_cmds.gate)
app.command()(convergence_cmds.convergence)
app.command()(preflight_cmds.preflight)
app.command()(audit_cmds.audit)
app.command()(live_cmds.watch)
app.command()(tour_cmds.demo)
app.command()(mcp_cmds.mcp)
app.command()(doctor_cmds.doctor)
app.command()(update_cmds.update)
app.add_typer(ctest_cmds.app, name="test")
app.add_typer(spec_cmds.app, name="spec")


@app.command()
def version() -> None:
    """Print the Courant version."""
    import platform
    import sys

    from courant import __version__

    # From the module, not importlib.metadata: the packaged binary has no
    # distribution metadata to read, and reporting "(dev)" to a customer whose
    # agent the server just rejected as too old helps nobody.
    frozen = getattr(sys, "frozen", False)
    console.print(
        f"courant {__version__} "
        f"[dim]({platform.system().lower()}/{platform.machine()}"
        f"{', packaged' if frozen else ''})[/dim]"
    )


#: The extension point: when a package of this name is installed, its `register(app)` adds
#: commands. The published distribution ships none, and its `courant` is the gate alone.
EXTENSION = "courant.extensions"


def _extend(target: typer.Typer) -> None:
    import importlib

    try:
        module = importlib.import_module(EXTENSION)
    except ModuleNotFoundError as exc:
        if exc.name != EXTENSION:
            raise
        return
    module.register(target)


_extend(app)


if __name__ == "__main__":
    app()
