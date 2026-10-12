"""No traceback reaches a user (TRD GATE-10).

An unexpected exception in any command ends with exit 4 and one line that names the command and
the Courant version and points at `--debug`, which prints the traceback. An input the user got
wrong is not an unexpected exception: commands report those themselves (`InputError`).
"""

from __future__ import annotations

import functools
import os
import sys
import traceback

import typer

#: Set by `--debug` (or by the user) to print the traceback instead of the one line.
DEBUG_ENV = "COURANT_DEBUG"


def debugging() -> bool:
    return os.environ.get(DEBUG_ENV, "") not in ("", "0")


def internal_error(command: str, exc: BaseException) -> str:
    """The one line a user sees for an unexpected exception."""
    from courant import __version__

    return (f"courant {command}: internal error ({type(exc).__name__}) in courant {__version__}. "
            f"Re-run with --debug for the traceback.")


def report(command: str, exc: BaseException, debug: bool = False) -> typer.Exit:
    """Print the one line (or the traceback under --debug) and return the exit-4 to raise."""
    if debug or debugging():
        traceback.print_exception(exc, file=sys.stderr)
    else:
        print(internal_error(command, exc), file=sys.stderr)
    return typer.Exit(4)


def _command_endings() -> tuple[type[BaseException], ...]:
    """How a command ends on purpose: typer's Exit and Abort, and the usage errors of the click
    typer is built on (`typer.BadParameter` and its kin), whichever click that is. typer 0.27
    vendors its own (`typer._click`); earlier releases used `click`. Naming `click` here made an
    undeclared dependency (a plain `pip install` had no `click` and every command crashed on
    import) and, on typer 0.27, turned a usage error into an internal error."""
    usage = next(c for c in typer.BadParameter.__mro__ if c.__name__ == "ClickException")
    return (typer.Exit, typer.Abort, usage, SystemExit, KeyboardInterrupt)


def guarded(command: str):
    """Decorator for a typer command. A `debug` keyword (the command's `--debug`) turns the
    traceback on (so does COURANT_DEBUG=1); `typer.Exit` and click's own exceptions pass through
    untouched (`_command_endings`)."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except _command_endings():
                raise
            except Exception as exc:                         # the point of the guard
                raise report(command, exc, bool(kwargs.get("debug"))) from exc

        return inner

    return wrap
