"""Entry point for the frozen agent binary.

A separate file rather than pointing PyInstaller at ``courant/cli/main.py``: the
frozen binary needs a little setup that an installed package does not, and doing
it here keeps that out of the module people import.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Without this, any library that starts a process re-executes the frozen
    # binary as if it were a fresh `courant` invocation — the classic
    # PyInstaller fork bomb. Costs nothing when nothing forks.
    multiprocessing.freeze_support()

    from courant.cli.main import app

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
