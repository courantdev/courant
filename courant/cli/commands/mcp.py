"""`courant mcp`: the MCP server, from the same binary (TRD PKG-8).

    courant mcp --root ~/runs        # stdio; give this command to Claude Code, Cursor, ...

Equivalent to `courant-mcp`. Every tool is read-only and confined to the roots given.

    courant mcp --print-config --root ~/runs   # the client configuration, for Claude Code and Codex
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import typer


def client_config(roots: list[str]) -> str:
    """The configuration a client needs to start this server (GATE-6): Claude Code's command
    and `.mcp.json`, and Codex's `config.toml`. Generated from one argument list, so the three
    cannot disagree; a test parses the JSON and the TOML."""
    args = ["mcp", *(a for r in roots for a in ("--root", r))]
    shell = " ".join(shlex.quote(a) for a in ["claude", "mcp", "add", "courant", "--",
                                              "courant", *args])
    mcp_json = json.dumps({"mcpServers": {"courant": {"command": "courant", "args": args}}},
                          indent=2)
    toml_args = ", ".join(json.dumps(a) for a in args)
    from courant import DISTRIBUTION

    return (f"# Needs `courant` on PATH: pip install '{DISTRIBUTION}[mcp]' (PyPI's `courant` is "
            f"another project), or the single-file binary.\n\n"
            f"# Claude Code, from a shell:\n{shell}\n\n"
            f"# Claude Code or Cursor, in .mcp.json:\n{mcp_json}\n\n"
            f"# Codex, in ~/.codex/config.toml:\n[mcp_servers.courant]\n"
            f'command = "courant"\nargs = [{toml_args}]\n')


def mcp(
    root: list[str] = typer.Option(None, "--root", help="A directory the agent may read runs from. Repeatable. Default: the current directory."),
    print_config: bool = typer.Option(False, "--print-config", help="Print the client configuration (Claude Code, Cursor, Codex) for this server and exit."),
) -> None:
    """Serve Courant's read-only tools to an AI agent over MCP (stdio)."""
    if print_config:
        sys.stdout.write(client_config([str(Path(r).resolve())
                                        for r in (root or ["."])]))
        raise typer.Exit(0)
    import importlib.util

    if importlib.util.find_spec("mcp") is None:             # the optional extra is missing
        print("courant mcp needs the MCP extra: pip install 'courant-dev[mcp]' "
              "(this build was made without it)", file=sys.stderr)
        raise typer.Exit(4)
    from courant.mcp.server import build

    build(list(root) if root else ["."]).run("stdio")
