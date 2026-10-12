"""`courant doctor`: what this installation is, and what it can reach for.

It contacts nothing, with one exception the user asks for by name: `--models` reads
OpenRouter's public model listing (no key) and names the host before it does
(docs/NETWORK.md). The previous product's cloud account commands (`login`, `logout`,
`whoami`) and doctor's probe of its API are gone: there is no Courant service to reach.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from courant import __version__
from courant.cli.config import config_dir

console = Console()


def doctor(
    models: bool = typer.Option(
        False, "--models",
        help="list the models the model door may call, and which of them OpenRouter's "
             "live listing has dropped, and exit. This reads OpenRouter's public model listing "
             "over the network, without a key; nothing else in doctor uses the network"),
    bundle: str = typer.Option(
        None, "--bundle",
        help="write a support bundle (courant.support/1: versions, platform, the command "
             "line, and a verdict if --verdict names one; no case data) to this file, and exit"),
    verdict: str = typer.Option(
        None, "--verdict", help="with --bundle: a verdict JSON file to include"),
    pseudonymise: bool = typer.Option(
        False, "--pseudonymise",
        help="with --bundle: replace the run id, patch names, file names and item names with "
             "stable pseudonyms and strip paths and hosts, so the bundle can be shared"),
) -> None:
    """Diagnose this installation, the first thing to run when something is off. Offline;
    only --models contacts OpenRouter."""
    if bundle:
        import sys
        from pathlib import Path

        from courant.verify import support
        from courant.verify.schema import canonical_json

        try:
            out = Path(bundle)
            out.write_text(canonical_json(support.build(sys.argv, verdict, pseudonymise=pseudonymise)))
        except (ValueError, OSError) as exc:
            console.print(f"courant doctor: {exc}", style="red", markup=False)
            raise typer.Exit(4) from exc
        console.print(f"support bundle written: {out} (nothing was sent)", markup=False)
        return
    if models:
        _show_models()
        return

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))

    def check(label: str, ok: bool | None, detail: str = "") -> None:
        mark = "[green]✓[/green]" if ok else ("[yellow]—[/yellow]" if ok is None
                                              else "[red]✗[/red]")
        # Escaped: a detail such as "pip install 'courant-dev[sandbox]'" is text, not markup.
        table.add_row(mark, label, f"[dim]{escape(detail)}[/dim]")

    check("version", True, __version__)
    check("config dir", config_dir().exists(), str(config_dir()))
    # Said, because "does it phone home" is the question a security review asks first.
    check("network", True, "doctor contacted nothing; which commands do: docs/NETWORK.md")

    # The reasoning model, as the model door decides it. On a customer laptop
    # this is normally none, and that is correct: the deterministic diagnoser is
    # the floor. It is shown because "why is it not using the LLM" is otherwise
    # an invisible question, and a key alone is not the answer: the door also
    # needs a zero-data-retention model measured on diagnosis (ARC-9, ARC-12).
    from courant.config import Settings
    from courant.diagnosis.llm import DIAGNOSIS_CALL
    from courant.llm import default_door

    local = Settings()
    door = default_door()
    chosen = door.eligible(**DIAGNOSIS_CALL)
    if chosen:
        check("reasoning model", True, f"{chosen[0].provider} · {chosen[0].id}")
    elif local.providers:
        check("reasoning model", None,
              f"{' · '.join(local.providers)} key set; diagnosis stays deterministic: "
              f"{door.why_not(**DIAGNOSIS_CALL)}")
    else:
        check("reasoning model", None,
              "no model provider configured; diagnosis is deterministic")

    # Optional, and only interesting to someone running real solves locally —
    # so a dash rather than a cross when it is absent, and the *reason* when it
    # is, because "no docker" on a machine where `docker run` works is a
    # missing Python package, not a broken daemon. This asks the local Docker
    # daemon (its socket, or DOCKER_HOST if the user set one) whether it runs.
    if local.docker_ready():
        check("docker (real solves)", True, local.openfoam_image)
    else:
        check("docker (real solves)", None, local.docker_detail)

    console.print(table)


def _show_models() -> None:
    """`courant doctor --models`: the models the model door may call, and what OpenRouter's
    live listing says of them today.

    The allowlist is in the repository (courant/llm/allowlist.py), not discovered: free is
    decided by the list and the exact `:free` suffix, never by a price that reads "0" (ARC-6).
    Free ids are renamed and retired on OpenRouter's schedule, so the live listing, read
    without a key, marks every listed id it no longer serves or that is past its expiration
    date (ARC-10). A dropped id is a reason to refresh the allowlist, not to pick another
    model here.
    """
    from datetime import UTC, datetime

    from courant.llm import LISTING_HOST, LLMError, public_listing
    from courant.llm.allowlist import ALLOWLIST, READ_ON
    from courant.llm.policy import refresh

    dropped: dict[str, str] = {}
    live = False
    console.print(f"[dim]contacting {LISTING_HOST}: OpenRouter's public model listing, "
                  "without a key[/dim]\n")
    try:
        listing = public_listing()["models"]
        live = True
    except LLMError as exc:
        console.print(f"[yellow]Could not read OpenRouter's live listing[/yellow] "
                      f"({exc.message}); showing the committed allowlist only.\n")
    if live:
        today = datetime.now(UTC).date()
        _, gone = refresh(ALLOWLIST.get("openrouter", []), listing, today)
        dropped = {entry["id"]: reason for entry, reason in gone}

    table = Table(box=None, padding=(0, 2, 0, 0))
    for column in ("provider", "model", "capabilities", "zdr", "measured", "expires", "today"):
        # The id is what a refresh or a pin needs verbatim, so it never wraps or truncates.
        table.add_column(column, style="cyan" if column == "model" else None,
                         no_wrap=column == "model")
    for provider, entries in ALLOWLIST.items():
        for entry in entries:
            measured = ", ".join(f"{task}: {run}" for task, run in entry["measured"].items())
            status = (f"[red]{dropped[entry['id']]}[/red]" if entry["id"] in dropped
                      else ("listed" if live and provider == "openrouter" else "—"))
            table.add_row(provider, entry["id"], ", ".join(entry["capabilities"]),
                          {True: "yes", False: "no", None: "unknown"}[entry["zdr"]],
                          measured or "not yet (ARC-12)", entry["expires"] or "—", status)

    console.print(f"[bold]The model door's allowlist[/bold], read {READ_ON} "
                  f"(courant/llm/allowlist.py)\n")
    console.print(table)
    console.print("\nCustomer data goes only to zdr: yes (ARC-9). A model diagnoses only once "
                  "it is measured on diagnosis (ARC-12). Paid models need S's "
                  "COURANT_ALLOW_PAID_LLM=1 and a caller that pins them.")
    if dropped:
        console.print("[yellow]Refresh the allowlist:[/yellow] "
                      "python scripts/llm_allowlist.py refresh")
