"""``courant update`` — replace this agent with the current release.

The API refuses agents below a minimum version with a 426, and the transport
tells the user to run this. A command that is advertised in an error message and
does not exist is worse than no command at all, so it exists.

Two installations, two different answers:

  * **packaged binary** — download the release for this platform, verify the
    release's signed checksums (OPS-7: the Ed25519 signature of ``SHA256SUMS``
    under the release key pinned in this build, ``courant.release``), and swap it
    in atomically. Same download and verification path as
    the installer, because it is the same problem.
  * **pip install** — do not touch it. Uninstalling a package from inside the
    process that imported it is a way to leave a machine with no working agent
    at all. Print the pip command and stop, having contacted nothing.

Its purpose is a network call, and it says so before it makes one: a packaged binary's update
downloads the release and its signed checksums from github.com (``RELEASES``), nothing else. It
contacts nothing when it is a pip installation or when ``--check`` is given (docs/NETWORK.md).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import typer
from rich.console import Console

from courant import DISTRIBUTION, RELEASE_REPO, __version__, release
from courant.release import SUMS, SUMS_SIG, ReleaseError, check_version, verify_asset

console = Console()

REPO = RELEASE_REPO
RELEASES = f"https://github.com/{REPO}/releases"
#: What this command contacts, said before it does (docs/NETWORK.md).
CONTACTS = "github.com"
PIP_UPGRADE = f"pip install --upgrade {DISTRIBUTION}"


def target_triple() -> str | None:
    """The release asset name for this machine, or None if we do not ship one."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    os_name = {"linux": "linux", "darwin": "darwin"}.get(system)
    arch = {"x86_64": "x86_64", "amd64": "x86_64",
            "arm64": "arm64", "aarch64": "arm64"}.get(machine)

    return f"{os_name}-{arch}" if os_name and arch else None


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def update(
    version: str = typer.Option(
        "latest", "--version", "-v", help="Version to install, e.g. 0.2.0."
    ),
    check: bool = typer.Option(
        False, "--check", help="Report what would happen; contact nothing, change nothing."
    ),
    allow_downgrade: bool = typer.Option(
        False, "--allow-downgrade",
        help="With --version: install a release older than this one."
    ),
) -> None:
    """Update this Courant binary to the current release, downloaded from github.com.

    The one visible command whose purpose is a network call: a packaged binary downloads the
    release and its signed checksums from github.com. A pip installation is managed by pip, and this
    prints the pip command and contacts nothing.
    """
    console.print(f"This agent: [cyan]{__version__}[/cyan]")

    if not is_packaged():
        # A pip-installed agent is managed by pip. Anything else we did here
        # would fight the user's environment.
        console.print(
            "\nThis is a [bold]pip[/bold] installation, so pip owns it:\n"
            f"  [cyan]{PIP_UPGRADE}[/cyan]\n"
            "  [dim]Nothing was contacted.[/dim]\n"
        )
        raise typer.Exit(0)

    triple = target_triple()
    if triple is None:
        console.print(
            f"\n[yellow]No published binary for "
            f"{platform.system()}/{platform.machine()}.[/yellow]\n"
            f"  [cyan]{PIP_UPGRADE}[/cyan]\n"
            "  [dim]Nothing was contacted.[/dim]\n"
        )
        raise typer.Exit(1)

    if version != "latest":
        try:
            check_version(version.lstrip("v"), version.lstrip("v"), __version__, allow_downgrade)
        except ReleaseError as exc:
            console.print(f"\n[red]Refusing:[/red] {exc}. [dim]Nothing was contacted.[/dim]\n")
            raise typer.Exit(1) from None

    base = (f"{RELEASES}/latest/download" if version == "latest"
            else f"{RELEASES}/download/v{version.lstrip('v')}")
    asset = f"courant-{triple}"

    if check:
        console.print(f"Would download [cyan]{base}/{asset}[/cyan] from {CONTACTS}. "
                      "[dim]Nothing was contacted.[/dim]")
        raise typer.Exit(0)

    destination = Path(sys.executable).resolve()
    console.print(f"  [dim]contacting {CONTACTS}: downloading {asset}, the release's "
                  f"{SUMS} and its signature from {RELEASES}[/dim]")

    try:
        payload = _download(f"{base}/{asset}")
        sums = _download_optional(f"{base}/{SUMS}")
        signature = _download_optional(f"{base}/{SUMS_SIG}")
    except Exception as exc:  # noqa: BLE001 — every failure here is the network
        console.print(
            f"\n[red]Could not download the update from {CONTACTS}:[/red] {exc}\n"
            f"  [dim]{RELEASES}[/dim]\n"
        )
        raise typer.Exit(1) from None

    # OPS-7: the signature is verified before anything is replaced, against the release key
    # pinned in this build, never the key the signature file names.
    if sums is None or signature is None:
        console.print(f"\n[red]The release has no signed {SUMS} — refusing to install.[/red]\n")
        raise typer.Exit(1)
    try:
        verify_asset(asset, payload, sums, json.loads(signature.decode()),
                     public_key=release.RELEASE_PUBLIC_KEY,
                     requested=version if version == "latest" else version.lstrip("v"),
                     current=__version__, allow_older=allow_downgrade)
    except (ReleaseError, ValueError) as exc:
        console.print(
            f"\n[red]The release signature does not check — refusing to install:[/red] {exc}\n"
            f"  [dim]Please report this: https://github.com/{REPO}/issues[/dim]\n"
        )
        raise typer.Exit(1) from None

    try:
        _replace(destination, payload)
    except PermissionError:
        console.print(
            f"\n[red]No permission to write[/red] {destination}\n"
            "  [dim]Installed system-wide? Re-run with the rights to replace it, "
            "or reinstall to your home directory:[/dim]\n"
            "  [cyan]curl -fsSL https://github.com/courantdev/courant/releases/latest/download/install.sh | sh[/cyan]\n"
        )
        raise typer.Exit(1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Could not replace the binary:[/red] {exc}\n")
        raise typer.Exit(1) from None

    console.print(f"\n[green]Updated.[/green] [dim]{destination}[/dim]")
    console.print("  [dim]courant version[/dim]")


# ---------------------------------------------------------------------------


def _download(url: str) -> bytes:
    import httpx

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _download_optional(url: str) -> bytes | None:
    try:
        return _download(url)
    except Exception:  # noqa: BLE001 — a missing checksum or signature is refused by the caller
        return None


def _replace(destination: Path, payload: bytes) -> None:
    """Swap the running binary for a new one, atomically.

    Written next to the target and renamed over it. Two reasons that ordering
    matters: ``rename`` within a filesystem is atomic, so an interrupted update
    cannot leave a truncated executable on PATH; and POSIX lets you replace the
    file of a *running* process, which is exactly what we are doing.
    """
    directory = destination.parent
    handle, staged = tempfile.mkstemp(prefix=".courant-update-", dir=directory)
    try:
        with os.fdopen(handle, "wb") as fh:
            fh.write(payload)
        os.chmod(staged, os.stat(destination).st_mode | stat.S_IXUSR)

        # Keep the old binary until the new one is proven to start. An update
        # that leaves a machine with no working agent is worse than a stale one.
        backup = destination.with_suffix(".old")
        shutil.copy2(destination, backup)
        os.replace(staged, destination)
        backup.unlink(missing_ok=True)
    except Exception:
        Path(staged).unlink(missing_ok=True)
        raise
