# PyInstaller spec for the Courant agent (Tech Bible §16, Phase B.4).
#
# What this produces: one self-contained executable that runs `courant watch` on
# a machine with no Python at all. That constraint is not incidental — the
# machines this has to run on are HPC login nodes where the system Python is
# 3.6, `pip install` needs a ticket, and the engineer who wants to try Courant
# has an hour, not a week.
#
# Build:
#     pip install -e ".[build]"
#     pyinstaller packaging/courant.spec --clean --noconfirm
#
# The excludes below are the whole trick. The CLI never imports FastAPI,
# SQLAlchemy, Alembic, or the Anthropic SDK — the cloud does — but PyInstaller's
# dependency analysis follows any import it can reach, including inside `if
# TYPE_CHECKING` blocks it cannot evaluate. Naming them explicitly is what keeps
# the agent binary from carrying an entire web server it will never start.
#
# `packaging/test_binary.sh` runs the built artefact through a real diverging
# case. A packaged binary that was never executed is not a build, it is a hope.

import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

BLOCK_CIPHER = None
ROOT = Path(SPECPATH).parent  # noqa: F821 — SPECPATH is injected by PyInstaller

# The open-source gate and nothing else (H-DISTRIBUTION): the packages pyproject.toml lists for
# the wheel, read from there so the binary and the wheel cannot carry different code. Every other
# package under courant/ is excluded by name, so nothing reaches it even through a lazy import.
PUBLIC = set(tomllib.loads((ROOT / "pyproject.toml").read_text())
             ["tool"]["setuptools"]["packages"]["find"]["include"])


def _package(module: str) -> str:
    """The package a module belongs to: itself when it is a package, else its parent."""
    path = ROOT.joinpath(*module.split("."))
    return module if (path / "__init__.py").is_file() else module.rsplit(".", 1)[0]


PRIVATE = sorted(".".join(d.relative_to(ROOT).parts)
                 for d in (ROOT / "courant").rglob("*")
                 if d.is_dir() and (d / "__init__.py").is_file()
                 and ".".join(d.relative_to(ROOT).parts) not in PUBLIC)

# Imported lazily, by string, or through Typer's decorators — all invisible to
# static analysis, and all fatal at runtime if missing.
HIDDEN = [
    # v0.3: the verifier imports most of itself lazily, inside the commands. Collect it whole,
    # the public packages only.
    *collect_submodules("courant", filter=lambda name: _package(name) in PUBLIC),
    "courant.cli.commands.doctor",
    "courant.cli.commands.update",
    # The taxonomy and KB are imported inside functions to keep CLI start-up fast.
    "courant.diagnosis.kb_data",
    "courant.taxonomy",
    # foamlib reaches for its parser backend at first use, not at import.
    *collect_submodules("foamlib"),
    # Rich resolves terminal themes by name.
    "pygments.formatters",
    "pygments.lexers",
]

# Server-only. See the note above about why these must be named.
EXCLUDED = [
    "alembic",
    "anthropic",
    "docker",
    "fastapi",
    "fluidfoam",
    "matplotlib",
    "psycopg",
    "pytest",
    "sqlalchemy",
    "starlette",
    "uvicorn",
    "websockets",
    # numpy pulls these in for interactive use; the agent does neither.
    "IPython",
    "tkinter",
    # The private repository's packages: never in the binary.
    *PRIVATE,
]

analysis = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entrypoint.py")],
    pathex=[str(ROOT)],
    binaries=[],
    # PKG-5, PKG-7: the certification summary, the demo runs and the Courant Test tasks, so
    # `courant demo` works on the machines where it matters most: HPC login nodes with no route
    # to the internet.
    datas=[(str(ROOT / "courant" / "data"), "courant/data"),
           (str(ROOT / "courant" / "ctest" / "data"), "courant/ctest/data")],
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=BLOCK_CIPHER)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    [],
    name="courant",
    debug=False,
    bootloader_ignore_signals=False,
    # Stripping breaks the codesign signature on macOS arm64, and the saving is
    # a few MB against a binary that is already tens.
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,      # host arch; cross-builds are done per-runner in CI
    codesign_identity=None,
    entitlements_file=None,
)
