"""Courant: physics acceptance testing for machine-produced simulation.

A read-only, deterministic gate between a simulation (an AI agent's or an engineer's) and the
person who signs it: `accept`, `reject` or `insufficient`, with the reasons it read. Without a
task specification it decides whether a run is valid; with one, whether it is the problem that
was asked. First integration: OpenFOAM.
"""

__version__ = "0.5.0"

#: The public repository of the open-source gate (H-DISTRIBUTION, 11 Oct 2026): its source, its
#: releases (the signed binaries, the wheel and the installer) and its GitHub Action.
#:
#: This is the one home for that fact. `courant update` reads it here and
#: `packaging/install.sh` is asserted against it by
#: `tests/unit/test_release_channel.py` — because the two of them once named
#: different repositories, and the one nobody had run was the one that was
#: wrong for every customer.
RELEASE_REPO = "courantdev/courant"

#: The distribution's name on the package index: ``pip install courant-dev``. PyPI's ``courant``
#: is an unrelated project, so every install line names this, never plain ``courant``
#: (tests/unit/test_network.py). The import package and the commands stay ``courant``.
DISTRIBUTION = "courant-dev"

__all__ = ["__version__", "RELEASE_REPO", "DISTRIBUTION"]
