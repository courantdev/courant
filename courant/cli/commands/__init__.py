"""CLI command implementations, one module per concern.

Kept out of ``main.py`` so the entry point stays a thin registration surface —
the packaged binary imports this lazily, and a broken command should not stop
``courant --help`` from working.
"""
