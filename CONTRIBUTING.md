# Contributing to Courant

Thank you for helping. Courant is the open-source gate: `courant verify`, `courant spec`,
`courant gate` and the GitHub Action, the MCP tools, and the checks they stand on.

## The rule every change keeps

Deterministic code decides every verdict, number and grade. A language model may draft prose,
and may extract a task specification that is shown, hashed and can only lower a verdict to
`insufficient`. Nothing else. A pull request that lets a model, a heuristic with no recorded
threshold, or the network into the verdict's path will not be merged.

## Working here

<!-- docs-skip: builds a virtual environment -->
```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/python -m ruff check --select F,E9 courant tests
.venv/bin/python -m pytest -q          # offline: no Docker, no network, no solver
```

- Add a test that fails without your change. Real solver output beats a synthetic log.
- Every number a document quotes needs its denominator.
- Read what the solver read: the case as the solver saw it, and OpenFOAM's own vocabulary.

By contributing you agree that your contribution is licensed under the Apache License 2.0
(see `LICENSE`), as section 5 of that licence says.
