# gate-repo

A fixture repository for the `courant-gate` Action. Its check fails on purpose.

- `runs/cavity_healthy`: a real, healthy icoFoam cavity run (accept).
- `runs/cavity_blowup`: the same case with a timestep too large; the Courant number reaches 9.66
  (reject, rule VER-5.3, first reason `cfl.max`).
- `courant.toml`: `fail-on = "not-accept"`, packs kept in `evidence/`.

`tests/unit/test_action.py` runs the gate on a copy of this directory and requires exit code 1
with the cited reason in the job summary. `.github/workflows/gate-action.yml` runs the Action on
it in CI and requires the check to fail.
