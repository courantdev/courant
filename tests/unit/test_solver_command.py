"""A production OpenFOAM run is never invoked bare.

It is `mpirun -np 64 <solver> -parallel`, or the scheduler's equivalent. `_base`
took the first token of that string while its docstring claimed to strip the MPI
wrapper, so it returned `mpirun` for every parallel command and the solver went
unrecognised on every industrial run:

  * `is_steady` returned False, silencing every steady-only check — the
    relaxation finding among them, and under-relaxation on steady cases is the
    failure mode a practising CFD consultant named first when asked what breaks;
  * `honours_adjust_time_step` returned False, which makes the timestep repair
    propose a fixed step on a solver family that does read `adjustTimeStep`.
"""

from __future__ import annotations

import pytest

from courant.case.solvers import _base, honours_adjust_time_step, is_steady


@pytest.mark.parametrize("command,expected", [
    ("icoFoam", "icoFoam"),
    ("/usr/bin/pimpleFoam", "pimpleFoam"),
    ("mpirun -np 4 icoFoam -parallel", "icoFoam"),
    ("mpirun --allow-run-as-root -np 8 rhoSimpleFoam -parallel", "rhoSimpleFoam"),
    ("mpiexec -n 16 interFoam -parallel", "interFoam"),
    ("srun -n 128 pimpleFoam -parallel", "pimpleFoam"),
    ("ibrun buoyantPimpleFoam -parallel", "buoyantPimpleFoam"),
    # A flag that takes a value, with a solver we do not have in the table.
    ("mpirun -np 4 -hostfile hosts myCustomFoam -parallel", "myCustomFoam"),
    # `-x VAR=value` passes an environment variable through the launcher.
    ("mpirun -np 2 -x FOAM_SETTINGS=1 rhoPimpleFoam -parallel", "rhoPimpleFoam"),
    ("", ""),
])
def test_the_solver_is_found_behind_its_launcher(command, expected) -> None:
    assert _base(command) == expected


def test_capability_survives_the_launcher() -> None:
    """The reason this matters, rather than the mechanism."""
    steady_parallel = "mpirun --allow-run-as-root -np 8 rhoSimpleFoam -parallel"
    assert is_steady(steady_parallel), "steady-only checks would be silenced"
    assert not honours_adjust_time_step(steady_parallel)

    adaptive_parallel = "srun -n 128 pimpleFoam -parallel"
    assert not is_steady(adaptive_parallel)
    assert honours_adjust_time_step(adaptive_parallel), (
        "the timestep repair would propose a fixed step on an adaptive solver"
    )


def test_a_known_solver_wins_over_position() -> None:
    """Exactness first: a recognised name anywhere beats any flag heuristic."""
    assert _base("mpirun -np 4 --mca btl self,tcp simpleFoam -parallel") == "simpleFoam"
