"""T-SPEC-3 review findings (10 Oct 2026), each a regression test.

1. SPEC-9's sign and dimension rules apply only to entries that hold the quantity: a heat flux on
   a temperature patch, a passive scalar T, an MRF or dynamic-mesh omega are not impossible.
2. A bound is never met on a per-face list's modal value: every face is checked.
3. A directive (`#calc`, `#eval`, `#codeStream`) is not evaluated, so its value is unreadable,
   never the last number in its text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courant.spec import Spec, SpecItem, check_conformance, deck
from courant.spec.impossible import impossibilities

HEAD = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n}\n"


def _spec(*items: dict) -> Spec:
    return Spec.model_validate({"schema": "courant.spec/1", "items": list(items)})


THERMO = {"name": "solver", "file": "system/controlDict", "key": "application",
          "value": "buoyantPimpleFoam"}
PASSIVE = {"name": "solver", "file": "system/controlDict", "key": "application",
           "value": "scalarTransportFoam"}


# ---------------------------------------------------------------- 1. only where it is the quantity
class TestQuantityScope:
    @pytest.mark.parametrize("key", ["q", "Q", "refGradient", "uniformGradient", "gradient"])
    def test_a_flux_or_gradient_on_a_temperature_patch_is_not_a_temperature(self, key):
        item = {"name": "wall", "file": "0/T", "patch": "wall", "key": key, "value": -1000.0}
        assert impossibilities(_spec(item, THERMO)) == ()

    def test_a_passive_scalar_t_of_zero_is_possible(self):
        item = {"name": "T at inlet", "file": "0/T", "patch": "inlet", "value": 0.0}
        assert impossibilities(_spec(item, PASSIVE)) == ()
        assert impossibilities(_spec(item)) == ()          # no solver named: not judged

    def test_a_thermodynamic_temperature_is_still_judged(self):
        item = {"name": "T at floor", "file": "0/T", "patch": "floor", "value": -5.0}
        assert impossibilities(_spec(item, THERMO))
        inlet = {"name": "T in", "file": "0/T", "patch": "inlet", "key": "inletValue",
                 "value": 0.0}
        assert impossibilities(_spec(inlet, THERMO))

    @pytest.mark.parametrize("file, path", [
        ("constant/MRFProperties", ("MRF1",)),
        ("constant/dynamicMeshDict", ("solidBodyMotionFunction",)),
        ("constant/dynamicMeshDict", ()),
    ])
    def test_an_angular_velocity_named_omega_is_not_turbulence(self, file, path):
        item = {"name": "rotation", "file": file, "path": list(path), "key": "omega",
                "value": -104.72}
        assert impossibilities(_spec(item)) == ()

    def test_a_key_outside_the_files_that_hold_its_quantity_is_not_judged(self):
        item = {"name": "nu in a function object", "file": "system/controlDict",
                "path": ["functions", "x"], "key": "nu", "value": -1.0}
        assert impossibilities(_spec(item)) == ()
        held = {"name": "nu", "file": "constant/physicalProperties", "key": "nu", "value": -1.0}
        assert impossibilities(_spec(held))
        phase = {"name": "rho", "file": "constant/thermophysicalProperties.water",
                 "path": ["mixture", "equationOfState"], "key": "rho", "value": 0.0}
        assert impossibilities(_spec(phase))

    def test_the_turbulence_fields_are_still_judged(self):
        k = {"name": "k in", "file": "0/k", "patch": "inlet", "value": -1.0}
        omega = {"name": "omega in", "file": "0/omega", "patch": "inlet", "value": -1.0}
        assert impossibilities(_spec(k)) and impossibilities(_spec(omega))


# ---------------------------------------------------------------- 2. bounds on per-face lists
def _field(root: Path, faces: str) -> Path:
    p = root / "0" / "U"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
                 "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n"
                 f"        value {faces};\n    }}\n}}\n")
    return root


class TestBoundsOnFaces:
    LIST = "nonuniform List<vector> 10(" + " ".join(["(0 0 0)"] * 6 + ["(50 0 0)"] * 4) + ")"

    def test_a_bound_is_checked_on_every_face_not_the_mode(self, tmp_path):
        case = _field(tmp_path, self.LIST)
        block = check_conformance(_spec({"name": "U", "file": "0/U", "patch": "inlet",
                                         "max": 10.0}), case, spec_sha256="x")
        assert block.items[0].conforms is False
        assert "10 face values" in (block.items[0].basis or "")

    def test_every_face_inside_the_bound_conforms(self, tmp_path):
        case = _field(tmp_path, self.LIST)
        block = check_conformance(_spec({"name": "U", "file": "0/U", "patch": "inlet",
                                         "max": 60.0}), case, spec_sha256="x")
        assert block.items[0].conforms is True

    def test_a_scalar_list_too(self, tmp_path):
        p = tmp_path / "0" / "T"
        p.parent.mkdir(parents=True)
        p.write_text(HEAD + "boundaryField\n{\n    floor { type fixedValue; value nonuniform "
                     "List<scalar> 5(300 300 300 300 900); }\n}\n")
        item = {"name": "T", "file": "0/T", "patch": "floor"}
        low = check_conformance(_spec({**item, "max": 400.0}), tmp_path, spec_sha256="x")
        high = check_conformance(_spec({**item, "min": 250.0, "max": 1000.0}), tmp_path,
                                 spec_sha256="x")
        assert low.items[0].conforms is False and high.items[0].conforms is True

    def test_a_value_item_still_reads_the_mode(self, tmp_path):
        case = _field(tmp_path, self.LIST)
        block = check_conformance(_spec({"name": "U", "file": "0/U", "patch": "inlet",
                                         "value": [0, 0, 0]}), case, spec_sha256="x")
        assert block.items[0].conforms is True


# ---------------------------------------------------------------- 3. directives are not evaluated
class TestDirectives:
    @pytest.mark.parametrize("line", [
        'a 0.1;\nnu #calc "${a}*2";',
        'a 0.1;\nnu #calc "$a*2";',
        'nu #eval "0.1*2";',
        'nu #eval{ 0.1*2 };',
        'a #calc "0.1*2";\nnu $a;',
    ])
    def test_a_directive_value_is_unreadable(self, tmp_path, line):
        p = tmp_path / "constant" / "transportProperties"
        p.parent.mkdir(parents=True)
        p.write_text(HEAD + line + "\n")
        for value in (2.0, 0.2):
            block = check_conformance(_spec({"name": "nu", "file": "constant/transportProperties",
                                             "key": "nu", "value": value}), tmp_path,
                                      spec_sha256="x")
            assert block.items[0].conforms is None, (line, value, block.items[0])

    def test_a_boundary_value_from_a_directive_is_unreadable(self, tmp_path):
        p = tmp_path / "0" / "U"
        p.parent.mkdir(parents=True)
        p.write_text(HEAD + 'boundaryField\n{\n    inlet { type fixedValue; value uniform '
                     '#calc "vector(1, 0, 0)*2"; }\n}\n')
        found = deck.read(tmp_path, SpecItem(name="U", file="0/U", patch="inlet",
                                             value=[2.0, 0, 0]))
        assert found.value is None and "directive" in found.note
