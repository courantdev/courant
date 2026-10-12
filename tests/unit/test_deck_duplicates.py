"""A key defined twice: the deck reader reads what OpenFOAM reads (review, 11 Oct 2026).

OpenFOAM's dictionary reads in `merge` mode by default (`#inputMode merge`): a repeated
primitive entry overwrites the earlier one, so the LAST definition is the value; a repeated
sub-dictionary is merged into the earlier one, so its entries are the union, the later winning
on each repeated key. Before this, the reader took the first definition: a case that wrote
`nu 1e-05;` and later `nu 2e-05;` ran with 2e-05 and was compared on 1e-05.

Where the file changes those rules (`#inputMode overwrite|protect|warn|error`, or the per-entry
`#default`, `#overwrite`, `#warn`, `#error`, `#remove`) and the key is repeated, the reader says
it cannot read the value rather than guess. A macro defined twice is the same: `$a` takes the
definition before its use, which this reader does not track.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courant.spec import Spec, SpecItem, check_conformance, deck

HEAD = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n}\n"


def _case(tmp_path: Path, rel: str, body: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD + body)
    return tmp_path


NU = SpecItem(name="nu", file="constant/transportProperties", key="nu", value=0.0)
MU = SpecItem(name="mu", file="constant/thermophysicalProperties",
              path=("mixture", "transport"), key="mu", value=0.0)


class TestLastDefinitionWins:
    def test_a_repeated_scalar_key(self, tmp_path):
        case = _case(tmp_path, NU.file, "nu [0 2 -1 0 0 0 0] 1e-05;\nnu [0 2 -1 0 0 0 0] 2e-05;\n")
        assert deck.read(case, NU).value == 2e-05

    def test_on_one_line_too(self, tmp_path):
        case = _case(tmp_path, NU.file, "nu 1e-05; a 1; nu 3e-05;\n")
        assert deck.read(case, NU).value == 3e-05

    def test_a_key_repeated_inside_a_sub_dictionary(self, tmp_path):
        case = _case(tmp_path, MU.file,
                     "mixture\n{\n    transport\n    {\n        mu 1.8e-05;\n        mu 2e-05;\n"
                     "    }\n}\n")
        assert deck.read(case, MU).value == 2e-05

    def test_a_repeated_sub_dictionary_is_merged(self, tmp_path):
        case = _case(tmp_path, MU.file,
                     "mixture\n{\n    transport { mu 1.8e-05; Pr 0.7; }\n}\n"
                     "mixture\n{\n    thermodynamics { Cp 1005; }\n}\n")
        assert deck.read(case, MU).value == 1.8e-05             # kept from the first
        cp = SpecItem(name="Cp", file=MU.file, path=("mixture", "thermodynamics"), key="Cp",
                      value=0.0)
        assert deck.read(case, cp).value == 1005.0                # added by the second
        case = _case(tmp_path, MU.file,
                     "mixture\n{\n    transport { mu 1.8e-05; }\n}\n"
                     "mixture\n{\n    transport { mu 2.5e-05; }\n}\n")
        assert deck.read(case, MU).value == 2.5e-05               # the later wins

    def test_a_key_repeated_inside_a_boundary_patch(self, tmp_path):
        item = SpecItem(name="U", file="0/U", patch="inlet", value=[0.0, 0, 0])
        case = _case(tmp_path, "0/U",
                     "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n"
                     "        value uniform (1 0 0);\n        value uniform (2 0 0);\n    }\n}\n")
        assert deck.read(case, item).value == [2.0, 0.0, 0.0]

    def test_a_repeated_patch_block_is_merged(self, tmp_path):
        item = SpecItem(name="U", file="0/U", patch="inlet", value=[0.0, 0, 0])
        case = _case(tmp_path, "0/U",
                     "boundaryField\n{\n    inlet { type fixedValue; value uniform (1 0 0); }\n"
                     "    outlet { type zeroGradient; }\n"
                     "    inlet { value uniform (4 0 0); }\n}\n")
        assert deck.read(case, item).value == [4.0, 0.0, 0.0]
        typ = SpecItem(name="t", file="0/U", patch="inlet", key="type", value="x")
        assert deck.read(case, typ).value == "fixedValue"

    def test_a_literal_patch_still_wins_over_a_later_regex_group(self, tmp_path):
        item = SpecItem(name="U", file="0/U", patch="inlet", value=[0.0, 0, 0])
        case = _case(tmp_path, "0/U",
                     "boundaryField\n{\n    inlet { type fixedValue; value uniform (1 0 0); }\n"
                     '    "(inlet|outlet)" { type fixedValue; value uniform (9 0 0); }\n}\n')
        assert deck.read(case, item).value == [1.0, 0.0, 0.0]

    def test_conformance_judges_the_value_the_solver_used(self, tmp_path):
        case = _case(tmp_path, NU.file, "nu 1e-05;\nnu 2e-05;\n")
        spec = Spec.model_validate({"schema": "courant.spec/1", "items": [
            {"name": "nu", "file": NU.file, "key": "nu", "value": 1e-05}]})
        assert check_conformance(spec, case, spec_sha256="x").items[0].conforms is False


class TestWhereTheRulesChange:
    @pytest.mark.parametrize("directive", ["#inputMode overwrite", "#inputMode protect",
                                           "#inputMode warn", "#inputMode error"])
    def test_a_non_default_input_mode_with_a_repeated_key_is_unreadable(self, tmp_path,
                                                                        directive):
        case = _case(tmp_path, NU.file, f"{directive}\nnu 1e-05;\nnu 2e-05;\n")
        found = deck.read(case, NU)
        assert found.value is None and "inputMode" in found.note

    def test_merge_mode_said_out_loud_is_the_default(self, tmp_path):
        case = _case(tmp_path, NU.file, "#inputMode merge\nnu 1e-05;\nnu 2e-05;\n")
        assert deck.read(case, NU).value == 2e-05

    @pytest.mark.parametrize("line", ["#default nu 5e-05;", "#overwrite nu 5e-05;",
                                      "#remove nu"])
    def test_a_per_entry_directive_on_the_key_is_unreadable(self, tmp_path, line):
        case = _case(tmp_path, NU.file, f"nu 1e-05;\n{line}\n")
        assert deck.read(case, NU).value is None

    def test_a_directive_elsewhere_with_no_repeat_is_harmless(self, tmp_path):
        case = _case(tmp_path, NU.file, "#inputMode protect\nnu 1e-05;\n")
        assert deck.read(case, NU).value == 1e-05

    def test_a_macro_defined_twice_is_unresolved(self, tmp_path):
        case = _case(tmp_path, NU.file, "a 1e-05;\nnu $a;\na 2e-05;\n")
        found = deck.read(case, NU)
        assert found.value is None and "could not be found" in found.note
