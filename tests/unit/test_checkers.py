"""Unit tests for the deterministic checkers (Layer 1 of diagnosis)."""

import pytest

from courant.case.config import CaseConfig
from courant.case.mesh import MeshQuality
from courant.diagnosis.checkers import (
    BoundaryConfigChecker,
    BoundingChecker,
    CheckContext,
    CourantChecker,
    CrashSignatureChecker,
    MeshChecker,
    PhysicsChecker,
    ResidualAnalyzer,
    SchemeToleranceChecker,
    run_all_checkers,
)
from courant.diagnosis.findings import FindingSeverity
from courant.events import (
    CrashPayload,
    DivergencePayload,
    Event,
    EventKind,
    ResidualPayload,
    Severity,
    StallPayload,
    WarnPayload,
)


def residual(seq, time, res, courant=None):
    return Event(run_id="r", seq=seq, kind=EventKind.RESIDUAL,
                 payload=ResidualPayload(time=time, initial_residuals=res,
                                         courant_max=courant))


def ctx(events=None, **kw):
    return CheckContext(run_id="r", events=events or [], **kw)


class TestCourantChecker:
    def test_flags_high_and_climbing_courant(self):
        events = [residual(0, 0.001, {"p": 0.1}, courant=0.4),
                  residual(1, 0.002, {"p": 0.2}, courant=87.3)]
        findings = CourantChecker().run(ctx(events, config=CaseConfig(adjust_timestep=False)))
        ids = {f.id for f in findings}
        assert "cfl.max" in ids and "cfl.climbing" in ids
        cfl = next(f for f in findings if f.id == "cfl.max")
        assert cfl.value == 87.3 and cfl.severity is FindingSeverity.CRITICAL
        assert "NUM.CFL_VIOLATION" in cfl.suggests

    def test_silent_when_courant_low(self):
        events = [residual(0, 0.001, {"p": 0.1}, courant=0.4)]
        assert CourantChecker().run(ctx(events)) == []


class TestResidualAnalyzer:
    def test_nan_divergence(self):
        div = Event(run_id="r", seq=5, kind=EventKind.DIVERGENCE, severity=Severity.ERROR,
                    payload=DivergencePayload(field="p", is_nan=True))
        findings = ResidualAnalyzer().run(ctx([div]))
        assert any(f.id == "residual.nan" and f.severity is FindingSeverity.CRITICAL
                   for f in findings)

    def test_plateau_stall(self):
        stall = Event(run_id="r", seq=6, kind=EventKind.STALL,
                      payload=StallPayload(reason="residual_plateau", plateau_iterations=50))
        findings = ResidualAnalyzer().run(ctx([stall]))
        assert any(f.id == "residual.plateau" and "CONV.RESIDUAL_PLATEAU" in f.suggests
                   for f in findings)

    def test_silent_non_convergence(self):
        events = [residual(0, 1, {"p": 0.05, "Ux": 0.02})]  # high final residual, no divergence
        findings = ResidualAnalyzer().run(ctx(events))
        assert any(f.id == "residual.not_converged" for f in findings)


class TestBoundingChecker:
    def test_turbulence_bounding_suggests_turb_inlet(self):
        events = [Event(run_id="r", seq=i, kind=EventKind.WARN, severity=Severity.WARN,
                        payload=WarnPayload(category="bounding", field="k"))
                  for i in range(4)]
        findings = BoundingChecker().run(ctx(events))
        k = next(f for f in findings if f.id == "bounding.k")
        assert k.value == 4 and "BC.TURB_INLET" in k.suggests


class TestCrashSignatureChecker:
    def test_missing_file_signature(self):
        crash = Event(run_id="r", seq=0, kind=EventKind.CRASH, severity=Severity.ERROR,
                      payload=CrashPayload(foam_fatal=True, tag="foam_fatal"))
        log = "--> FOAM FATAL IO ERROR: cannot find file constant/transportProperties"
        findings = CrashSignatureChecker().run(ctx([crash], log_text=log))
        assert any("ENV.MISSING_DICT" in f.suggests for f in findings)

    def test_generic_crash_when_no_signature(self):
        crash = Event(run_id="r", seq=0, kind=EventKind.CRASH, severity=Severity.ERROR,
                      payload=CrashPayload(tag="nonzero_exit"))
        findings = CrashSignatureChecker().run(ctx([crash], log_text="something went wrong"))
        assert any(f.id == "crash.fatal" for f in findings)


class TestStaticCheckers:
    def test_mesh_negative_volume(self):
        m = MeshQuality(n_negative_volume_cells=3)
        findings = MeshChecker().run(ctx(mesh=m))
        assert any(f.id == "mesh.negative_volume" and "MESH.NEGATIVE_VOLUME" in f.suggests
                   for f in findings)

    def test_mesh_non_orthogonality(self):
        m = MeshQuality(max_non_orthogonality=82.0)
        findings = MeshChecker().run(ctx(mesh=m))
        assert any(f.id == "mesh.non_orthogonality" for f in findings)

    def test_scheme_over_relaxation(self):
        c = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                       relaxation_factors={"p": 1.0, "U": 1.0})
        findings = SchemeToleranceChecker().run(ctx(config=c))
        f = next(f for f in findings if f.id == "scheme.relaxation")
        assert "NUM.OVER_RELAXATION" in f.suggests

    def test_bc_overconstrained(self):
        c = CaseConfig(boundary={"inlet": {"p": "fixedValue", "U": "fixedValue"}})
        findings = BoundaryConfigChecker().run(ctx(config=c))
        assert any("BC.OVERCONSTRAINED_PU" in f.suggests for f in findings)

    def test_physics_bad_nu(self):
        c = CaseConfig(transport={"nu": 5.0})  # absurd viscosity
        findings = PhysicsChecker().run(ctx(config=c))
        assert any("PHYS.TRANSPORT_UNITS" in f.suggests for f in findings)


class TestRunAll:
    def test_aggregates_unique_findings(self):
        events = [residual(0, 0.001, {"p": 0.1}, courant=0.4),
                  residual(1, 0.002, {"p": 0.2}, courant=87.3)]
        c = CaseConfig(adjust_timestep=False, relaxation_factors={"p": 0.95})
        fs = run_all_checkers(ctx(events, config=c))
        assert len(fs) == len(fs.ids)  # unique
        assert "NUM.CFL_VIOLATION" in fs.suggested_codes()


class TestBoundedSchemeVocabulary:
    """Our idea of a bounded scheme has to match OpenFOAM's.

    The check was `"bounded" not in scheme and "upwind" not in scheme`, which is
    wrong on the single most common turbulence-convection scheme there is:
    `div(phi,k) Gauss limitedLinear 1` contains neither word, and `limitedLinear`
    is *definitionally* limited. Both of OpenFOAM's own turbulent tutorials use
    it, so `scheme.unbounded_div` fired on every real RAS case — and because that
    finding suggests NUM.UNBOUNDED_SCHEME, it outranked the true cause. A CFL
    blow-up on pisoFoam/RAS/cavity was diagnosed as a scheme problem.

    The vocabulary is read from
    `src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes` in
    v2406, not recalled.
    """

    @pytest.mark.parametrize("scheme", [
        "Gauss limitedLinear 1",                        # pisoFoam/RAS/cavity
        "bounded Gauss limitedLinear 1",                # simpleFoam/pitzDaily
        "bounded Gauss linearUpwind grad(U)",
        "Gauss linearUpwind grad(U)",
        "Gauss CoBlended 1 upwind linearUpwind grad(U)",
        "Gauss vanLeer",
        "Gauss limitedCubic 1",
        "Gauss SuperBee",
        "Gauss Minmod",
    ])
    def test_real_bounded_schemes_are_recognised(self, scheme):
        from courant.case.schemes import is_bounded_scheme

        assert is_bounded_scheme(scheme) is True

    @pytest.mark.parametrize("scheme", [
        "Gauss linear",       # pure central differencing — genuinely unbounded
        "Gauss cubic",
        "Gauss midPoint",
    ])
    def test_genuinely_unbounded_schemes_still_flagged(self, scheme):
        """The fix must not blind the checker to the thing it exists for."""
        from courant.case.schemes import is_bounded_scheme

        assert is_bounded_scheme(scheme) is False

    def test_matching_is_word_wise_not_substring(self):
        """`linear` is a substring of `limitedLinear`. A naive `in` test would
        call the genuinely unbounded `Gauss linear` bounded, or miss the real
        one — depending which way round it was written."""
        from courant.case.schemes import is_bounded_scheme

        assert is_bounded_scheme("Gauss linear") is False
        assert is_bounded_scheme("Gauss limitedLinear 1") is True

    @pytest.mark.parametrize("term,is_convection", [
        ("div(phi,U)", True),
        ("div(phi,k)", True),
        ("div(phi,epsilon)", True),
        # pisoFoam/RAS/cavity also transports these, and an unbounded scheme on
        # them is just as real — the old check hardcoded three terms.
        ("div(phi,omega)", True),
        ("div(phi,R)", True),
        ("div(phi,nuTilda)", True),
        # Not convection of a transported field; `Gauss linear` is correct here
        # and flagging it would be a false positive on every case.
        ("div(R)", False),
        ("div(U)", False),
        ("div((nuEff*dev2(T(grad(U)))))", False),
        ("div(nonlinearStress)", False),
    ])
    def test_only_convection_terms_are_examined(self, term, is_convection):
        from courant.case.schemes import is_convection_term

        assert is_convection_term(term) is is_convection

class TestReverseFlowIsNotACrash:
    """The most expensive bug class this product has: a false CRITICAL.

    `pisoFoam/RAS/cavity` — a healthy OpenFOAM tutorial — contains

        setFlow1:
            operating mode: function
            reverse flow direction at time: 1

    which is the `setFlow` function object announcing its own configuration.
    Matching "reverse flow" as a fatal signature produced `crash.reversed_outflow`
    at CRITICAL on a perfectly good run — and because it is critical it then
    outranked the true cause on the broken run, so a CFL blow-up was diagnosed
    as a boundary-condition problem.
    """

    @staticmethod
    def _findings(log: str, *, crashed: bool):
        from courant.diagnosis.checkers import CrashSignatureChecker
        from courant.events import CrashPayload, Event, EventKind, Severity

        events = []
        if crashed:
            events.append(Event(run_id="r", seq=0, kind=EventKind.CRASH,
                                severity=Severity.ERROR,
                                payload=CrashPayload(tag="fatal_signal")))
        ctx = CheckContext(run_id="r", events=events, config=None, mesh=None,
                           log_text=log)
        return CrashSignatureChecker().run(ctx)

    def test_a_function_object_banner_is_not_a_finding(self):
        log = ("setFlow1:\n    operating mode: function\n"
               "    reverse flow direction at time: 1\n")
        assert self._findings(log, crashed=False) == []

    def test_persistent_reverse_flow_is_reported_as_a_warning(self):
        """The capability is kept — genuine recurring reverse flow at an outlet
        is a real taxonomy entry — but it is a warning, not a crash."""
        from courant.diagnosis.findings import FindingSeverity

        log = "\n".join("--> FOAM Warning : reverse flow at outlet" for _ in range(8))
        findings = self._findings(log, crashed=False)

        assert len(findings) == 1
        assert findings[0].id == "flow.reversed_outflow"
        assert findings[0].severity is FindingSeverity.WARN
        assert "BC.REVERSED_OUTFLOW" in findings[0].suggests

    def test_fatal_signatures_are_only_consulted_when_it_crashed(self):
        """They name *why it crashed*. Scanning unconditionally means a healthy
        run can be handed a CRITICAL crash finding."""
        log = 'cannot find file "0/p"\n'

        assert self._findings(log, crashed=False) == []

        crashed = self._findings(log, crashed=True)
        assert any(f.id == "crash.missing_dict" for f in crashed)

    def test_a_crash_with_no_signature_still_reports_one(self):
        """Losing the generic fallback would be worse than the false positive."""
        findings = self._findings("Time = 1\n", crashed=True)
        assert [f.id for f in findings] == ["crash.fatal"]


def bounding_warn(seq, field):
    return Event(run_id="r", seq=seq, kind=EventKind.WARN,
                 payload=WarnPayload(category="bounding", field=field, count=1))


def nan_divergence(seq, field):
    return Event(run_id="r", seq=seq, kind=EventKind.DIVERGENCE,
                 payload=DivergencePayload(field=field, is_nan=True))


class TestTheSchemeVocabularyHasOneImplementation:
    """The check that would have caught the half-applied fix.

    "Is this scheme bounded?" was answered in two places: the checker and the
    knowledge base's `unbounded_div` config predicate. When the checker's copy
    was corrected to read `limitedLinear` as bounded, the KB's was not — so the
    KB kept adding 2.0 to NUM.UNBOUNDED_SCHEME on every RAS case, and that code
    appeared in the wrong answer of eight of the twelve scenarios that failed
    the first full certification matrix.

    Neither copy was untested. What was untested was that they *agree*.
    """

    #: Real `divSchemes` entries from OpenFOAM v2406's own tutorials, plus the
    #: genuinely unbounded ones the finding exists to catch.
    SCHEMES = [
        "Gauss limitedLinear 1",
        "bounded Gauss limitedLinear 1",
        "Gauss linearUpwind grad(U)",
        "Gauss CoBlended 0.01 limitedLinearV 1 0.05 linearUpwind grad(U)",
        "Gauss vanLeer",
        "Gauss upwind",
        "Gauss linear",
        "Gauss cubic",
        "Gauss midPoint",
    ]

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_the_kb_predicate_and_the_checker_agree(self, scheme):
        from courant.case.schemes import is_bounded_scheme
        from courant.diagnosis.kb import CONFIG_PREDICATES

        config = CaseConfig(solver="pisoFoam", div_schemes={"div(phi,U)": scheme})

        kb_says_unbounded = CONFIG_PREDICATES["unbounded_div"](config)
        assert kb_says_unbounded is not is_bounded_scheme(scheme)

    def test_the_kb_predicate_ignores_non_convection_terms(self):
        """`div((nuEff*dev2(T(grad(U)))))` is a diffusion term. `Gauss linear`
        is correct there and every real case has one, so a predicate that looked
        at it would fire on the entire population."""
        from courant.diagnosis.kb import CONFIG_PREDICATES

        config = CaseConfig(
            solver="simpleFoam",
            div_schemes={"div(phi,U)": "bounded Gauss limitedLinear 1",
                         "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"})

        assert CONFIG_PREDICATES["unbounded_div"](config) is False


class TestUnboundedSchemeNeedsASymptom:
    """An unbounded scheme is a predisposition, not a fault.

    `div(phi,U) Gauss linear` is what the OpenFOAM project itself ships for
    `icoFoam/cavity`, and it is right: pure central differencing on a laminar
    Re=10 lid-driven cavity. Reported unconditionally the finding scored
    NUM.UNBOUNDED_SCHEME at 6.0 on every case that had one, which displaced the
    true cause on a Courant blow-up, three mesh faults and a wrong turbulence
    model in a single certification run.
    """

    UNBOUNDED = CaseConfig(solver="icoFoam", delta_t=0.005,
                           div_schemes={"div(phi,U)": "Gauss linear"})

    def test_silent_on_a_clean_run(self):
        """The exact configuration of OpenFOAM's own cavity tutorial, solving
        happily. Saying anything here is a false positive on correct input."""
        findings = SchemeToleranceChecker().run(
            ctx(events=[residual(1, 0.1, {"Ux": 1e-6})], config=self.UNBOUNDED))

        assert not [f for f in findings if f.id == "scheme.unbounded_div"]

    def test_bounding_of_a_transported_field_is_evidence(self):
        findings = SchemeToleranceChecker().run(
            ctx(events=[bounding_warn(1, "k"), bounding_warn(2, "epsilon")],
                config=self.UNBOUNDED))

        found = [f for f in findings if f.id == "scheme.unbounded_div"]
        assert found and "epsilon, k" in found[0].detail

    def test_a_non_finite_residual_is_not_evidence(self):
        """Tried, and too weak. *Every* blow-up ends with a NaN somewhere, so
        this re-admitted the finding on three mesh faults and a bad initial
        field — cases where the scheme is blameless. Being the last thing to
        break is not being the cause."""
        findings = SchemeToleranceChecker().run(
            ctx(events=[nan_divergence(1, "k")], config=self.UNBOUNDED))

        assert not [f for f in findings if f.id == "scheme.unbounded_div"]

    def test_a_courant_runaway_is_not_evidence(self):
        """The ranking failure in one test. A climbing Courant number is a
        timestep problem at any discretisation, so counting it as scheme
        evidence is how NUM.UNBOUNDED_SCHEME came to outrank NUM.CFL_VIOLATION
        on a case whose fault was literally an oversized timestep."""
        events = [residual(i, i * 0.005, {"Ux": 10.0 ** i}, courant=2.0 ** i)
                  for i in range(1, 8)]

        findings = SchemeToleranceChecker().run(
            ctx(events=events, config=self.UNBOUNDED))

        assert not [f for f in findings if f.id == "scheme.unbounded_div"]

    def test_a_bounded_case_says_nothing_even_with_symptoms(self):
        """The gate narrows the finding; it must not widen it."""
        bounded = CaseConfig(solver="pisoFoam",
                             div_schemes={"div(phi,k)": "Gauss limitedLinear 1"})

        findings = SchemeToleranceChecker().run(
            ctx(events=[bounding_warn(1, "k")], config=bounded))

        assert not [f for f in findings if f.id == "scheme.unbounded_div"]


class TestUnreadableIsNotAbsent:
    """A case we could not parse and a case that omitted the entry are different
    facts, and they used to be the same one.

    Every config checker begins `if c is None: return []` or reads a field that
    is `None`, so a `controlDict` that failed to parse produced *silence* —
    identical to a healthy case. The engineer then gets a diagnosis built on a
    configuration nobody ever read, with nothing anywhere saying so.

    Reachable on real cases: `foamDictionary -set` rewrites a dictionary with
    its `#include` directives expanded, and OpenFOAM's own `pisoFoam/RAS/cavity`
    expands into `fields ( U { mean on; ... } );`, which foamlib cannot parse.
    """

    def test_it_reports_the_file_and_the_reason(self):
        from courant.diagnosis.checkers import CaseIntegrityChecker

        config = CaseConfig(unreadable={"system/controlDict":
                                        "FoamFileDecodeError: line 48"})

        found = CaseIntegrityChecker().run(ctx(config=config))

        assert len(found) == 1
        assert found[0].id == "case.unreadable_dict"
        assert "system/controlDict" in found[0].detail
        assert "line 48" in found[0].detail

    def test_a_readable_case_says_nothing(self):
        from courant.diagnosis.checkers import CaseIntegrityChecker

        assert CaseIntegrityChecker().run(
            ctx(config=CaseConfig(solver="icoFoam"))) == []

    def test_it_is_a_warning_not_a_crisis(self):
        """The run may be perfectly healthy — what is wrong is our reading of
        it. A healthy tutorial must never produce a CRITICAL finding, and that
        rule is what the certification pipeline asserts hardest."""
        from courant.diagnosis.checkers import CaseIntegrityChecker

        found = CaseIntegrityChecker().run(
            ctx(config=CaseConfig(unreadable={"system/fvSolution": "x"})))

        assert found[0].severity is FindingSeverity.WARN

    def test_it_is_wired_into_the_default_set(self):
        from courant.diagnosis.checkers import run_all_checkers

        found = run_all_checkers(
            ctx(config=CaseConfig(unreadable={"system/controlDict": "x"})))

        assert "case.unreadable_dict" in found.ids


class TestNotConvergedIsNotACause:
    """"The residuals were above tolerance" is true of every failed run ever.

    Reported unconditionally, `residual.not_converged` put
    `CONV.RESIDUAL_PLATEAU` at the top of a missing dictionary, a negative-volume
    mesh and a wrong turbulence model in a single certification run — beating,
    in the mesh case, a CRITICAL `mesh.negative_volume` that says exactly what
    is wrong. It is a restatement of the failure, not an explanation of it.
    """

    def _finished_high(self):
        return [residual(1, 0.1, {"Ux": 0.5, "p": 0.5})]

    def test_a_crashed_run_does_not_get_it(self):
        crash = Event(run_id="r", seq=2, kind=EventKind.CRASH,
                      payload=CrashPayload(signal="SIGFPE"))

        found = ResidualAnalyzer().run(ctx(events=[*self._finished_high(), crash]))

        assert "residual.not_converged" not in {f.id for f in found}

    def test_a_run_that_merely_stalled_still_gets_it(self):
        found = ResidualAnalyzer().run(ctx(events=self._finished_high()))

        assert "residual.not_converged" in {f.id for f in found}

    def test_it_blames_the_linear_solver_when_the_tolerance_is_loose(self):
        """Both NUM.SOLVER_TOLERANCE and CONV.RESIDUAL_PLATEAU list this finding,
        so they score identically and `suggests` decides — and it always decided
        for the plateau, including on a case whose pressure solver had been told
        to stop at 0.1."""
        from courant.case.config import CaseConfig, LinearSolverSpec

        # A steady solver: a transient one is judged on stability, not on a
        # residual bar (see the next test), so it never gets this finding.
        config = CaseConfig(solver="simpleFoam", linear_solvers={
            "p": LinearSolverSpec(field="p", tolerance=0.1, rel_tol=0.1)})

        found = ResidualAnalyzer().run(ctx(events=self._finished_high(),
                                           config=config))

        assert [f for f in found
                if f.id == "residual.not_converged"][0].suggests \
            == ["NUM.SOLVER_TOLERANCE"]

    def test_a_transient_run_is_judged_on_stability_not_a_residual_bar(self):
        """One definition of converged: `solve_converged` judges a time-marching
        solve on stability. The residual bar here raised a WARN on 4 of the 7
        unmodified tutorials, all transient (RRD RM-01 v0)."""
        from courant.case.config import CaseConfig

        found = ResidualAnalyzer().run(ctx(events=self._finished_high(),
                                           config=CaseConfig(solver="pisoFoam")))

        assert "residual.not_converged" not in {f.id for f in found}

    def test_a_properly_solved_inner_loop_is_a_plateau(self):
        from courant.case.config import CaseConfig, LinearSolverSpec

        config = CaseConfig(solver="simpleFoam", linear_solvers={
            "p": LinearSolverSpec(field="p", tolerance=1e-6, rel_tol=0.05)})

        found = ResidualAnalyzer().run(ctx(events=self._finished_high(),
                                           config=config))

        assert [f for f in found
                if f.id == "residual.not_converged"][0].suggests \
            == ["CONV.RESIDUAL_PLATEAU"]


class TestLaminarOnATurbulentFlow:
    """The Courant number is what separates "the physics is wrong" from "the
    timestep is too big".

    Both make a laminar solve grow and die, and they look identical in the
    residuals. If the flow outran the timestep, the timestep is the explanation
    and the turbulence model is a bystander. If it did not, the numerics were
    fine and the physics is what is left.

    Without that discriminator this check could not fire on a crashed run at all
    — it would have hit every icoFoam blow-up in the corpus — so
    PHYS.LAMINAR_TURBULENT was left scoring on `simulationType laminar` alone,
    with no Finding behind it. It reached the top of a diagnosis citing nothing,
    which is Phase 1's definition of done failing quietly.
    """

    LAMINAR = CaseConfig(solver="icoFoam", turbulence_type="laminar")

    def _crashed(self, courant=None):
        return [residual(1, 0.1, {"Ux": 0.5}, courant=courant),
                Event(run_id="r", seq=2, kind=EventKind.CRASH,
                      payload=CrashPayload(signal="SIGFPE"))]

    def _ids(self, events, config=None):
        return {f.id for f in PhysicsChecker().run(
            ctx(events=events, config=config or self.LAMINAR))}

    def test_a_laminar_crash_within_the_courant_limit_is_evidence(self):
        assert "physics.laminar_unstable" in self._ids(self._crashed(courant=0.4))

    def test_a_courant_blow_up_is_not(self):
        """The check that keeps this from firing on every icoFoam divergence."""
        assert "physics.laminar_unstable" not in self._ids(self._crashed(courant=66.8))

    def test_a_run_that_never_converged_is_evidence(self):
        found = PhysicsChecker().run(
            ctx(events=[residual(1, 0.1, {"Ux": 0.5})], config=self.LAMINAR))

        hit = [f for f in found if f.id == "physics.laminar_unstable"]
        assert hit and hit[0].severity is FindingSeverity.WARN

    def test_a_healthy_laminar_case_says_nothing(self):
        """`icoFoam/cavity` is laminar and correct. This must be silent on it —
        a CRITICAL or WARN on a tutorial the OpenFOAM project ships as right is
        the most expensive class of bug this product has."""
        assert "physics.laminar_unstable" not in self._ids(
            [residual(1, 0.1, {"Ux": 1e-7}, courant=0.4)])

    def test_a_turbulent_case_is_never_implicated(self):
        config = CaseConfig(solver="pisoFoam", turbulence_type="RAS",
                            turbulence_model="kEpsilon")

        assert "physics.laminar_unstable" not in self._ids(
            self._crashed(courant=0.4), config=config)


class TestASymptomMustNotBecomeTheDefaultAnswer:
    """Three findings named a cause unconditionally, and each became the default
    answer for every run that produced the symptom.

    `residual.growth` -> NUM.OVER_RELAXATION: growing residuals *are*
    divergence. It topped three unrelated pitzDaily faults on a case whose
    relaxation factors are OpenFOAM's own 0.7.

    `bounding.k` -> BC.TURB_INLET: two INFO bounding findings put the inlet
    condition at the top of a fault that was *literally* an unbounded scheme on
    `div(phi,k)`. This was a known, documented weakness before it was measured.

    Both are now answered from the case, which knows.
    """

    STEADY_HOT = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                            relaxation_factors={"U": 0.95})
    STEADY_CALM = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                             relaxation_factors={"U": 0.7, "p": 0.3})
    TRANSIENT = CaseConfig(solver="pisoFoam", ddt_scheme="Euler")

    def _growth(self):
        return [Event(run_id="r", seq=1, kind=EventKind.DIVERGENCE,
                      payload=DivergencePayload(field="Ux", slope=0.8))]

    def _suggests(self, config):
        found = ResidualAnalyzer().run(ctx(events=self._growth(), config=config))
        return [f for f in found if f.id == "residual.growth"][0].suggests

    def test_a_steady_case_with_high_relaxation_still_gets_the_suggestion(self):
        assert self._suggests(self.STEADY_HOT) == ["NUM.OVER_RELAXATION"]

    def test_a_steady_case_at_openfoams_own_defaults_does_not(self):
        assert self._suggests(self.STEADY_CALM) == []

    def test_a_transient_case_never_does(self):
        """There is no outer loop to under-relax on a PISO transient."""
        assert self._suggests(self.TRANSIENT) == []

    def test_bounding_blames_the_scheme_that_is_unbounded(self):
        config = CaseConfig(solver="simpleFoam",
                            div_schemes={"div(phi,k)": "Gauss linear"})

        found = BoundingChecker().run(
            ctx(events=[bounding_warn(1, "k")], config=config))

        assert found[0].suggests == ["NUM.UNBOUNDED_SCHEME"]

    def test_bounding_blames_the_inlet_when_the_scheme_is_bounded(self):
        """With the discretisation already limited, something else put k out of
        range — and the inlet condition is the interesting hypothesis again."""
        config = CaseConfig(solver="simpleFoam",
                            div_schemes={"div(phi,k)": "Gauss limitedLinear 1"})

        found = BoundingChecker().run(
            ctx(events=[bounding_warn(1, "k")], config=config))

        assert found[0].suggests == ["BC.TURB_INLET", "NUM.OVER_RELAXATION"]

    def test_it_only_looks_at_the_bounded_fields_own_term(self):
        """`div(phi,U)` being unbounded says nothing about why `k` was clipped."""
        config = CaseConfig(solver="simpleFoam", div_schemes={
            "div(phi,U)": "Gauss linear",
            "div(phi,k)": "Gauss limitedLinear 1"})

        found = BoundingChecker().run(
            ctx(events=[bounding_warn(1, "k")], config=config))

        assert found[0].suggests == ["BC.TURB_INLET", "NUM.OVER_RELAXATION"]

    def test_a_non_turbulence_field_suggests_nothing(self):
        found = BoundingChecker().run(
            ctx(events=[bounding_warn(1, "alpha")], config=self.TRANSIENT))

        assert found[0].suggests == []


class TestAMissingDictionaryEntryIsDiagnosable:
    """`Entry 'X' not found in dictionary` is one of the most common errors
    OpenFOAM emits, and Courant read nothing from it.

    Measured on `pisoFoam/RAS/cavity` with kEpsilon swapped for kOmegaSST: the
    model needs a wall-distance method the case does not carry, so the run dies
    with

        --> FOAM FATAL IO ERROR: (openfoam-2406)
        Entry 'method' not found in dictionary "/case/system/fvSchemes/wallDist"

    naming the exact missing entry. Courant's diagnosis was **empty** —
    `crash.fatal` fired, no KB entry references it, and the answer to "what went
    wrong" was nothing at all on a log that says precisely what went wrong.
    """

    LOG = ('--> FOAM FATAL IO ERROR: (openfoam-2406)\n'
           'Entry \'method\' not found in dictionary '
           '"/case/system/fvSchemes/wallDist"\n\nFOAM exiting\n')

    def _findings(self, log):
        crash = Event(run_id="r", seq=1, kind=EventKind.CRASH,
                      payload=CrashPayload(signal=None))
        return {f.id: f for f in CrashSignatureChecker().run(
            ctx(events=[crash], log_text=log))}

    def test_the_missing_entry_is_named(self):
        found = self._findings(self.LOG)

        assert "crash.missing_dict" in found
        assert "ENV.MISSING_DICT" in found["crash.missing_dict"].suggests

    def test_it_reaches_a_diagnosis(self):
        """The point. `crash.fatal` alone is in no KB signature, so it produced
        an empty diagnosis."""
        from courant.diagnosis.findings import FindingSet
        from courant.diagnosis.kb_data import build_default_kb

        matches = build_default_kb().query(
            findings=FindingSet(list(self._findings(self.LOG).values())),
            log_text=self.LOG)

        assert matches and matches[0].code == "ENV.MISSING_DICT"

    def test_a_healthy_log_is_untouched(self):
        """The signature is only consulted when the run actually crashed, and
        must not match ordinary solver chatter."""
        assert self._findings(
            "Time = 0.1\nsmoothSolver: Solving for Ux, Initial residual = 1e-7\n"
        ).get("crash.missing_dict") is None


class TestTheCaseReaderSpeaksEveryFamilysDialect:
    """Two bugs found independently by two solver families in the same hour.

    `RASModel` is what OpenFOAM calls the key; the reader asked for `model`, so
    `turbulence_model` was `None` on every RAS case in the corpus while
    `RASModel kEpsilon;` sat in the dictionary being read. The fault injector
    already wrote `RAS/RASModel` correctly — the writer and the reader disagreed
    about the same key.

    And `transport` was read only from `constant/transportProperties`, which
    compressible and buoyant cases do not have. Their viscosity is `mu` at
    `mixture/transport/mu` in `thermophysicalProperties`, so `nu_out_of_range`
    and `physics.nu` could not fire on either family — a units error in `mu`,
    the most common real mistake there is, was undetectable.
    """

    def _case(self, tmp_path, rel, body):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FoamFile { version 2.0; format ascii; class dictionary; "
                        f'object {path.name}; }}\n{body}')
        return tmp_path

    def test_the_ras_model_is_read(self, tmp_path):
        case = self._case(tmp_path, "constant/turbulenceProperties",
                          "simulationType RAS;\n"
                          "RAS { RASModel kEpsilon; turbulence on; }\n")

        config = CaseConfig.from_foam(str(case))

        assert config.turbulence_type == "RAS"
        assert config.turbulence_model == "kEpsilon"

    def test_the_les_model_is_read(self, tmp_path):
        case = self._case(tmp_path, "constant/turbulenceProperties",
                          "simulationType LES;\nLES { LESModel WALE; }\n")

        assert CaseConfig.from_foam(str(case)).turbulence_model == "WALE"

    def test_thermophysical_viscosity_is_read(self, tmp_path):
        """The real v2406 shape, from `rhoPimpleFoam/RAS/cavity`."""
        case = self._case(tmp_path, "constant/thermophysicalProperties",
                          "thermoType { type hePsiThermo; }\n"
                          "mixture\n{\n"
                          "    specie { molWeight 28.9; }\n"
                          "    thermodynamics { Cp 1007; Hf 0; }\n"
                          "    transport { mu 1.84e-05; Pr 0.7; }\n}\n")

        transport = CaseConfig.from_foam(str(case)).transport

        assert transport["mu"] == 1.84e-05
        assert transport["Pr"] == 0.7
        assert transport["Cp"] == 1007

    def test_an_incompressible_case_is_unchanged(self, tmp_path):
        case = self._case(tmp_path, "constant/transportProperties",
                          "transportModel Newtonian;\nnu 1e-05;\n")

        assert CaseConfig.from_foam(str(case)).transport["nu"] == 1e-05
