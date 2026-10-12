"""The fixed sentences of the convergence assessment (TRD CONV-14, M1-O-7).

Product surface: rendered character for character, and every numeral in them is a slot
filled from the assessment's `data` (M1-O-8). M1 wrote them for an allocation-unit pack.
Where M1 said "allocation figure", these say "waste figure", because the waste figure is
what this product reports by default. The FIA sentences are kept for `--ccf-ghz`.
"""

NO_STOPPING_POINT_UNSTEADY = (
    "This run used a steady solver on a flow that is not steady. The monitored quantity is in a "
    "limit cycle of period {period_steps} iterations with a peak-to-peak amplitude of {amplitude}, "
    "which is {amplitude_rel} of the reference scale. There is no fixed point to stop at, so no "
    "stopping point and no waste figure are reported. The mean over {n_periods} complete "
    "cycles is {cycle_mean}; it is an indicative cycle mean and not a converged value.")

ITERATION_OSCILLATION = (
    "{monitor} alternates with a period of {lag} iterations, a numerical oscillation of the steady "
    "iteration rather than unsteady flow, with a peak-to-peak amplitude of {amplitude}. Its cycle "
    "mean is {cycle_mean}, and the answer is known only to within {half} of it, which is more "
    "than the tolerance of {tol}. No stopping point is reported.")

NOT_CONVERGED = (
    "This run did not reach a fixed point. {monitor} was still moving at the end, by {drift_rel} "
    "of the reference scale across the last {window} samples, against a limit of {tol}. No "
    "stopping point is reported.")

NOT_STATIONARY = (
    "After removing the initial transient, the remaining series for {monitor} is not stationary. "
    "{which_failed} No mean, no confidence interval and no stopping point are reported, because a "
    "confidence interval on the mean of a non-stationary series does not describe anything.")

INSUFFICIENT_DATA = (
    "There is not enough data to separate the initial transient from the steady state for "
    "{monitor}. {detail} No stopping point is reported. Running longer is the only remedy; there "
    "is no setting that makes a shorter series answer this question.")

CORE_HOURS_NOT_COMPUTED = (
    "Core-hours were not computed, because {missing}. Iterations past convergence are reported "
    "above and do not depend on it.")

ALLOCATION_NOT_COMPUTED = (
    "Allocation unit hours were not computed, because {missing}. Supply it with {flags} and the "
    "figure will be reported. Wall-clock seconds and core count past convergence are reported "
    "above and do not depend on it.")

ALLOCATION_DISCLAIMER = (
    "This allocation figure is computed under the formula in FIA 2026 Sporting Regulations "
    "Section F Article F4.4.4 from the inputs listed above. It is not an FIA determination.")

RUN_KILLED = (
    "This run was stopped by a signal at step {last_complete_step}, not by reaching its end "
    "condition. The convergence assessment above covers the samples that were written. No "
    "wall-clock or allocation figure is reported, because time after a kill was not spent.")

RUN_CRASHED = (
    "This run ended in a crash at step {last_complete_step}, not by reaching its end condition. "
    "The convergence assessment above covers the samples that were written. No waste figure is "
    "reported, because a run that crashed did not finish the work it was set.")

NOT_ASSESSABLE_NO_MONITORS = (
    "No engineering monitor was found in this case, so convergence of the answer could not be "
    "assessed. Residual behaviour is reported below and is not a substitute. The paths searched "
    "were: {paths}.")

TRANSIENT_NO_INTENT = (
    "This is a transient run. Whether its answer is a time-average or a trajectory is the "
    "engineer's call, so the statistics below are reported for reference and do not grade the "
    "run. Declare --transient-average to judge time-averaged monitors.")

ZERO_MEAN_SCALE = (
    "{monitor} sits at zero within its own variability, so a relative tolerance has no scale to "
    "be relative to. It is not judged. Supply --ref-scale {monitor}=VALUE or an absolute "
    "--tol to judge it.")

DECIMATED = (
    "The stopping point was located to a resolution of {step_resolution} steps rather than to a "
    "single step, because the exhaustive search would have exceeded this module's evaluation "
    "budget. The reported stopping point is the later of the two bracketing candidates, so the "
    "reported saving is understated rather than overstated.")

#: Every sentence, for the character-for-character test.
ALL = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, str)}
