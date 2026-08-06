"""Surface-robustness measurement over epistemic states.

An accuracy figure for a grounded-extraction system is a joint property of the
model *and* the wording the benchmark happened to use. Measured on 2026-08-05:
holding transcripts fixed and varying only the denial phrase moved `absent`
accuracy between 28.1% and 99.3% on one checkpoint. A single held-out surface
form is a sample of size one.

Worse, per-arm accuracy is not reproducible across seeds. Two checkpoints
differing only in seed showed a mean absolute difference of 2.5 points on
in-distribution phrasings and 31.5 points on external ones, with Kendall
tau = 0.00 between their rankings of which external phrasing was easiest. So an
arm measured on one seed carries almost no signal; an arm averaged over seeds
does.

This module holds the pure part -- arm definitions, substitution, and the
aggregation math. Model loading and inference live in
`nano_ai/training/run_surface_harness.py`.

Vocabulary follows `papers/SELECTIVE_VOCABULARY.md`.

---------------------------------------------------------------------------
Metric:            surface_robust_accuracy / surface_mean_accuracy /
                   surface_sensitivity / seed_instability
Purpose:           separate a model's competence at an epistemic state from its
                   fit to the specific wordings a benchmark samples.
Equation:          arm_mean(a)  = mean over seeds of accuracy(a, seed)
                   robust       = min over arms of arm_mean(a)
                   mean         = mean over arms of arm_mean(a)
                   sensitivity  = max arm_mean - min arm_mean
                   instability  = mean over arms of (max_seed - min_seed)
Inputs:            joint-exact correct/total per (arm, seed, gold state)
Output range:      [0,1] for the first three; [0,1] for instability
Interpretation:    `robust` is what may be promised; `sensitivity` says how much
                   of the headline number is a property of the wording;
                   `instability` says how much is a property of the seed.
                   sensitivity >> instability  -> a real surface effect.
                   instability >= sensitivity  -> arms are not distinguishable;
                                                  report the mean only.
Why this formula:  the product claim is trustworthiness, so the loss is
                   asymmetric (one bad phrasing in production is a wrong
                   record) and the worst case is the promisable quantity
                   (`rules/math-toolkit.md` sections 7 and 11). `min` over arm
                   *means* rather than over raw observations, because the seed
                   replication showed single observations are noise.
Why not simpler:   a single held-out accuracy cannot distinguish concept
                   competence from surface memorisation -- that failure caused
                   a model generation to be misdiagnosed.
Why not complex:   no variance model or significance test is fitted; with a
                   handful of seeds the sample cannot support one, and claiming
                   otherwise would hide uncertainty behind decimals.
False positives:   arms that differ in register (clinician vs patient voice) or
                   tokenisation, not polarity, inflate `sensitivity`.
False negatives:   arms drawn from one lexicon share its idiosyncrasies and can
                   understate sensitivity.
Calibration:       arms whose phrasings the model was trained on are the
                   in-distribution reference; external arms come from lexicons
                   vendored under `data/external/`.
Threshold:         none. This reports; it does not gate. Promotion to a gate
                   requires the seed count in `MIN_SEEDS_FOR_ARM_CLAIM` and the
                   data-sufficiency clause in `rules/math-toolkit.md` section 7.
Recheck:           python3 -m nano_ai.training.run_surface_harness --help
When to retire:    when evaluation moves to real documents with naturally
                   varied surface forms, at which point sampling replaces
                   substitution.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

# Below this many seeds, a single arm's accuracy is reported but must not be
# compared against another arm's. Derived from the 2026-08-05 replication:
# mean per-arm seed spread was 31.5 points out-of-distribution, which swamps
# most between-arm differences.
MIN_SEEDS_FOR_ARM_CLAIM = 3


class SurfaceError(ValueError):
    """A surface arm cannot be applied without guessing."""


#: A structural transform: takes (transcript, target) and returns a rewritten
#: (transcript, target), or None when the arm does not apply to this document
#: (e.g. no conflicting field present). Unlike `mapping`, a transform can move
#: text rather than only substitute it -- needed for arms that swap the order
#: of two existing values or change the distance between them, neither of
#: which a sequence of independent (source, target) string replacements can
#: express without one replacement corrupting the text the other just wrote.
ArmTransform = Callable[[str, str], "tuple[str, str] | None"]


@dataclass(frozen=True, slots=True)
class SurfaceArm:
    """One wording of a target concept, substituted into fixed documents."""

    label: str
    axis: str  # "denial", "hedge", ...
    mapping: tuple[tuple[str, str], ...]  # (original phrase, replacement)
    provenance: str  # where the wording came from; "training distribution" etc.
    in_distribution: bool = False
    transform: ArmTransform | None = None

    def replacement_for(self, original: str) -> str | None:
        for source, target in self.mapping:
            if source == original:
                return target
        return None


def substitute(text: str, arm: SurfaceArm, *, require_unique: bool = True) -> str:
    """Apply an arm's rewrites to one string.

    Raises SurfaceError when an original phrase occurs more than once and
    uniqueness is required -- an ambiguous rewrite would silently corrupt the
    gold spans rather than fail loudly.
    """
    out = text
    for source, target in arm.mapping:
        count = out.count(source)
        if count == 0:
            continue
        if require_unique and count != 1:
            raise SurfaceError(
                f"{arm.label}: {source!r} occurs {count} times; rewrite is ambiguous"
            )
        out = out.replace(source, target)
    return out


def apply_arm(transcript: str, target: str, arm: SurfaceArm) -> tuple[str, str] | None:
    """Rewrite one (transcript, target) pair under an arm.

    Mapping-based arms (the common case) call `substitute` on the transcript
    and mirror the same phrase pairs into the bracketed gold target string.
    Transform-based arms delegate entirely to `arm.transform`, which owns
    both strings and returns None when the arm does not apply to this
    document -- the caller must treat None as "drop this document for this
    arm", exactly like a `SurfaceError` from the mapping path.
    """
    if arm.transform is not None:
        if arm.mapping:
            raise SurfaceError(
                f"{arm.label}: an arm cannot carry both a mapping and a transform"
            )
        return arm.transform(transcript, target)
    try:
        new_transcript = substitute(transcript, arm)
    except SurfaceError:
        return None
    new_target = target
    for source, replacement in arm.mapping:
        new_target = new_target.replace(f"[{source}]", f"[{replacement}]")
        new_target = new_target.replace(f"[{source};", f"[{replacement};")
        new_target = new_target.replace(f";{source}]", f";{replacement}]")
    return new_transcript, new_target


@dataclass(frozen=True, slots=True)
class ArmObservation:
    """Joint-exact counts for one arm, one seed, one gold state."""

    arm: str
    axis: str
    seed: str
    state: str
    correct: int
    total: int
    in_distribution: bool = False

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def aggregate(
    observations: Sequence[ArmObservation],
    *,
    axis: str,
    state: str,
    in_distribution: bool | None = None,
) -> dict[str, object] | None:
    """Collapse observations for one (axis, state) into the reported metrics.

    `in_distribution` selects the reference arms (True), the held-out arms
    (False), or all arms (None).
    """
    rows = [
        o
        for o in observations
        if o.axis == axis
        and o.state == state
        and o.total
        and (in_distribution is None or o.in_distribution is in_distribution)
    ]
    if not rows:
        return None

    per_arm: dict[str, list[float]] = {}
    for row in rows:
        per_arm.setdefault(row.arm, []).append(row.accuracy)

    arm_means = {arm: _mean(values) for arm, values in per_arm.items()}
    # Seed instability: how much one arm moves between seeds. Only arms with
    # more than one seed contribute; with none, instability is unmeasured.
    spreads = [max(v) - min(v) for v in per_arm.values() if len(v) > 1]
    seed_counts = {len(v) for v in per_arm.values()}
    values = list(arm_means.values())

    worst = min(arm_means, key=lambda a: arm_means[a])
    best = max(arm_means, key=lambda a: arm_means[a])
    sensitivity = arm_means[best] - arm_means[worst]
    instability = _mean(spreads) if spreads else None

    return {
        "axis": axis,
        "state": state,
        "arms": len(arm_means),
        "seeds_per_arm": sorted(seed_counts),
        "surface_robust_accuracy": round(arm_means[worst], 4),
        "surface_mean_accuracy": round(_mean(values), 4),
        "surface_sensitivity": round(sensitivity, 4),
        "seed_instability": round(instability, 4) if instability is not None else None,
        "worst_arm": worst,
        "best_arm": best,
        "arm_means": {a: round(v, 4) for a, v in sorted(arm_means.items())},
        # The honesty flag: if seeds cannot separate arms, do not let a reader
        # treat the spread as a surface effect.
        "arm_comparison_supported": bool(
            min(seed_counts) >= MIN_SEEDS_FOR_ARM_CLAIM
            and instability is not None
            and sensitivity > instability
        ),
    }


def report_lines(summary: Mapping[str, object]) -> list[str]:
    """Render one aggregate as human-readable lines."""
    instability = summary["seed_instability"]
    lines = [
        f"{summary['axis']}/{summary['state']}: "
        f"robust {summary['surface_robust_accuracy']:.1%}  "
        f"mean {summary['surface_mean_accuracy']:.1%}  "
        f"sensitivity {summary['surface_sensitivity']:.1%}  "
        f"instability {'n/a' if instability is None else f'{instability:.1%}'}  "
        f"({summary['arms']} arms x seeds {summary['seeds_per_arm']})"
    ]
    if not summary["arm_comparison_supported"]:
        lines.append(
            "    arm-level comparison NOT supported "
            f"(needs >= {MIN_SEEDS_FOR_ARM_CLAIM} seeds/arm and sensitivity > instability); "
            "read the mean, not the ordering"
        )
    return lines
