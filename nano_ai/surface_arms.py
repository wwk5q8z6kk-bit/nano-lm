"""Surface arms: the wordings each epistemic state is varied over.

Every arm records where its wording came from. Three provenance classes, and the
distinction matters when reading results:

  training distribution -- verbatim from the calibration partition. The
      in-distribution reference.
  external lexicon      -- built around a trigger from an open-licensed
      inventory vendored under `data/external/`, authored with no knowledge of
      this project.
  author constructed    -- written here. Honest about not being independent;
      used only where no suitable open inventory exists.

The development partition's own wordings are the baseline arm (`DEV`).
"""

from __future__ import annotations

from nano_ai.surface import SurfaceArm

# What the sealed development partition actually says, per concept. These are
# the strings every other arm replaces.
DEV_DENIAL_MEDICATION = "Nothing at all."
DEV_DENIAL_ALLERGY = "None whatsoever."
DEV_HEDGES = ("I'm unsure.", "I cannot say for certain.", "I don't remember.")


def _denial(label, med, alg, provenance, *, in_distribution=False) -> SurfaceArm:
    return SurfaceArm(
        label=label,
        axis="denial",
        mapping=((DEV_DENIAL_MEDICATION, med), (DEV_DENIAL_ALLERGY, alg)),
        provenance=provenance,
        in_distribution=in_distribution,
    )


def _hedge(label, phrase, provenance, *, in_distribution=False) -> SurfaceArm:
    return SurfaceArm(
        label=label,
        axis="hedge",
        mapping=tuple((h, phrase) for h in DEV_HEDGES),
        provenance=provenance,
        in_distribution=in_distribution,
    )


BASELINE_DENIAL = _denial(
    "DEV", DEV_DENIAL_MEDICATION, DEV_DENIAL_ALLERGY, "sealed development partition"
)
BASELINE_HEDGE = SurfaceArm(
    label="DEV",
    axis="hedge",
    mapping=(),  # identity: leave the development wordings in place
    provenance="sealed development partition",
)

# ---------------------------------------------------------------- denial arms

_TRAIN_DENIALS = (
    ("No, nothing.", "No allergies!"),
    ("No nothing yet!", "Not that I know of!"),
    ("I deny taking medications.", "I deny allergies."),
    ("I denied taking medicine.", "I denied any allergy."),
)

# Patient-voice denials whose negation trigger comes from a vendored lexicon.
# The trigger is independent; the sentence framing is ours.
_EXTERNAL_DENIALS = (
    ("don't", "I don't take medications.", "I don't have allergies."),
    ("not", "I'm not on any medications.", "I'm not allergic."),
    ("never", "I never took medications.", "I never had allergies."),
    ("no", "I have no medications.", "I have no allergies."),
    ("negative for", "Negative for medications.", "Negative for allergies."),
    ("didn't", "I didn't take medications.", "I didn't have allergies."),
    ("cannot", "I cannot take medications.", "No signs of allergies."),
    ("absence of", "Absence of medications.", "Absence of allergies."),
    # medspacy NEGATED_EXISTENCE, independent of negspacy
    ("denies", "Denies medications.", "Denies allergies."),
    ("without", "Without medications.", "Without allergies."),
    ("free of", "Free of medications.", "Free of allergies."),
    ("ruled out", "Medications ruled out.", "Allergies ruled out."),
)

DENIAL_ARMS = (
    BASELINE_DENIAL,
    *(
        _denial(f"TRAIN[{i}]", med, alg, "training distribution", in_distribution=True)
        for i, (med, alg) in enumerate(_TRAIN_DENIALS)
    ),
    *(
        _denial(
            f"EXTERNAL[{i}]",
            med,
            alg,
            f"external lexicon trigger {trigger!r} "
            "(negspacy en_clinical / medspacy NEGATED_EXISTENCE, both MIT)",
        )
        for i, (trigger, med, alg) in enumerate(_EXTERNAL_DENIALS)
    ),
)

# ----------------------------------------------------------------- hedge arms
#
# No open-licensed inventory of *patient-voice epistemic* hedges was found.
# medspacy POSSIBLE_EXISTENCE is clinician-register diagnostic hedging ("rule
# out", "suspicious for") and does not express a speaker's uncertainty about
# their own history, so it is deliberately not used as a source here. The
# held-out hedge arms are therefore author-constructed and labelled as such;
# they are weaker evidence than the denial arms and must not be reported as
# independent.

_TRAIN_HEDGES = (
    "I do not have dependable information for it.",
    "I would not want to guess about that.",
    "I cannot confirm the requested fact.",
    "I cannot give a reliable detail.",
    "That point is not clear in my memory.",
    "I lack confidence in that answer.",
)

_CONSTRUCTED_HEDGES = (
    "I honestly couldn't tell you.",
    "I'm not really sure about that.",
    "Hard to say.",
    "I might be misremembering.",
    "Your guess is as good as mine.",
    "I couldn't swear to it.",
)

HEDGE_ARMS = (
    BASELINE_HEDGE,
    *(
        _hedge(f"TRAIN[{i}]", phrase, "training distribution", in_distribution=True)
        for i, phrase in enumerate(_TRAIN_HEDGES)
    ),
    *(
        _hedge(f"CONSTRUCTED[{i}]", phrase, "author constructed -- NOT independent")
        for i, phrase in enumerate(_CONSTRUCTED_HEDGES)
    ),
)

# ----------------------------------------------------------------- value arms
#
# `denial` and `hedge` vary *how* a concept is said. `value` varies *what the
# world is*: which medication or allergy the transcript names, holding the
# sealed development set's own answer template fixed. This is the axis the
# vocabulary-ceiling result did not touch -- DP-1 and the lexical-substitution
# probes are all about the ten denial strings, never about which of the 24
# medication/allergy names development uses.
#
# The development and calibration name pools are disjoint by construction (0
# overlap out of 24 checked both ways, verified against
# `artifacts/nano_h6/kaggle/dataset-dev/dev.jsonl` and
# `artifacts/nano_h5/data/calibration.jsonl` on 2026-08-06) -- the same "world"
# disjointness the plan's axis table (`PLAN_20260805_SURFACE_ROBUSTNESS.md`
# section 2) names as confounded with surface and document structure in the
# original held-out number. Holding the template fixed and swapping only the
# name isolates that one axis.

_DEV_MEDICATION_VALUES = (
    "adapalene gel", "azelastine spray", "capsaicin cream", "carvedilol tablet",
    "cefpodoxime tablet", "diltiazem capsule", "empagliflozin tablet",
    "escitalopram tablet", "etodolac tablet", "famciclovir tablet",
    "hyoscyamine tablet", "magnesium glycinate capsule", "nystatin cream",
    "pantoprazole tablet", "pravastatin tablet", "psyllium powder",
    "riboflavin tablet", "rizatriptan wafer", "scopolamine patch",
    "sucralfate suspension", "tiotropium inhaler", "tizanidine tablet",
    "topiramate tablet", "zafirlukast tablet",
)
_DEV_ALLERGY_VALUES = (
    "acrylic adhesive", "anchovies", "apricots", "balsam of peru",
    "cedar pollen", "cefuroxime", "cranberries", "dragon fruit", "elm pollen",
    "fennel seed", "goat's milk", "guinea pig dander", "hornet venom",
    "hydromorphone", "linezolid", "macadamia nuts", "passion fruit",
    "pine nuts", "poppy seeds", "povidone iodine", "quail eggs",
    "rubber accelerator", "tilapia", "vancomycin",
)

# The calibration partition's medication/allergy vocabulary -- the same
# partition DP-1 measured against (`denial_probe_calibration.json`,
# "partition": "calibration (development not opened)"). This, not development,
# is the in-distribution reference: it is what the model was scored on but
# never selected against.
_CALIBRATION_MEDICATION_VALUES = (
    "albuterol inhaler", "benzoyl peroxide wash", "budesonide inhaler",
    "electrolyte powder", "fluticasone spray", "ginger capsule",
    "glucosamine tablet", "guaifenesin tablet", "ipratropium spray",
    "iron tablet", "levothyroxine tablet", "meclizine tablet",
    "mesalamine tablet", "mometasone cream", "nitrofurantoin capsule",
    "sertraline tablet",
)
_CALIBRATION_ALLERGY_VALUES = (
    "banana", "birch pollen", "cat dander", "celery root", "chlorhexidine",
    "chromium", "codeine", "egg whites", "kiwi fruit", "mango", "papaya",
    "peaches", "strawberries", "tramadol", "wheat", "wool",
)


def _value(label, med_value, alg_value, provenance, *, in_distribution=False) -> SurfaceArm:
    # Every known development name maps to this arm's one calibration name; in
    # any given document only one medication and one allergy name are
    # actually present, so only one pair per field fires (the rest are
    # no-ops), exactly as `substitute` already handles for denial/hedge.
    return SurfaceArm(
        label=label,
        axis="value",
        mapping=(
            *((dev, med_value) for dev in _DEV_MEDICATION_VALUES),
            *((dev, alg_value) for dev in _DEV_ALLERGY_VALUES),
        ),
        provenance=provenance,
        in_distribution=in_distribution,
    )


BASELINE_VALUE = SurfaceArm(
    label="DEV", axis="value", mapping=(), provenance="sealed development partition",
)

VALUE_ARMS = (
    BASELINE_VALUE,
    *(
        _value(
            f"TRAIN[{i}]", med, alg,
            "calibration partition (training distribution, development not opened)",
            in_distribution=True,
        )
        for i, (med, alg) in enumerate(
            zip(_CALIBRATION_MEDICATION_VALUES, _CALIBRATION_ALLERGY_VALUES, strict=True)
        )
    ),
)

# --------------------------------------------------------------- template arms
#
# `template` is `value`'s complement: hold the name fixed, vary the sentence
# that carries it. Together the two axes separate "reads the frame" (template)
# from "reads the value" (value) for the `supported` state -- a distinction
# `hedge` and `denial` cannot make because their target concept has no
# independent open value.
#
# Because the development/calibration name pools are closed and enumerated
# above, every (template, name) combination can be precomputed as an exact
# literal line -- no runtime `{VALUE}` interpolation is needed in the
# substitution path, so this reuses `SurfaceArm`/`substitute` unmodified. The
# gold bracket in `target` (e.g. `MED:S[capsaicin cream]`) is untouched by a
# template arm, which is correct: the value does not change, only its frame.

_DEV_MEDICATION_TEMPLATE = "Only {VALUE} so far."
_DEV_ALLERGY_TEMPLATE = "I do — {VALUE}."

# The four templates the calibration partition uses for both medication and
# allergy answers (verified identical sets on 2026-08-06).
_CALIBRATION_TEMPLATES = (
    "For the record, the detail is {VALUE} today.",
    "I would document the answer as {VALUE}",
    "[{VALUE}]",
    "{VALUE} — that is my reply.",
)


def _template(label, template, provenance, *, in_distribution=False) -> SurfaceArm:
    mapping = tuple(
        (_DEV_MEDICATION_TEMPLATE.replace("{VALUE}", v), template.replace("{VALUE}", v))
        for v in _DEV_MEDICATION_VALUES
    ) + tuple(
        (_DEV_ALLERGY_TEMPLATE.replace("{VALUE}", v), template.replace("{VALUE}", v))
        for v in _DEV_ALLERGY_VALUES
    )
    return SurfaceArm(
        label=label, axis="template", mapping=mapping, provenance=provenance,
        in_distribution=in_distribution,
    )


BASELINE_TEMPLATE = SurfaceArm(
    label="DEV", axis="template", mapping=(), provenance="sealed development partition",
)

TEMPLATE_ARMS = (
    BASELINE_TEMPLATE,
    *(
        _template(
            f"TRAIN[{i}]", tmpl,
            "calibration partition (training distribution, development not opened)",
            in_distribution=True,
        )
        for i, tmpl in enumerate(_CALIBRATION_TEMPLATES)
    ),
)

# `value` and `template` both target the `supported` state but only ever
# rewrite the medication/allergy fields; unlike `denial`/`hedge` their arms
# must not be scored against chief_complaint/duration/severity, which never
# vary between arms and would dilute sensitivity toward zero. Harness runners
# read this to restrict scoring -- see `run_surface_harness.py::_PRIMARY_FIELDS`.
VALUE_TEMPLATE_FIELDS = ("medication", "allergy")

ALL_AXES = {
    "denial": DENIAL_ARMS,
    "hedge": HEDGE_ARMS,
    "value": VALUE_ARMS,
    "template": TEMPLATE_ARMS,
}
