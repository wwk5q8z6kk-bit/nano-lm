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

ALL_AXES = {"denial": DENIAL_ARMS, "hedge": HEDGE_ARMS}
