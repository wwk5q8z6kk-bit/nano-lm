"""Multi-instance surface-value resampling for the campaign_v2 suite.

Structure is held fixed and surface values are resampled, so a result can be
reported as mean +- across-instance SD instead of a single 16-slot point
estimate. This is the same instrument shape Paper 1 used (m0-m4) and it is what
lets the leakage ablation say anything about effect size rather than direction
alone.

``i0`` reproduces the original hand-authored values byte-for-byte, so the
campaign_v1 partition and every prior claim scored on it stay comparable;
``test_i0_reproduces_the_original_sources`` pins that.

Invariants every instance must satisfy — asserted mechanically in
``test_instance_invariants``, not by inspection:

1. each present gold value occurs exactly once in its own source (the
   constrained selector abstains on ambiguous quotes, so a duplicated surface
   value would silently become an abstention);
2. each absent value occurs zero times in its encounter's source (otherwise the
   false-positive probe is not probing absence);
3. no instance value collides with the neutral system-prompt examples, which
   would re-open the leakage channel the C1-off cell is meant to close;
4. every instance carries the same slot count and the same atom-type multiset,
   so across-instance SD is computed over comparable draws.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanoscribe.encounter import AtomType

# Values used as format examples in the neutral system prompt. No instance may
# reuse these — see invariant 3.
RESERVED_VALUES = frozenset({"ankle", "No prior surgery.", "lightheaded"})

AbsentSlot = tuple[str, AtomType, str]


@dataclass(frozen=True, slots=True)
class InstanceValues:
    """Surface values for one draw of the fixed encounter structure."""

    instance_id: str
    # enc-1 — symptom / assessment / denial / history
    e1_site: str
    e1_assessment: str
    e1_denial: str
    e1_history: str
    # enc-2 — uncertainty
    e2_probe: str
    e2_value: str
    # enc-3 — family-history attribution
    e3_family: str
    e3_symptom: str
    # enc-4 — absent-atom probe
    e4_site: str
    e4_present: str
    e4_absent: tuple[AbsentSlot, ...]
    # enc-5 — explicit denial
    e5_denied_history: str
    e5_denied_symptom: str

    def present_values(self) -> tuple[tuple[str, str], ...]:
        """(encounter_id, value) for every value that must occur exactly once."""
        return (
            ("enc-1", self.e1_site),
            ("enc-1", self.e1_assessment),
            ("enc-1", self.e1_denial),
            ("enc-1", self.e1_history),
            ("enc-2", self.e2_value),
            ("enc-3", self.e3_family),
            ("enc-3", self.e3_symptom),
            ("enc-4", self.e4_present),
            ("enc-5", self.e5_denied_history),
            ("enc-5", self.e5_denied_symptom),
        )

    def all_values(self) -> tuple[str, ...]:
        return tuple(value for _enc, value in self.present_values()) + tuple(
            value for _id, _type, value in self.e4_absent
        )


def _absent(
    med: str, sym: str, allergy: str, hist: str, meas: str
) -> tuple[AbsentSlot, ...]:
    return (
        ("atom-absent-med", AtomType.MEDICATION, med),
        ("atom-absent-fever", AtomType.SYMPTOM, sym),
        ("atom-absent-allergy", AtomType.ALLERGY, allergy),
        ("atom-absent-hist", AtomType.HISTORY, hist),
        ("atom-absent-vital", AtomType.MEASUREMENT, meas),
    )


# i0 reproduces the original hand-authored encounters exactly.
INSTANCES: tuple[InstanceValues, ...] = (
    InstanceValues(
        instance_id="i0",
        e1_site="neck",
        e1_assessment="cervical strain",
        e1_denial="No allergies.",
        e1_history="migraines",
        e2_probe="chest pain",
        e2_value="pressure",
        e3_family="diabetes",
        e3_symptom="tired",
        e4_site="throat",
        e4_present="sore",
        e4_absent=_absent("lisinopril", "fever", "penicillin", "asthma", "blood pressure"),
        e5_denied_history="smoked",
        e5_denied_symptom="wheezing",
    ),
    InstanceValues(
        instance_id="i1",
        e1_site="shoulder",
        e1_assessment="rotator cuff strain",
        e1_denial="No known allergies.",
        e1_history="eczema",
        e2_probe="stomach pain",
        e2_value="cramping",
        e3_family="glaucoma",
        e3_symptom="dizzy",
        e4_site="knee",
        e4_present="swollen",
        e4_absent=_absent("metformin", "chills", "sulfa", "bronchitis", "heart rate"),
        e5_denied_history="vaped",
        e5_denied_symptom="coughing",
    ),
    InstanceValues(
        instance_id="i2",
        e1_site="wrist",
        e1_assessment="tendon sprain",
        e1_denial="No allergies to report.",
        e1_history="sinusitis",
        e2_probe="back pain",
        e2_value="numbness",
        e3_family="psoriasis",
        e3_symptom="restless",
        e4_site="elbow",
        e4_present="stiff",
        e4_absent=_absent("atorvastatin", "nausea", "latex", "epilepsy", "body temperature"),
        e5_denied_history="drank",
        e5_denied_symptom="gasping",
    ),
    InstanceValues(
        instance_id="i3",
        e1_site="jaw",
        e1_assessment="joint strain",
        e1_denial="No allergies at all.",
        e1_history="vertigo",
        e2_probe="ear pain",
        e2_value="ringing",
        e3_family="anemia",
        e3_symptom="queasy",
        e4_site="hip",
        e4_present="tender",
        e4_absent=_absent("omeprazole", "sweating", "iodine", "hepatitis", "oxygen level"),
        e5_denied_history="smoked cigars",
        e5_denied_symptom="rattling",
    ),
    InstanceValues(
        instance_id="i4",
        e1_site="hand",
        e1_assessment="carpal tunnel syndrome",
        e1_denial="No medication allergies.",
        e1_history="shingles",
        e2_probe="hip pain",
        e2_value="throbbing",
        e3_family="gout",
        e3_symptom="hoarse",
        e4_site="foot",
        e4_present="itchy",
        e4_absent=_absent("warfarin", "fatigue", "shellfish", "pneumonia", "blood sugar"),
        e5_denied_history="chewed tobacco",
        e5_denied_symptom="whistling",
    ),
)

INSTANCE_IDS: tuple[str, ...] = tuple(item.instance_id for item in INSTANCES)


def instance(instance_id: str) -> InstanceValues:
    for item in INSTANCES:
        if item.instance_id == instance_id:
            return item
    raise KeyError(f"unknown instance: {instance_id}")


def split_encounter_id(encounter_id: str) -> tuple[str, str]:
    """'enc-4@i2' -> ('enc-4', 'i2'); bare ids belong to i0."""
    if "@" in encounter_id:
        base, _, inst = encounter_id.partition("@")
        return base, inst
    return encounter_id, "i0"
