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
    InstanceValues(
        instance_id="i5", e1_site="hip", e1_assessment="bursitis",
        e1_denial="No allergies known.", e1_history="tonsillitis",
        e2_probe="jaw pain", e2_value="clicking",
        e3_family="lupus", e3_symptom="sluggish",
        e4_site="shin", e4_present="bruised",
        e4_absent=_absent("ibuprofen", "vomiting", "pollen", "measles", "pulse rate"),
        e5_denied_history="gambled", e5_denied_symptom="snoring",
    ),
    InstanceValues(
        instance_id="i6", e1_site="thumb", e1_assessment="joint inflammation",
        e1_denial="No allergies noted.", e1_history="chickenpox",
        e2_probe="neck pain", e2_value="stiffness",
        e3_family="arthritis", e3_symptom="listless",
        e4_site="calf", e4_present="cramped",
        e4_absent=_absent("aspirin", "rash", "peanuts", "tuberculosis", "respiratory rate"),
        e5_denied_history="fasted", e5_denied_symptom="hiccupping",
    ),
    InstanceValues(
        instance_id="i7", e1_site="heel", e1_assessment="plantar irritation",
        e1_denial="No allergies whatsoever.", e1_history="mumps",
        e2_probe="eye pain", e2_value="blurring",
        e3_family="cataracts", e3_symptom="forgetful",
        e4_site="wrist", e4_present="puffy",
        e4_absent=_absent("insulin", "shivering", "dust", "jaundice", "weight reading"),
        e5_denied_history="skipped meals", e5_denied_symptom="grunting",
    ),
    InstanceValues(
        instance_id="i8", e1_site="temple", e1_assessment="tension headache",
        e1_denial="No allergies at this time.", e1_history="scarlet fever",
        e2_probe="throat pain", e2_value="scratchiness",
        e3_family="epilepsy", e3_symptom="clumsy",
        e4_site="thigh", e4_present="numb",
        e4_absent=_absent("codeine", "dizziness", "mould", "rickets", "glucose reading"),
        e5_denied_history="used inhalers", e5_denied_symptom="sighing",
    ),
    InstanceValues(
        instance_id="i9", e1_site="spine", e1_assessment="disc irritation",
        e1_denial="No allergies that we know.", e1_history="whooping cough",
        e2_probe="hand pain", e2_value="tingling",
        e3_family="dementia", e3_symptom="anxious",
        e4_site="jaw", e4_present="locked",
        e4_absent=_absent("paracetamol", "bloating", "grass", "polio", "temperature reading"),
        e5_denied_history="worked nights", e5_denied_symptom="rasping",
    ),
    InstanceValues(
        instance_id="i10", e1_site="knuckle", e1_assessment="soft tissue swelling",
        e1_denial="No allergies in the notes.", e1_history="pleurisy",
        e2_probe="foot pain", e2_value="burning",
        e3_family="thyroid disease", e3_symptom="irritable",
        e4_site="rib", e4_present="blotchy",
        e4_absent=_absent("ramipril", "sneezing", "nickel", "malaria", "blood count"),
        e5_denied_history="relied on sleeping pills", e5_denied_symptom="groaning",
    ),
    InstanceValues(
        instance_id="i11", e1_site="collarbone", e1_assessment="ligament strain",
        e1_denial="No allergies on file.", e1_history="appendicitis",
        e2_probe="leg pain", e2_value="heaviness",
        e3_family="osteoporosis", e3_symptom="drowsy",
        e4_site="forearm", e4_present="mottled",
        e4_absent=_absent("antibiotics", "belching", "wool", "meningitis", "oxygen reading"),
        e5_denied_history="missed appointments", e5_denied_symptom="panting",
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


# --------------------------------------------------------------------------
# Concept labels — the Q_SURFACE fix
# --------------------------------------------------------------------------
#
# A slot has to be identifiable or the task is underdetermined, but naming it
# with its gold SURFACE string hands the model the answer: a parrot that never
# reads the transcript scored within one slot of the perfect-reader ceiling in
# every cell of the first 2x2. The two jobs are separable — these labels
# identify a slot by its ROLE in the encounter, so the question stays fully
# specified while the surface form stays unsaid.
#
# Structure is fixed across instances, so labels are keyed by atom_id and shared
# by every draw; only the surface values resample. Invariants are enforced per
# instance in test_campaign_instances (stem-disjoint from that instance's
# raw_values, in both directions, instance-wide — not merely slot-local).
CONCEPT_LABELS: dict[str, str] = {
    # enc-1
    "atom-neck": "the place the patient says is hurting",
    "atom-alg": "whether the patient rules out reactions to anything",
    "atom-hist": "a condition the patient had years earlier",
    "atom-assess": "the clinician's stated impression",
    "medication": "a drug the patient is currently taking",
    # enc-2
    "atom-chest": "the sensation the patient is unsure about",
    # enc-3
    "atom-fh": "the illness a relative had",
    "atom-tired": "how the patient has felt lately",
    # enc-4
    "atom-throat": "the way the patient describes their complaint",
    "atom-absent-med": "a prescription the patient takes",
    "atom-absent-fever": "a symptom other than the presenting complaint",
    "atom-absent-allergy": "a substance the patient says they react to",
    "atom-absent-hist": "a past illness in the patient's own record",
    "atom-absent-vital": "a recorded vital sign",
    # enc-5
    "atom-smoke": "a habit the patient rules out",
    "atom-wheeze": "the breathing sign the patient denies",
}


def concept_label(atom_id: str) -> str:
    return CONCEPT_LABELS[atom_id]


# --- stem-level overlap checking (invariant support) -----------------------

_STOPWORDS = frozenset(
    """a an and any are as at be been being by do does for from had has have how
    if in is it its lately of on or other own says that the their them they this
    to what when where whether which who with""".split()
)

_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ies", "es", "ed", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def content_stems(text: str) -> set[str]:
    """Casefolded, punctuation-stripped, stopword-free, suffix-stripped tokens."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    tokens = [tok for tok in cleaned.casefold().split() if tok and tok not in _STOPWORDS]
    return {_stem(tok) for tok in tokens if len(tok) > 2}


def stems_overlap(left: str, right: str, *, prefix_floor: int = 4) -> set[str]:
    """Shared content stems, tolerant of morphological variation.

    A plain substring test is not enough: raw_value "migraines" and label
    "past migraine condition" contain neither string in the other yet still
    share the stem. Two stems collide when they are equal, or when one is a
    prefix of the other and the shared prefix is at least `prefix_floor` long
    (catches migraine/migrainous, temperature/temperatures).
    """
    shared: set[str] = set()
    right_stems = content_stems(right)
    for a in content_stems(left):
        for b in right_stems:
            if a == b or (
                len(min(a, b, key=len)) >= prefix_floor
                and (a.startswith(b) or b.startswith(a))
            ):
                shared.add(a if a == b else f"{a}~{b}")
    return shared
