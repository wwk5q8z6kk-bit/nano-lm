"""Value vocabularies with an enforced train/dev/held-out split.

P1's core claim is exact transport of OPEN-VOCABULARY values. A model can only be
shown to generalize if the values it is evaluated on were never trained on, so
value pools are partitioned deterministically and the split is asserted, not
assumed.

The frozen screening suite's 37 values are reserved: no generated example may use
one. `forbidden_values()` is the single source of that reservation and
corpus.leakage re-checks it against the live suite at build time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from nanoscribe.native.corpus.schema import Partition

# Values reserved to p1_screening_eval_v1. DERIVED from the live suite, never
# hand-authored: an authored list drifts silently and a missed entry leaks an
# evaluation value into training. (An earlier hand-written version of this
# constant invented 14 values that are not in the suite and omitted 14 that are —
# three of which appear in the generation pools below.)
#
# The snapshot exists only for offline reproducibility and is asserted equal to
# the live suite by tests; the live derivation always wins at build time.
_FROZEN_EVAL_SNAPSHOT: tuple[str, ...] = (
    "acetaminophen", "allergic rhinitis", "allergies", "amlodipine", "anxiety",
    "atorvastatin", "back", "cervical strain", "chest", "cough", "dizziness",
    "fatigue", "fever", "gastroenteritis", "headache", "hypertension",
    "ibuprofen", "levothyroxine", "lisinopril", "medication", "metformin",
    "migraine", "muscle strain", "nausea", "neck", "omeprazole", "pain",
    "pressure", "rash", "sinusitis", "stiffness", "swelling", "tension headache",
    "tingling", "tired", "viral syndrome", "weakness",
)


def _live_frozen_eval_values() -> frozenset[str] | None:
    """Read the reserved vocabulary from the frozen suite itself."""
    try:
        from nanoscribe.campaign_datasets import campaign_cases

        return frozenset(
            (spec.raw_value or "").lower()
            for case in campaign_cases("p1_screening_eval_v1")
            for spec in case.atom_specs
            if spec.raw_value
        )
    except Exception:  # pragma: no cover - suite unavailable in minimal envs
        return None

SYMPTOMS: tuple[str, ...] = (
    "sore throat", "joint ache", "shortness of breath", "blurred vision",
    "heartburn", "chills", "numbness", "palpitations", "constipation",
    "insomnia", "wheezing", "bruising", "night sweats", "dry mouth",
    "muscle cramps", "ringing in the ears", "loss of appetite", "cold hands",
    "hoarseness", "itchy eyes", "shoulder stiffness", "jaw tightness",
    "leg heaviness", "morning stiffness", "throbbing temple", "burning feet",
    "hip soreness", "wrist weakness", "scalp tenderness", "ankle puffiness",
    "eye strain", "tingling fingers", "shallow breathing", "sour taste",
    "clammy skin", "restless legs", "hand tremors", "foggy thinking",
)

ASSESSMENTS: tuple[str, ...] = (
    "reactive airway", "tension myalgia", "contact dermatitis",
    "orthostatic hypotension", "bacterial pharyngitis", "cluster headache",
    "GERD flare", "plantar fasciitis", "costochondritis", "viral gastritis",
    "seborrheic dermatitis", "cervical strain", "allergic rhinitis",
    "lumbar sprain", "rotator cuff irritation", "carpal tunnel syndrome",
    "iron deficiency", "vitamin D insufficiency", "temporomandibular strain",
    "peripheral neuropathy", "benign positional vertigo", "chronic sinus congestion",
)

MEDICATIONS: tuple[str, ...] = (
    "losartan", "pantoprazole", "rosuvastatin", "albuterol", "gabapentin",
    "hydrochlorothiazide", "montelukast", "duloxetine", "famotidine",
    "levothyroxine", "escitalopram", "tamsulosin", "clopidogrel", "meloxicam",
    "cetirizine", "spironolactone", "bupropion", "ranitidine", "naproxen",
    "cyclobenzaprine", "fluticasone", "carvedilol", "ondansetron", "topiramate",
)

_POOLS: dict[str, tuple[str, ...]] = {
    "symptom": SYMPTOMS,
    "assessment": ASSESSMENTS,
    "medication": MEDICATIONS,
}


def forbidden_values() -> frozenset[str]:
    """Values reserved to the frozen screening suite.

    Prefers the live suite; falls back to the pinned snapshot only when the suite
    cannot be imported. Union of both, so a fallback can never widen generation.
    """
    live = _live_frozen_eval_values()
    snapshot = frozenset(v.lower() for v in _FROZEN_EVAL_SNAPSHOT)
    return snapshot if live is None else (live | snapshot)


def _bucket(value: str, seed_namespace: str) -> Partition:
    """Assign a value to a partition deterministically.

    Hash-based so the split is stable across builds and machines: adding values
    never reshuffles existing ones, which keeps a DEV value from silently
    becoming a TRAIN value between corpus revisions.

    80/10/10 TRAIN/DEV/INTERNAL_TEST.
    """
    digest = hashlib.sha256(f"{seed_namespace}:{value.lower()}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return Partition.TRAIN
    if bucket < 90:
        return Partition.DEV
    return Partition.INTERNAL_TEST


def value_partitions(seed_namespace: str = "native_corpus_v1") -> dict[str, Partition]:
    """Map every generatable value to its partition."""
    out: dict[str, Partition] = {}
    reserved = forbidden_values()
    for pool in _POOLS.values():
        for value in pool:
            if value.lower() in reserved:
                continue
            out[value] = _bucket(value, seed_namespace)
    return out


def values_for(
    kind: str, partition: Partition, seed_namespace: str = "native_corpus_v1"
) -> list[str]:
    """Values of `kind` assigned to `partition`, excluding reserved eval values."""
    reserved = forbidden_values()
    return [
        v
        for v in _POOLS[kind]
        if v.lower() not in reserved and _bucket(v, seed_namespace) is partition
    ]


# Compositional axes. A curated list of ~80 values cannot test OPEN-vocabulary
# transport: the model can memorise it. Composing modifier x site x quality
# yields thousands of distinct, never-memorised surface forms while keeping
# generation deterministic and the train/dev/test split hash-stable.
_MODIFIERS: tuple[str, ...] = (
    "dull", "sharp", "throbbing", "aching", "burning", "stabbing", "cramping",
    "tingling", "pressing", "pulsing", "gnawing", "searing", "tight", "heavy",
    "prickling", "radiating", "intermittent", "constant", "worsening", "dull-edged",
)
_SITES: tuple[str, ...] = (
    "left shoulder", "right shoulder", "lower back", "upper back", "left knee",
    "right knee", "left elbow", "right elbow", "left hip", "right hip",
    "right ankle", "left ankle", "right wrist", "left wrist", "mid spine",
    "right calf", "left calf", "right forearm", "left forearm", "right thigh",
    "left thigh", "upper abdomen", "lower abdomen", "right temple", "left temple",
)
_QUALITIES: tuple[str, ...] = (
    "discomfort", "soreness", "ache", "sensation", "tightness", "irritation",
    "throb", "twinge", "heaviness", "numbness",
)


def composed_values(seed_namespace: str = "native_corpus_v1") -> tuple[str, ...]:
    """Deterministic compositional value space (modifier + site + quality)."""
    reserved = forbidden_values()
    out: list[str] = []
    for modifier in _MODIFIERS:
        for site in _SITES:
            for quality in _QUALITIES:
                value = f"{modifier} {site} {quality}"
                if value.lower() not in reserved:
                    out.append(value)
    return tuple(out)


def composed_values_for(
    partition: Partition, seed_namespace: str = "native_corpus_v1", limit: int | None = None
) -> list[str]:
    """Compositional values assigned to `partition` by the same stable hash."""
    vals = [v for v in composed_values(seed_namespace) if _bucket(v, seed_namespace) is partition]
    return vals[:limit] if limit is not None else vals


def all_kinds() -> tuple[str, ...]:
    return tuple(_POOLS)


def pool_overlap(values: Iterable[str]) -> set[str]:
    """Any of `values` that collide with the reserved frozen-eval vocabulary."""
    reserved = forbidden_values()
    return {v for v in values if v.lower() in reserved}
