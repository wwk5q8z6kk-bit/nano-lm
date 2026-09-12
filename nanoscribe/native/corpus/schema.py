"""Schema and verdict vocabulary for the Native corpus factory.

The prior training corpus (`nano.distill.train.v1`, 96 rows / 4,516 chars) is
retained ONLY as NATIVE_UNIT_OVERFIT_FIXTURE. It must never support architecture
ranking, scale conclusions, held-out capability claims, promotion decisions, or
student-vs-Native comparison.

Note the loader contract: nanoscribe/native/data.load_train_examples IGNORES the
file contents and regenerates from code whenever schema == "nano.distill.train.v1".
A larger corpus written under that schema would be silently discarded, so the
factory emits CORPUS_SCHEMA instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

CORPUS_SCHEMA = "nano.native.corpus.v1"
FIXTURE_SCHEMA = "nano.distill.train.v1"
FIXTURE_LABEL = "NATIVE_UNIT_OVERFIT_FIXTURE"


class Partition(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    INTERNAL_TEST = "INTERNAL_TEST"
    FROZEN_SCREENING_EVAL = "FROZEN_SCREENING_EVAL"


class Axis(str, Enum):
    """Capability axes. Every generated example declares the axes it exercises.

    Coverage is reported per axis so a corpus cannot silently under-serve a
    capability the evaluation later measures.
    """

    EXACT_COPY = "exact_copy"
    ASSERTION = "assertion"
    NEGATION = "negation"
    UNCERTAINTY = "uncertainty"
    NOT_MENTIONED = "not_mentioned"
    SPEAKER = "speaker"
    EXPERIENCER = "experiencer"
    TEMPORALITY = "temporality"
    MULTI_MENTION = "multi_mention"
    CONFLICT = "conflict"
    WRONG_SOURCE = "wrong_source"
    SUPERSPAN = "superspan"
    ABSTENTION = "abstention"
    SPURIOUS_TEMPTATION = "spurious_temptation"
    OPEN_VOCAB = "open_vocab"


class Layer(str, Enum):
    """Corpus layers from the data program."""

    MECHANISM = "B_mechanism_curriculum"
    ADVERSARIAL = "C_adversarial_curriculum"
    TEACHER_VERIFIED = "D_verified_teacher_data"


class Provenance(str, Enum):
    SYNTHETIC_GOLD = "SYNTHETIC_GOLD"
    TEACHER_VERIFIED = "TEACHER_VERIFIED"
    TEACHER_UNVERIFIED = "TEACHER_UNVERIFIED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class CorpusExample:
    """One training row plus the metadata needed to audit it.

    prompt/target/encounter_id/atom_id match what
    nanoscribe.native.data.load_train_examples consumes, so a built corpus is
    directly trainable. The remaining fields exist so coverage, dedupe, leakage
    and partitioning are auditable rather than assumed.
    """

    encounter_id: str
    atom_id: str
    prompt: str
    target: str
    raw_value: str
    axes: tuple[Axis, ...]
    layer: Layer
    template_id: str
    partition: Partition
    provenance: Provenance = Provenance.SYNTHETIC_GOLD
    seed: int = 0
    notes: str = ""

    def to_entry(self) -> dict[str, object]:
        return {
            "encounter_id": self.encounter_id,
            "atom_id": self.atom_id,
            "prompt": self.prompt,
            "target": self.target,
            "raw_value": self.raw_value,
            "axes": [a.value for a in self.axes],
            "layer": self.layer.value,
            "template_id": self.template_id,
            "partition": self.partition.value,
            "provenance": self.provenance.value,
            "seed": self.seed,
            "notes": self.notes,
        }


@dataclass
class CorpusBuild:
    corpus_id: str
    revision: str
    seed_namespace: str
    examples: list[CorpusExample] = field(default_factory=list)

    def by_partition(self, partition: Partition) -> list[CorpusExample]:
        return [e for e in self.examples if e.partition is partition]
