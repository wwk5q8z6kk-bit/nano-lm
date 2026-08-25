"""Layer C — adversarial / selectivity curriculum.

These cases exist to punish the failure modes a fluent model falls into: copying
a plausible nearby span, treating silence as denial, resolving a conflict by
picking one side silently, and asserting an inference the source never states.

Every target here is deliberately conservative. When the source does not support
a current patient-asserted value, the correct answer is NOT_MENTIONED rather than
a confident guess.
"""

from __future__ import annotations

from collections.abc import Iterator

from nanoscribe.adapters import Speaker
from nanoscribe.native.corpus import vocab
from nanoscribe.native.corpus.generators import _example
from nanoscribe.native.corpus.schema import Axis, CorpusExample, Layer, Partition


def generate_adversarial(
    partition: Partition, *, limit_composed: int = 400
) -> Iterator[CorpusExample]:
    pairs: list[tuple[str, str]] = []
    for kind in vocab.all_kinds():
        pairs.extend((v, kind) for v in vocab.values_for(kind, partition))
    pairs.extend((v, "composed") for v in vocab.composed_values_for(partition, limit=limit_composed))
    tag = partition.value.lower()

    for i, (value, kind) in enumerate(pairs):
        other = pairs[(i + 5) % len(pairs)][0]

        # --- WRONG SOURCE: value appears, but in a prior-records aside --------
        yield _example(
            enc_id=f"{tag}-adv-src-{i:06d}",
            value=value,
            kind=kind,
            turns=(
                (Speaker.CLINICIAN, f"Your outside records mention {value}."),
                (Speaker.PATIENT, "That wasn't me, that was my brother's chart."),
            ),
            target="NOT_MENTIONED",
            axes=(Axis.WRONG_SOURCE, Axis.NOT_MENTIONED, Axis.SPURIOUS_TEMPTATION),
            template_id="adv_wrong_source_records",
            partition=partition,
            layer=Layer.ADVERSARIAL,
            notes="value present in a non-patient source",
        )

        # --- CONFLICT: patient contradicts themselves; do not silently resolve
        if i % 2 == 0:
            yield _example(
                enc_id=f"{tag}-adv-conf-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "Tell me about it."),
                    (Speaker.PATIENT, f"I have {value}. Actually no, that was last year."),
                ),
                target=f"UNCERTAIN: {value}",
                axes=(Axis.CONFLICT, Axis.UNCERTAINTY, Axis.TEMPORALITY),
                template_id="adv_self_contradiction",
                partition=partition,
                layer=Layer.ADVERSARIAL,
                notes="assertion retracted in the same turn; uncertainty must survive",
            )

        # --- SUPERSPAN: correct value embedded in a longer clause -------------
        if i % 3 == 1:
            yield _example(
                enc_id=f"{tag}-adv-span-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "Anything else?"),
                    (
                        Speaker.PATIENT,
                        f"Ever since the move I've had {value} on and off most mornings.",
                    ),
                ),
                target=f"ASSERTED: {value}",
                axes=(Axis.SUPERSPAN, Axis.EXACT_COPY, Axis.ASSERTION),
                template_id="adv_superspan_clause",
                partition=partition,
                layer=Layer.ADVERSARIAL,
                notes="must copy the value, not the surrounding clause",
            )

        # --- SILENCE != DENIAL ------------------------------------------------
        if i % 4 == 2:
            yield _example(
                enc_id=f"{tag}-adv-silence-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, f"And {value}?"),
                    (Speaker.PATIENT, "Let's talk about something else."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.NOT_MENTIONED, Axis.ABSTENTION, Axis.NEGATION),
                template_id="adv_silence_not_denial",
                partition=partition,
                layer=Layer.ADVERSARIAL,
                notes="unanswered question is not a denial",
            )

        # --- PLAUSIBLE UNSUPPORTED INFERENCE ----------------------------------
        if i % 5 == 3:
            yield _example(
                enc_id=f"{tag}-adv-infer-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "What have you noticed?"),
                    (Speaker.PATIENT, f"Something a lot like {other}, but not {value}."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.SPURIOUS_TEMPTATION, Axis.NOT_MENTIONED, Axis.NEGATION),
                template_id="adv_plausible_unsupported",
                partition=partition,
                layer=Layer.ADVERSARIAL,
                notes="a similar value is asserted; the queried one is explicitly excluded",
            )

        # --- FUTURE PLAN != COMPLETED EVENT -----------------------------------
        if i % 6 == 4:
            yield _example(
                enc_id=f"{tag}-adv-future-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "What's the plan?"),
                    (Speaker.PATIENT, f"We're going to start {value} next month."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.TEMPORALITY, Axis.NOT_MENTIONED, Axis.SPURIOUS_TEMPTATION),
                template_id="adv_future_plan",
                partition=partition,
                layer=Layer.ADVERSARIAL,
                notes="planned, not current or completed",
            )
