"""Layer B — P1 mechanism curriculum.

Generates span-port training examples across capability axes using the CANONICAL
prompt builder (nanoscribe.prompt.build_span_port_prompt) so training prompts are
byte-identical in form to what evaluation and deployment produce. Hand-rolling a
second prompt format here would train the model on a surface the evaluator never
shows it.

Everything is deterministic: given a seed namespace and index, the same example
is produced on any machine.
"""

from __future__ import annotations

from collections.abc import Iterator

from nanoscribe.adapters import AtomSpec, AtomType, Speaker
from nanoscribe.native.corpus import vocab
from nanoscribe.native.corpus.schema import Axis, CorpusExample, Layer, Partition
from nanoscribe.prompt import build_span_port_prompt
from nanoscribe.distill_train_suite import assemble_source

_ATOM_TYPE = {
    "symptom": AtomType.SYMPTOM,
    "assessment": AtomType.ASSESSMENT,
    "medication": AtomType.MEDICATION,
    "composed": AtomType.SYMPTOM,
}

# Multiple phrasings per intent so the model cannot memorise one surface form.
_ASSERT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("assert_have", "I have {value}."),
    ("assert_getting", "I've been getting {value}."),
    ("assert_noticed", "I noticed {value} recently."),
    ("assert_dealing", "I'm dealing with {value}."),
    ("assert_bothering", "{value} has been bothering me."),
    ("assert_started", "It started as {value}."),
)
_DENY_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("deny_no", "No {value} at all."),
    ("deny_never", "I've never had {value}."),
    ("deny_not", "I do not have {value}."),
    ("deny_denies", "Nothing like {value}."),
)
_UNCERTAIN_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("unc_maybe", "Maybe {value}, I'm not sure."),
    ("unc_think", "I think it might be {value}."),
    ("unc_could", "Could be {value}, hard to say."),
    ("unc_possibly", "Possibly {value}, but I can't tell."),
)
_OPENERS: tuple[str, ...] = (
    "What symptoms are you having?",
    "What brings you in today?",
    "Tell me what's been going on.",
    "How have you been feeling?",
    "Any complaints since last time?",
)


def _example(
    *,
    enc_id: str,
    value: str,
    kind: str,
    turns: tuple[tuple[Speaker, str], ...],
    target: str,
    axes: tuple[Axis, ...],
    template_id: str,
    partition: Partition,
    layer: Layer = Layer.MECHANISM,
    notes: str = "",
) -> CorpusExample:
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=_ATOM_TYPE[kind],
        raw_value=value,
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(f"src-{enc_id}", turns)
    return CorpusExample(
        encounter_id=enc_id,
        atom_id=spec.atom_id,
        prompt=build_span_port_prompt(source, spec),
        target=target,
        raw_value=value,
        axes=axes,
        layer=layer,
        template_id=template_id,
        partition=partition,
        notes=notes,
    )


def _values_for(partition: Partition, limit_composed: int) -> list[tuple[str, str]]:
    """(value, kind) pairs available to a partition."""
    pairs: list[tuple[str, str]] = []
    for kind in vocab.all_kinds():
        pairs.extend((v, kind) for v in vocab.values_for(kind, partition))
    pairs.extend(
        (v, "composed") for v in vocab.composed_values_for(partition, limit=limit_composed)
    )
    return pairs


def generate_mechanism(
    partition: Partition, *, limit_composed: int = 1200
) -> Iterator[CorpusExample]:
    """Yield the mechanism curriculum for one partition.

    Covers assertion, negation, uncertainty, not-mentioned, speaker, experiencer,
    temporality and multi-mention. Open-vocabulary coverage comes from the
    compositional value space, which the model cannot memorise.
    """
    pairs = _values_for(partition, limit_composed)
    tag = partition.value.lower()

    for i, (value, kind) in enumerate(pairs):
        # --- ASSERTED (exact copy + assertion + open vocab) -------------------
        tid, phrasing = _ASSERT_TEMPLATES[i % len(_ASSERT_TEMPLATES)]
        opener = _OPENERS[i % len(_OPENERS)]
        yield _example(
            enc_id=f"{tag}-assert-{i:06d}",
            value=value,
            kind=kind,
            turns=((Speaker.CLINICIAN, opener), (Speaker.PATIENT, phrasing.format(value=value))),
            target=f"ASSERTED: {value}",
            axes=(Axis.EXACT_COPY, Axis.ASSERTION, Axis.OPEN_VOCAB),
            template_id=tid,
            partition=partition,
        )

        # --- DENIED (negation must not be read as assertion) ------------------
        if i % 3 == 0:
            tid, phrasing = _DENY_TEMPLATES[i % len(_DENY_TEMPLATES)]
            yield _example(
                enc_id=f"{tag}-deny-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, f"Any {value}?"),
                    (Speaker.PATIENT, phrasing.format(value=value)),
                ),
                target=f"DENIED: {value}",
                axes=(Axis.EXACT_COPY, Axis.NEGATION, Axis.OPEN_VOCAB),
                template_id=tid,
                partition=partition,
            )

        # --- UNCERTAIN (uncertainty must be preserved, not collapsed) ---------
        if i % 4 == 1:
            tid, phrasing = _UNCERTAIN_TEMPLATES[i % len(_UNCERTAIN_TEMPLATES)]
            yield _example(
                enc_id=f"{tag}-unc-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "What do you think is going on?"),
                    (Speaker.PATIENT, phrasing.format(value=value)),
                ),
                target=f"UNCERTAIN: {value}",
                axes=(Axis.EXACT_COPY, Axis.UNCERTAINTY, Axis.OPEN_VOCAB),
                template_id=tid,
                partition=partition,
            )

        # --- NOT_MENTIONED (silence is not a negative fact) -------------------
        if i % 5 == 2:
            other = pairs[(i + 7) % len(pairs)][0]
            yield _example(
                enc_id=f"{tag}-absent-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, _OPENERS[(i + 1) % len(_OPENERS)]),
                    (Speaker.PATIENT, f"Mostly just {other}."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.NOT_MENTIONED, Axis.ABSTENTION),
                template_id="absent_other_topic",
                partition=partition,
                notes="target value never appears; a distractor value does",
            )

        # --- MULTI_MENTION (choose the right mention of two) ------------------
        if i % 6 == 3:
            other = pairs[(i + 11) % len(pairs)][0]
            yield _example(
                enc_id=f"{tag}-multi-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "Walk me through everything."),
                    (Speaker.PATIENT, f"First {other}, and separately {value}."),
                ),
                target=f"ASSERTED: {value}",
                axes=(Axis.MULTI_MENTION, Axis.EXACT_COPY, Axis.OPEN_VOCAB),
                template_id="multi_two_mentions",
                partition=partition,
                notes="two candidate values present; must select the queried one",
            )

        # --- EXPERIENCER (family history is not patient history) --------------
        if i % 7 == 4:
            yield _example(
                enc_id=f"{tag}-family-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "Any family history?"),
                    (Speaker.PATIENT, f"My mother had {value}, but I haven't."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.EXPERIENCER, Axis.NOT_MENTIONED, Axis.SPURIOUS_TEMPTATION),
                template_id="experiencer_family",
                partition=partition,
                notes="value present in text but attributed to a relative",
            )

        # --- TEMPORALITY (historical is not current) --------------------------
        if i % 8 == 5:
            yield _example(
                enc_id=f"{tag}-hist-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, "Anything currently?"),
                    (Speaker.PATIENT, f"I had {value} years ago, nothing now."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.TEMPORALITY, Axis.NOT_MENTIONED, Axis.SPURIOUS_TEMPTATION),
                template_id="temporality_historical",
                partition=partition,
                notes="value present but resolved in the past",
            )

        # --- SPEAKER (clinician wording is not patient wording) ---------------
        if i % 9 == 6:
            yield _example(
                enc_id=f"{tag}-speaker-{i:06d}",
                value=value,
                kind=kind,
                turns=(
                    (Speaker.CLINICIAN, f"Some patients report {value}."),
                    (Speaker.PATIENT, "Not me."),
                ),
                target="NOT_MENTIONED",
                axes=(Axis.SPEAKER, Axis.WRONG_SOURCE, Axis.NOT_MENTIONED),
                template_id="speaker_clinician_only",
                partition=partition,
                notes="value spoken by clinician, not the patient",
            )
