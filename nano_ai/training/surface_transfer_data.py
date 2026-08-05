"""Deterministic H4 surface-transfer data for Nano's native scribe target.

H4 changes only the training distribution.  It preserves H3's five-field
conversation structure, paired state mutations, target grammar, and record
schema while replacing the shortcut-prone surface family with independently
authored fit and training-only calibration surfaces.

Gold spans are created from the rendered turns themselves.  This module does
not call or extend the deterministic scribe baseline, and it has no benchmark
or sealed-confirmation dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
)
from nano_ai.training.state_span_data import (
    FORBIDDEN_HISTORICAL_SENTINELS,
    STATE_VARIANTS,
    TARGET_GRAMMAR_VERSION,
    VARIANTS,
    StateSpanExample,
    canonical_json_bytes,
    encode_state_span,
    generate_split,
    supported_value_sets,
)

GENERATOR_VERSION = "nano.surface-transfer-dataset.v1"
MANIFEST_SCHEMA_VERSION = "nano.surface-transfer-manifest.v1"
TRAINING_RECIPE_VERSION = "nano-evidence-query-data-only-v1"
FIT_PARTITION = "fit"
CALIBRATION_PARTITION = "calibration"
PARTITIONS = (FIT_PARTITION, CALIBRATION_PARTITION)
FIT_GENERATOR_SEED = 20260807
CALIBRATION_GENERATOR_SEED = 20260808
LEXICON_PARTITION_SEED = 20260809
FIT_WORLDS = 2_800
CALIBRATION_WORLDS = 200
VARIANTS_PER_WORLD = 4
FIT_RECORDS = FIT_WORLDS * VARIANTS_PER_WORLD
CALIBRATION_RECORDS = CALIBRATION_WORLDS * VARIANTS_PER_WORLD
TRAINING_SEEDS = (20260805, 20260806)
BATCH_SIZE = 32
EPOCHS = 3
STEPS_PER_EPOCH = 350
STEPS_PER_SEED = 1_050

_PARTITION_SEEDS = {
    FIT_PARTITION: FIT_GENERATOR_SEED,
    CALIBRATION_PARTITION: CALIBRATION_GENERATOR_SEED,
}
_PARTITION_WORLDS = {
    FIT_PARTITION: FIT_WORLDS,
    CALIBRATION_PARTITION: CALIBRATION_WORLDS,
}
_LABELS = {
    FieldName.CHIEF_COMPLAINT: "CC",
    FieldName.DURATION: "DUR",
    FieldName.SEVERITY: "SEV",
    FieldName.MEDICATION: "MED",
    FieldName.ALLERGY: "ALG",
}
_OPEN_FIELDS = tuple(
    field for field in FIELD_ORDER if field is not FieldName.SEVERITY
)
_BOUNDARIES = ("value_only", "value_first", "value_medial", "value_final")

# Three openers x four field-specific cores give exactly twelve fit question
# realizations per field.  Calibration uses an independent two x two family.
_FIT_QUESTION_OPENERS = (
    "Please tell me: ",
    "For this note, ",
    "Let's clarify: ",
)
_CALIBRATION_QUESTION_OPENERS = (
    "Could you specify: ",
    "One detail for the record: ",
)
_FIT_QUESTION_CORES: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.CHIEF_COMPLAINT: (
        "what problem led to this visit?",
        "which symptom needs attention today?",
        "what concern should I record first?",
        "what is bothering you most right now?",
    ),
    FieldName.DURATION: (
        "what length of time applies?",
        "how long has the problem been present?",
        "what duration should I record?",
        "how much time has passed since it began?",
    ),
    FieldName.SEVERITY: (
        "what intensity best describes it?",
        "how strong does the problem feel?",
        "what level of intensity should I note?",
        "how intense is the symptom now?",
    ),
    FieldName.MEDICATION: (
        "what treatment have you taken for it?",
        "which medicine have you used for this?",
        "what medication should I record?",
        "have you used a treatment for the symptom?",
    ),
    FieldName.ALLERGY: (
        "what allergy should I record?",
        "which substance causes an allergic reaction?",
        "what are you allergic to?",
        "is there an allergy I should note?",
    ),
}
_CALIBRATION_QUESTION_CORES: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.CHIEF_COMPLAINT: (
        "name the main reason for today's visit?",
        "identify the symptom you came about?",
    ),
    FieldName.DURATION: (
        "state the time span involved?",
        "identify when the symptom began?",
    ),
    FieldName.SEVERITY: (
        "state the symptom's intensity?",
        "identify its current strength?",
    ),
    FieldName.MEDICATION: (
        "name anything used to treat it?",
        "identify the medicine taken for this?",
    ),
    FieldName.ALLERGY: (
        "name any substance allergy?",
        "identify an item that triggers allergy?",
    ),
}

# Fit has three realizations at each of four evidence-boundary positions.
# Calibration has one independently authored realization at each position.
_FIT_SUPPORTED_ANSWERS: Mapping[str, tuple[str, ...]] = {
    "value_only": ("{value}", "\"{value}\"", "{value}."),
    "value_first": (
        "{value} is the main answer.",
        "{value}, that's correct.",
        "{value} would be my response.",
    ),
    "value_medial": (
        "The specific answer, {value}, is what I mean.",
        "For this detail I would put {value} in the note.",
        "What applies here is {value}, if that helps.",
    ),
    "value_final": (
        "My answer for that is {value}",
        "You can record this as {value}",
        "The clearest response I can give is {value}",
    ),
}
_CALIBRATION_SUPPORTED_ANSWERS: Mapping[str, tuple[str, ...]] = {
    "value_only": ("[{value}]",),
    "value_first": ("{value} — that is my reply.",),
    "value_medial": ("For the record, the detail is {value} today.",),
    "value_final": ("I would document the answer as {value}",),
}

# Every denial is accepted by the frozen Nano v0 contract.  The two
# partitions are exact-text disjoint and do not reuse known-development forms.
_FIT_DENIALS: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.MEDICATION: (
        "No medication.",
        "No medications.",
        "No medicine.",
        "No meds.",
        "I deny medication.",
        "I deny taking medicine.",
        "I denied medications.",
        "I denied taking meds.",
    ),
    FieldName.ALLERGY: (
        "No allergy.",
        "No known allergy.",
        "No known allergies.",
        "I deny allergy.",
        "I deny any allergy.",
        "I denied allergies.",
        "I denied any allergies.",
        "I denied allergy.",
    ),
}
_CALIBRATION_DENIALS: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.MEDICATION: (
        "No, nothing.",
        "No nothing yet!",
        "I denied taking medicine.",
        "I deny taking medications.",
    ),
    FieldName.ALLERGY: (
        "No allergies!",
        "Not that I know of!",
        "I denied any allergy.",
        "I deny allergies.",
    ),
}
_FIT_UNCERTAIN = (
    "I do not have a definite answer.",
    "That detail is unclear to me.",
    "I am unable to confirm that.",
    "I cannot answer that reliably.",
    "My memory is unclear on that point.",
    "I would only be guessing.",
    "I do not know the answer.",
    "That information is uncertain.",
    "I cannot verify that detail.",
    "I am not confident about it.",
    "I cannot provide a firm response.",
    "I do not remember that clearly.",
)
_CALIBRATION_UNCERTAIN = (
    "I cannot give a reliable detail.",
    "I lack confidence in that answer.",
    "That point is not clear in my memory.",
    "I would not want to guess about that.",
    "I cannot confirm the requested fact.",
    "I do not have dependable information for it.",
)


@dataclass(frozen=True, slots=True)
class _RenderedLine:
    speaker: Literal["Doctor", "Patient"]
    text: str
    field: FieldName
    evidence_key: str | None = None
    evidence_text: str | None = None


@dataclass(frozen=True, slots=True)
class _World:
    partition: str
    index: int
    target_field: FieldName
    values: Mapping[FieldName, str | None]
    questions: Mapping[FieldName, str]
    answers: Mapping[FieldName, str]
    answer_boundaries: Mapping[FieldName, str | None]


@dataclass(frozen=True, slots=True)
class _Trace:
    question_templates: tuple[tuple[str, str], ...]
    answer_templates: tuple[tuple[str, str, str], ...]
    denial_templates: tuple[tuple[str, str], ...]
    uncertainty_template: tuple[str, str] | None
    correction_template: tuple[str, str, str] | None
    values: tuple[tuple[str, str], ...]
    correction_value: tuple[str, str] | None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: str, role: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _stable_seed(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _permuted(items: Sequence[str], *seed_parts: object) -> tuple[str, ...]:
    values = list(items)
    random.Random(_stable_seed(*seed_parts)).shuffle(values)
    return tuple(values)


@cache
def _balanced_index_schedule(
    partition: str,
    field: FieldName,
    role: str,
    observations: int,
    categories: int,
) -> tuple[int, ...]:
    """Return an independently seeded schedule with exact floor/ceil balance."""

    if observations <= 0 or categories <= 0:
        raise ValueError("balanced schedule dimensions must be positive")
    schedule = [index % categories for index in range(observations)]
    random.Random(
        _stable_seed(_PARTITION_SEEDS[partition], field.value, role, "schedule")
    ).shuffle(schedule)
    return tuple(schedule)


def _question_bank(partition: str, field: FieldName) -> tuple[str, ...]:
    if partition == FIT_PARTITION:
        openers = _FIT_QUESTION_OPENERS
        cores = _FIT_QUESTION_CORES[field]
    elif partition == CALIBRATION_PARTITION:
        openers = _CALIBRATION_QUESTION_OPENERS
        cores = _CALIBRATION_QUESTION_CORES[field]
    else:
        raise ValueError("unknown H4 partition")
    return tuple(opener + core for opener in openers for core in cores)


def _answer_bank(partition: str) -> Mapping[str, tuple[str, ...]]:
    if partition == FIT_PARTITION:
        return _FIT_SUPPORTED_ANSWERS
    if partition == CALIBRATION_PARTITION:
        return _CALIBRATION_SUPPORTED_ANSWERS
    raise ValueError("unknown H4 partition")


def _denial_bank(partition: str, field: FieldName) -> tuple[str, ...]:
    if partition == FIT_PARTITION:
        return _FIT_DENIALS[field]
    if partition == CALIBRATION_PARTITION:
        return _CALIBRATION_DENIALS[field]
    raise ValueError("unknown H4 partition")


def _uncertainty_bank(partition: str) -> tuple[str, ...]:
    if partition == FIT_PARTITION:
        return _FIT_UNCERTAIN
    if partition == CALIBRATION_PARTITION:
        return _CALIBRATION_UNCERTAIN
    raise ValueError("unknown H4 partition")


def _lexicons() -> Mapping[str, Mapping[FieldName, tuple[str, ...]]]:
    source = supported_value_sets("train")
    expected = {
        FieldName.CHIEF_COMPLAINT: (192, 48),
        FieldName.DURATION: (12, 3),
        FieldName.MEDICATION: (48, 16),
        FieldName.ALLERGY: (48, 16),
    }
    fit: dict[FieldName, tuple[str, ...]] = {}
    calibration: dict[FieldName, tuple[str, ...]] = {}
    for field in FIELD_ORDER:
        values = sorted(source[field], key=lambda item: (item.casefold(), item))
        if field is FieldName.SEVERITY:
            fit[field] = tuple(values)
            calibration[field] = tuple(values)
            continue
        shuffled = list(values)
        random.Random(_stable_seed(LEXICON_PARTITION_SEED, field.value)).shuffle(
            shuffled
        )
        fit_count, calibration_count = expected[field]
        if len(shuffled) != fit_count + calibration_count:
            raise AssertionError(f"unexpected source lexicon size for {field.value}")
        fit[field] = tuple(sorted(shuffled[:fit_count]))
        calibration[field] = tuple(sorted(shuffled[fit_count:]))
    return {FIT_PARTITION: fit, CALIBRATION_PARTITION: calibration}


def partition_lexicons() -> Mapping[str, Mapping[FieldName, tuple[str, ...]]]:
    """Expose the frozen H4 value partition without generating records."""

    return _lexicons()


def _absent_rank(field: FieldName, index: int) -> int:
    period = 4 if field is FieldName.MEDICATION else 5
    if index % period:
        raise AssertionError("absent rank requested for a supported world")
    return index // period


def _select_question(
    partition: str,
    field: FieldName,
    index: int,
    value: str | None,
) -> tuple[str, str]:
    bank = _permuted(
        _question_bank(partition, field),
        _PARTITION_SEEDS[partition],
        field.value,
        "question",
    )
    if value is None:
        rank = _absent_rank(field, index)
        observations = _absent_world_count(partition, field)
        template_index = _balanced_index_schedule(
            partition,
            field,
            "absent-question",
            observations,
            len(bank),
        )[rank]
    else:
        rank = _value_occurrence_rank(partition, field, index)
        offset = _stable_seed(
            _PARTITION_SEEDS[partition], field.value, value, "question-offset"
        ) % len(bank)
        template_index = (offset + rank) % len(bank)
    return bank[template_index], f"q{template_index:02d}"


def _select_answer_template(
    partition: str,
    field: FieldName,
    rank: int,
    *,
    role: str,
    value: str | None = None,
) -> tuple[str, str, str]:
    choices = [
        (template, boundary, f"{boundary}:{variation}")
        for boundary in _BOUNDARIES
        for variation, template in enumerate(_answer_bank(partition)[boundary])
    ]
    random.Random(
        _stable_seed(
            _PARTITION_SEEDS[partition], field.value, role, "answer-choices"
        )
    ).shuffle(choices)
    if role == "original":
        if value is None:
            raise AssertionError("original supported answer requires a value")
        offset = _stable_seed(
            _PARTITION_SEEDS[partition], field.value, value, "answer-offset"
        ) % len(choices)
        multiplier = 5 if len(choices) == 12 else 3
        choice_index = (offset + multiplier * rank) % len(choices)
    elif role == "correction":
        choice_index = _balanced_index_schedule(
            partition,
            field,
            "correction-answer",
            _PARTITION_WORLDS[partition] // len(FIELD_ORDER),
            len(choices),
        )[rank]
    else:
        raise ValueError(f"unknown answer-template role: {role}")
    return choices[choice_index]


def _is_absent(field: FieldName, index: int) -> bool:
    return (field is FieldName.MEDICATION and index % 4 == 0) or (
        field is FieldName.ALLERGY and index % 5 == 0
    )


def _absent_world_count(partition: str, field: FieldName) -> int:
    if field not in (FieldName.MEDICATION, FieldName.ALLERGY):
        return 0
    period = 4 if field is FieldName.MEDICATION else 5
    worlds = _PARTITION_WORLDS[partition]
    return (worlds + period - 1) // period


@cache
def _value_assignments(
    partition: str,
    field: FieldName,
) -> tuple[str | None, ...]:
    """Balance each field's values on an independently shuffled schedule."""

    worlds = _PARTITION_WORLDS[partition]
    values = _permuted(
        _lexicons()[partition][field],
        _PARTITION_SEEDS[partition],
        field.value,
        "value-bank",
    )
    assignments: list[str | None] = [None] * worlds
    positions = [
        index for index in range(worlds) if not _is_absent(field, index)
    ]
    schedule = _balanced_index_schedule(
        partition,
        field,
        "value",
        len(positions),
        len(values),
    )
    for index, category in zip(positions, schedule, strict=True):
        assignments[index] = values[category]
    return tuple(assignments)


@cache
def _value_occurrence_ranks(
    partition: str,
    field: FieldName,
) -> tuple[int | None, ...]:
    counts: Counter[str] = Counter()
    ranks: list[int | None] = []
    for value in _value_assignments(partition, field):
        if value is None:
            ranks.append(None)
            continue
        ranks.append(counts[value])
        counts[value] += 1
    return tuple(ranks)


def _value_occurrence_rank(partition: str, field: FieldName, index: int) -> int:
    rank = _value_occurrence_ranks(partition, field)[index]
    if rank is None:
        raise AssertionError("supported occurrence rank requested for absent value")
    return rank


def _value_for(field: FieldName, partition: str, index: int) -> str | None:
    return _value_assignments(partition, field)[index]


def _render_original_answer(
    partition: str,
    field: FieldName,
    index: int,
    value: str | None,
) -> tuple[str, str | None, str]:
    if value is None:
        rank = _absent_rank(field, index)
        bank = _permuted(
            _denial_bank(partition, field),
            _PARTITION_SEEDS[partition],
            field.value,
            "denial",
        )
        template_index = _balanced_index_schedule(
            partition,
            field,
            "denial",
            _absent_world_count(partition, field),
            len(bank),
        )[rank]
        return bank[template_index], None, f"d{template_index:02d}"
    template, boundary, template_id = _select_answer_template(
        partition,
        field,
        _value_occurrence_rank(partition, field, index),
        role="original",
        value=value,
    )
    answer = template.format(value=value)
    if answer.count(value) != 1:
        raise AssertionError("supported answer must contain its value exactly once")
    return answer, boundary, template_id


def _sample_world(partition: str, index: int) -> tuple[_World, _Trace]:
    if partition not in PARTITIONS:
        raise ValueError("unknown H4 partition")
    questions: dict[FieldName, str] = {}
    values: dict[FieldName, str | None] = {}
    answers: dict[FieldName, str] = {}
    boundaries: dict[FieldName, str | None] = {}
    question_trace: list[tuple[str, str]] = []
    answer_trace: list[tuple[str, str, str]] = []
    denial_trace: list[tuple[str, str]] = []
    value_trace: list[tuple[str, str]] = []
    for field in FIELD_ORDER:
        value = _value_for(field, partition, index)
        question, question_id = _select_question(partition, field, index, value)
        answer, boundary, answer_id = _render_original_answer(
            partition, field, index, value
        )
        questions[field] = question
        values[field] = value
        answers[field] = answer
        boundaries[field] = boundary
        question_trace.append((field.value, question_id))
        if value is None:
            denial_trace.append((field.value, answer_id))
        else:
            answer_trace.append((field.value, boundary or "", answer_id))
            value_trace.append((field.value, value))
    world = _World(
        partition=partition,
        index=index,
        target_field=FIELD_ORDER[index % len(FIELD_ORDER)],
        values=values,
        questions=questions,
        answers=answers,
        answer_boundaries=boundaries,
    )
    trace = _Trace(
        question_templates=tuple(question_trace),
        answer_templates=tuple(answer_trace),
        denial_templates=tuple(denial_trace),
        uncertainty_template=None,
        correction_template=None,
        values=tuple(value_trace),
        correction_value=None,
    )
    return world, trace


def _send_flow(
    graph: list[list[list[int]]],
    levels: Sequence[int],
    cursors: list[int],
    node: int,
    sink: int,
    offered: int,
) -> int:
    if node == sink:
        return offered
    while cursors[node] < len(graph[node]):
        edge = graph[node][cursors[node]]
        end, reverse, capacity = edge
        if capacity > 0 and levels[end] == levels[node] + 1:
            sent = _send_flow(
                graph,
                levels,
                cursors,
                end,
                sink,
                min(offered, capacity),
            )
            if sent:
                edge[2] -= sent
                graph[end][reverse][2] += sent
                return sent
        cursors[node] += 1
    return 0


@cache
def _conflict_alternative_schedule(
    partition: str,
    field: FieldName,
) -> tuple[str, ...]:
    """Balance corrections exactly, then match them away from current values."""

    values = _permuted(
        _lexicons()[partition][field],
        _PARTITION_SEEDS[partition],
        field.value,
        "conflict-value-order",
    )
    target_count = _PARTITION_WORLDS[partition] // len(FIELD_ORDER)
    counts = Counter(values[index % len(values)] for index in range(target_count))

    field_offset = FIELD_ORDER.index(field)
    currents = tuple(
        _value_for(
            field,
            partition,
            field_offset + target_rank * len(FIELD_ORDER),
        )
        for target_rank in range(target_count)
    )
    current_groups = tuple(
        sorted(
            set(currents),
            key=lambda item: (item is None, "" if item is None else item),
        )
    )
    current_counts = Counter(currents)

    # Solve the small value-to-current-group transportation problem exactly.
    # Edges on the diagonal are omitted, so supported corrections cannot copy
    # the original value; capacities preserve the precomputed balanced multiset.
    source = 0
    value_start = 1
    group_start = value_start + len(values)
    sink = group_start + len(current_groups)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int) -> list[int]:
        forward = [end, len(graph[end]), capacity]
        reverse = [start, len(graph[start]), 0]
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    edge_by_pair: dict[tuple[str, str | None], list[int]] = {}
    for value_index, value in enumerate(values):
        add_edge(source, value_start + value_index, counts[value])
        for group_index, current in enumerate(current_groups):
            if value == current:
                continue
            edge_by_pair[(value, current)] = add_edge(
                value_start + value_index,
                group_start + group_index,
                target_count,
            )
    for group_index, current in enumerate(current_groups):
        add_edge(group_start + group_index, sink, current_counts[current])

    total_flow = 0
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue = [source]
        for node in queue:
            for end, _reverse, capacity in graph[node]:
                if capacity > 0 and levels[end] < 0:
                    levels[end] = levels[node] + 1
                    queue.append(end)
        if levels[sink] < 0:
            break
        cursors = [0] * len(graph)
        while True:
            sent = _send_flow(
                graph,
                levels,
                cursors,
                source,
                sink,
                target_count - total_flow,
            )
            if not sent:
                break
            total_flow += sent
    if total_flow != target_count:
        raise AssertionError("could not construct balanced conflict derangement")

    assignments: list[str | None] = [None] * target_count
    for current in current_groups:
        positions = [
            index for index, observed in enumerate(currents) if observed == current
        ]
        alternatives: list[str] = []
        for value in values:
            edge = edge_by_pair.get((value, current))
            if edge is not None:
                alternatives.extend([value] * (target_count - edge[2]))
        random.Random(
            _stable_seed(
                _PARTITION_SEEDS[partition], field.value, current, "conflict-match"
            )
        ).shuffle(alternatives)
        if len(alternatives) != len(positions):
            raise AssertionError("conflict matching changed current-group capacity")
        for target_rank, alternative in zip(positions, alternatives, strict=True):
            assignments[target_rank] = alternative
    if any(value is None for value in assignments):
        raise AssertionError("conflict matching left an unassigned value")
    result = tuple(value for value in assignments if value is not None)
    if Counter(result) != counts:
        raise AssertionError("conflict matching changed balanced quotas")
    return result


def _deranged_alternative(
    partition: str,
    field: FieldName,
    current: str | None,
    target_rank: int,
) -> str:
    alternative = _conflict_alternative_schedule(partition, field)[target_rank]
    if current is not None and alternative == current:
        raise AssertionError("conflict correction must differ from supported value")
    return alternative


def _normal_lines(world: _World) -> list[_RenderedLine]:
    lines: list[_RenderedLine] = []
    for field in FIELD_ORDER:
        value = world.values[field]
        evidence_text = world.answers[field] if value is None else value
        lines.extend(
            (
                _RenderedLine("Doctor", world.questions[field], field),
                _RenderedLine(
                    "Patient",
                    world.answers[field],
                    field,
                    evidence_key=f"{field.value}:original",
                    evidence_text=evidence_text,
                ),
            )
        )
    return lines


def _variant_lines(
    world: _World,
    variant: str,
) -> tuple[list[_RenderedLine], _Trace]:
    lines = _normal_lines(world)
    target = world.target_field
    answer_index = FIELD_ORDER.index(target) * 2 + 1
    trace = _Trace((), (), (), None, None, (), None)
    if variant == "normal":
        return lines, trace
    if variant == "missing":
        del lines[answer_index]
        return lines, trace
    target_rank = world.index // len(FIELD_ORDER)
    if variant == "uncertain":
        bank = _permuted(
            _uncertainty_bank(world.partition),
            _PARTITION_SEEDS[world.partition],
            target.value,
            "uncertainty",
        )
        template_index = target_rank % len(bank)
        phrase = bank[template_index]
        lines[answer_index] = _RenderedLine(
            "Patient",
            phrase,
            target,
            evidence_key=f"{target.value}:uncertain",
            evidence_text=phrase,
        )
        return lines, _Trace(
            (),
            (),
            (),
            (target.value, f"u{template_index:02d}"),
            None,
            (),
            None,
        )
    if variant != "conflicting":
        raise ValueError(f"unsupported H4 variant: {variant}")
    alternative = _deranged_alternative(
        world.partition,
        target,
        world.values[target],
        target_rank,
    )
    template, boundary, template_id = _select_answer_template(
        world.partition,
        target,
        target_rank,
        role="correction",
    )
    correction = template.format(value=alternative)
    if correction.count(alternative) != 1:
        raise AssertionError("correction must contain its value exactly once")
    lines.extend(
        (
            _RenderedLine("Doctor", world.questions[target], target),
            _RenderedLine(
                "Patient",
                correction,
                target,
                evidence_key=f"{target.value}:correction",
                evidence_text=alternative,
            ),
        )
    )
    return lines, _Trace(
        (),
        (),
        (),
        None,
        (target.value, boundary, template_id),
        (),
        (target.value, alternative),
    )


def _render(lines: Sequence[_RenderedLine]) -> tuple[str, Mapping[str, EvidenceSpan]]:
    visible: list[str] = []
    spans: dict[str, EvidenceSpan] = {}
    cursor = 0
    for line in lines:
        rendered = f"{line.speaker}: {line.text}"
        if line.evidence_key is not None:
            assert line.evidence_text is not None
            if line.text.count(line.evidence_text) != 1:
                raise AssertionError("evidence must occur exactly once in its turn")
            start = cursor + len(f"{line.speaker}: ") + line.text.index(
                line.evidence_text
            )
            span = EvidenceSpan(
                start=start,
                end=start + len(line.evidence_text),
                text=line.evidence_text,
            )
            if line.evidence_key in spans:
                raise AssertionError("duplicate evidence key")
            spans[line.evidence_key] = span
        visible.append(rendered)
        cursor += len(rendered) + 1
    transcript = "\n".join(visible)
    for span in spans.values():
        span.validate_against(transcript)
    return transcript, spans


def _fields_for_variant(
    world: _World,
    variant: str,
    spans: Mapping[str, EvidenceSpan],
) -> tuple[FieldOutput, ...]:
    fields: list[FieldOutput] = []
    for field in FIELD_ORDER:
        current = world.values[field]
        original = spans.get(f"{field.value}:original")
        if field is world.target_field and variant == "missing":
            output = FieldOutput(field=field, state=FieldState.MISSING)
        elif field is world.target_field and variant == "uncertain":
            uncertain = spans[f"{field.value}:uncertain"]
            output = FieldOutput(
                field=field,
                state=FieldState.UNCERTAIN,
                evidence=(uncertain,),
            )
        elif field is world.target_field and variant == "conflicting":
            assert original is not None
            correction = spans[f"{field.value}:correction"]
            evidence = (
                (correction, original) if current is None else (original, correction)
            )
            output = FieldOutput(
                field=field,
                state=FieldState.CONFLICTING,
                evidence=evidence,
            )
        elif current is None:
            assert original is not None
            output = FieldOutput(
                field=field,
                state=FieldState.ABSENT,
                evidence=(original,),
            )
        else:
            assert original is not None
            output = FieldOutput(
                field=field,
                state=FieldState.SUPPORTED,
                value=current,
                evidence=(original,),
            )
        fields.append(output)
    return tuple(fields)


def _make_example(
    world: _World,
    variant: str,
) -> tuple[StateSpanExample, _Trace]:
    lines, variant_trace = _variant_lines(world, variant)
    transcript, spans = _render(lines)
    fields = _fields_for_variant(world, variant, spans)
    for index, output in enumerate(fields):
        output.validate_against(transcript, path=f"$.fields[{index}]")
    target = encode_state_span(fields)
    parsed = parse_state_span_summary(target, transcript)
    if tuple(item.state for item in parsed) != tuple(item.state for item in fields):
        raise AssertionError("direct H4 gold does not round-trip through target grammar")
    expected_state = None if variant == "normal" else STATE_VARIANTS[variant]
    namespace = world.partition
    return (
        StateSpanExample(
            split="train",
            example_id=f"train-{namespace}-{world.index:04d}-{variant}",
            world_id=f"train-world-{namespace}-{world.index:04d}",
            variant=variant,
            target_field=world.target_field,
            target_state=expected_state,
            transcript=transcript,
            target=target,
        ),
        variant_trace,
    )


def _generate_with_trace(
    partition: str,
    *,
    worlds: int | None = None,
) -> tuple[tuple[StateSpanExample, ...], tuple[_Trace, ...]]:
    if partition not in PARTITIONS:
        raise ValueError("partition must be fit or calibration")
    count = _PARTITION_WORLDS[partition] if worlds is None else worlds
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("worlds must be a positive integer")
    if count % len(FIELD_ORDER):
        raise ValueError("worlds must be divisible by five")
    examples: list[StateSpanExample] = []
    traces: list[_Trace] = []
    for index in range(count):
        world, world_trace = _sample_world(partition, index)
        for variant in VARIANTS:
            example, variant_trace = _make_example(world, variant)
            examples.append(example)
            traces.append(
                _Trace(
                    question_templates=world_trace.question_templates,
                    answer_templates=world_trace.answer_templates,
                    denial_templates=world_trace.denial_templates,
                    uncertainty_template=variant_trace.uncertainty_template,
                    correction_template=variant_trace.correction_template,
                    values=world_trace.values,
                    correction_value=variant_trace.correction_value,
                )
            )
    return tuple(examples), tuple(traces)


def generate_partition(
    partition: str,
    *,
    worlds: int | None = None,
) -> tuple[StateSpanExample, ...]:
    """Generate one independent H4 partition in canonical record order."""

    examples, _traces = _generate_with_trace(partition, worlds=worlds)
    return examples


def records_bytes(examples: Sequence[StateSpanExample]) -> bytes:
    return b"".join(canonical_json_bytes(example.to_dict()) for example in examples)


def transcript_multiset_sha256(examples: Sequence[StateSpanExample]) -> str:
    """Hash every transcript occurrence; duplicates are intentionally retained."""

    digests = sorted(_sha256(example.transcript.encode("utf-8")) for example in examples)
    return _sha256(canonical_json_bytes(digests))


def _string_multiset_sha256(values: Iterable[str]) -> str:
    return _sha256(canonical_json_bytes(sorted(values)))


def _normalized_line_skeletons(
    examples: Sequence[StateSpanExample],
    *,
    values: Iterable[str],
) -> tuple[str, ...]:
    ordered_values = sorted(set(values), key=lambda item: (-len(item), item.casefold()))
    skeletons: list[str] = []
    for example in examples:
        for line in example.transcript.splitlines():
            skeleton = line
            for value in ordered_values:
                skeleton = skeleton.replace(value, "{value}")
            skeletons.append(skeleton)
    return tuple(skeletons)


def _state_field_quota(examples: Sequence[StateSpanExample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        for proposal in parse_state_span_summary(example.target, example.transcript):
            counts[f"{proposal.field.value}:{proposal.state.value}"] += 1
    return dict(sorted(counts.items()))


def _trace_coverage(traces: Sequence[_Trace]) -> dict[str, Any]:
    questions: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    denials: Counter[str] = Counter()
    uncertainties: Counter[str] = Counter()
    corrections: Counter[str] = Counter()
    correction_boundaries: Counter[str] = Counter()
    values: dict[str, Counter[str]] = defaultdict(Counter)
    alternatives: dict[str, Counter[str]] = defaultdict(Counter)
    # World-owned trace facts are repeated for four variants.  Count them only
    # on each quartet's first trace; variant-only facts remain per record.
    for index, trace in enumerate(traces):
        if index % VARIANTS_PER_WORLD == 0:
            for field, template in trace.question_templates:
                questions[f"{field}:{template}"] += 1
            for field, boundary, template in trace.answer_templates:
                answers[f"{field}:{template}"] += 1
                boundaries[f"{field}:{boundary}"] += 1
            for field, template in trace.denial_templates:
                denials[f"{field}:{template}"] += 1
            for field, value in trace.values:
                values[field][value] += 1
        if trace.uncertainty_template is not None:
            field, template = trace.uncertainty_template
            uncertainties[f"{field}:{template}"] += 1
        if trace.correction_template is not None:
            field, boundary, template = trace.correction_template
            corrections[f"{field}:{template}"] += 1
            correction_boundaries[f"{field}:{boundary}"] += 1
        if trace.correction_value is not None:
            field, value = trace.correction_value
            alternatives[field][value] += 1

    def serial(counter: Counter[str]) -> dict[str, int]:
        return dict(sorted(counter.items()))

    return {
        "questions": serial(questions),
        "supported_answers": serial(answers),
        "supported_boundaries": serial(boundaries),
        "denials": serial(denials),
        "uncertainties": serial(uncertainties),
        "corrections": serial(corrections),
        "correction_boundaries": serial(correction_boundaries),
        "values": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(values.items())
        },
        "conflict_alternatives": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(alternatives.items())
        },
    }


def _partition_quality(
    partition: str,
    examples: Sequence[StateSpanExample],
    traces: Sequence[_Trace],
) -> dict[str, Any]:
    """Fail closed on diversity and conflict-balance properties required by H4."""

    if len(examples) != len(traces):
        raise ValueError(f"{partition} examples and traces are misaligned")
    unique_transcripts = len({example.transcript for example in examples})
    if unique_transcripts != len(examples):
        raise ValueError(f"{partition} contains duplicate transcripts")

    questions: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    boundaries: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    answers: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    alternatives: dict[str, Counter[str]] = defaultdict(Counter)
    supported_derangements = 0
    for index, trace in enumerate(traces):
        if index % VARIANTS_PER_WORLD == 0:
            question_by_field = dict(trace.question_templates)
            answer_by_field = {
                field: (boundary, template)
                for field, boundary, template in trace.answer_templates
            }
            for field, value in trace.values:
                boundary, template = answer_by_field[field]
                questions[field][value].add(question_by_field[field])
                boundaries[field][value].add(boundary)
                answers[field][value].add(template)
        if trace.correction_value is not None:
            field, alternative = trace.correction_value
            alternatives[field][alternative] += 1
            current = dict(trace.values).get(field)
            if current is not None:
                if current == alternative:
                    raise ValueError(
                        f"{partition} conflict correction repeats supported value"
                    )
                supported_derangements += 1

    lexicons = _lexicons()[partition]
    surface_detail: dict[str, Any] = {}
    surface_summary: dict[str, Any] = {}
    for field in FIELD_ORDER:
        field_name = field.value
        values = lexicons[field]
        if set(questions[field_name]) != set(values):
            raise ValueError(f"{partition} value surface coverage is incomplete")
        expected_questions = len(_question_bank(partition, field))
        expected_answers = sum(
            len(items) for items in _answer_bank(partition).values()
        )
        detail: dict[str, Any] = {}
        for value in values:
            entry = {
                "questions": sorted(questions[field_name][value]),
                "boundaries": sorted(boundaries[field_name][value]),
                "answer_templates": sorted(answers[field_name][value]),
            }
            if len(entry["questions"]) != expected_questions:
                raise ValueError(
                    f"{partition} question coverage is incomplete for {field_name}"
                )
            if len(entry["boundaries"]) != len(_BOUNDARIES):
                raise ValueError(
                    f"{partition} boundary coverage is incomplete for {field_name}"
                )
            if len(entry["answer_templates"]) != expected_answers:
                raise ValueError(
                    f"{partition} answer coverage is incomplete for {field_name}"
                )
            detail[value] = entry
        surface_detail[field_name] = detail
        surface_summary[field_name] = {
            "values": len(values),
            "questions_per_value": expected_questions,
            "boundaries_per_value": len(_BOUNDARIES),
            "answer_templates_per_value": expected_answers,
        }

    target_count = _PARTITION_WORLDS[partition] // len(FIELD_ORDER)
    alternative_detail: dict[str, Any] = {}
    alternative_summary: dict[str, Any] = {}
    for field in FIELD_ORDER:
        field_name = field.value
        values = lexicons[field]
        observed = alternatives[field_name]
        counts = [observed[value] for value in values]
        if sum(counts) != target_count or max(counts) - min(counts) > 1:
            raise ValueError(
                f"{partition} conflict alternatives are not exactly balanced"
            )
        if sum(count > 0 for count in counts) != min(target_count, len(values)):
            raise ValueError(
                f"{partition} conflict alternative coverage is incomplete"
            )
        alternative_detail[field_name] = {
            value: observed[value] for value in values if observed[value]
        }
        alternative_summary[field_name] = {
            "observations": target_count,
            "covered_values": sum(count > 0 for count in counts),
            "minimum_per_value": min(counts),
            "maximum_per_value": max(counts),
        }

    return {
        "unique_transcripts": unique_transcripts,
        "all_transcripts_unique": True,
        "supported_value_surfaces": surface_summary,
        "supported_value_surface_sha256": _sha256(
            canonical_json_bytes(surface_detail)
        ),
        "conflict_alternatives": alternative_summary,
        "conflict_alternative_sha256": _sha256(
            canonical_json_bytes(alternative_detail)
        ),
        "supported_conflict_derangements": supported_derangements,
        "all_supported_conflicts_deranged": True,
    }


def _template_identity(partition: str) -> dict[str, Any]:
    questions = {
        field.value: _question_bank(partition, field) for field in FIELD_ORDER
    }
    answer_map = _answer_bank(partition)
    answers = {
        boundary: tuple(answer_map[boundary]) for boundary in _BOUNDARIES
    }
    denials = {
        field.value: _denial_bank(partition, field)
        for field in (FieldName.MEDICATION, FieldName.ALLERGY)
    }
    identity: dict[str, Any] = {
        "questions": questions,
        "supported_answers": answers,
        "denials": denials,
        "uncertainties": _uncertainty_bank(partition),
    }
    return {
        "counts": {
            "questions_per_field": len(next(iter(questions.values()))),
            "supported_answers": sum(len(values) for values in answers.values()),
            "medication_denials": len(denials[FieldName.MEDICATION.value]),
            "allergy_denials": len(denials[FieldName.ALLERGY.value]),
            "uncertainties": len(identity["uncertainties"]),
        },
        "sha256": _sha256(canonical_json_bytes(identity)),
        "role_sha256": {
            role: _sha256(canonical_json_bytes(value))
            for role, value in sorted(identity.items())
        },
    }


def _partition_identity(
    partition: str,
    examples: Sequence[StateSpanExample],
    traces: Sequence[_Trace],
) -> dict[str, Any]:
    lexicons = _lexicons()[partition]
    open_values = [value for field in _OPEN_FIELDS for value in lexicons[field]]
    all_values = [value for field in FIELD_ORDER for value in lexicons[field]]
    skeletons = _normalized_line_skeletons(examples, values=all_values)
    return {
        "namespace": f"train-{partition}",
        "seed": _PARTITION_SEEDS[partition],
        "records": len(examples),
        "worlds": len({example.world_id for example in examples}),
        "ordered_records_sha256": _sha256(records_bytes(examples)),
        "transcript_multiset_sha256": transcript_multiset_sha256(examples),
        "world_id_multiset_sha256": _string_multiset_sha256(
            example.world_id for example in examples
        ),
        "record_id_multiset_sha256": _string_multiset_sha256(
            example.example_id for example in examples
        ),
        "open_value_set_sha256": _sha256(canonical_json_bytes(sorted(open_values))),
        "open_value_set_by_field": {
            field.value: {
                "count": len(lexicons[field]),
                "sha256": _sha256(canonical_json_bytes(sorted(lexicons[field]))),
            }
            for field in _OPEN_FIELDS
        },
        "component_line_skeleton_multiset_sha256": _string_multiset_sha256(
            skeletons
        ),
        "component_line_skeleton_unique_count": len(set(skeletons)),
        "state_field_quota": _state_field_quota(examples),
        "coverage": _trace_coverage(traces),
        "quality": _partition_quality(partition, examples, traces),
        "templates": _template_identity(partition),
    }


def _assert_static_surface_isolation() -> None:
    fit_questions = {
        item for field in FIELD_ORDER for item in _question_bank(FIT_PARTITION, field)
    }
    calibration_questions = {
        item
        for field in FIELD_ORDER
        for item in _question_bank(CALIBRATION_PARTITION, field)
    }
    if fit_questions & calibration_questions:
        raise ValueError("fit/calibration question overlap")
    fit_answers = {
        item for values in _FIT_SUPPORTED_ANSWERS.values() for item in values
    }
    calibration_answers = {
        item for values in _CALIBRATION_SUPPORTED_ANSWERS.values() for item in values
    }
    if fit_answers & calibration_answers:
        raise ValueError("fit/calibration supported-answer overlap")
    for field in (FieldName.MEDICATION, FieldName.ALLERGY):
        if set(_FIT_DENIALS[field]) & set(_CALIBRATION_DENIALS[field]):
            raise ValueError(f"fit/calibration denial overlap for {field.value}")
    if set(_FIT_UNCERTAIN) & set(_CALIBRATION_UNCERTAIN):
        raise ValueError("fit/calibration uncertainty overlap")


def _known_development_audit(
    fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
) -> dict[str, Any]:
    """Use known development only as a post-freeze rejection boundary."""

    dev_values_by_field = supported_value_sets("dev")
    h4_values = _lexicons()
    value_overlap: dict[str, list[str]] = {}
    for partition in PARTITIONS:
        for field in _OPEN_FIELDS:
            overlap = set(h4_values[partition][field]) & set(dev_values_by_field[field])
            if overlap:
                value_overlap[f"{partition}:{field.value}"] = sorted(overlap)
    if value_overlap:
        raise ValueError("H4 open values overlap known development")

    known_dev = generate_split("dev")
    dev_all_values = [
        value for values in dev_values_by_field.values() for value in values
    ]
    h4_all_values = [
        value
        for partition in PARTITIONS
        for values in h4_values[partition].values()
        for value in values
    ]
    normalization_values = tuple(set(dev_all_values) | set(h4_all_values))
    fit_skeletons = set(
        _normalized_line_skeletons(fit, values=normalization_values)
    )
    calibration_skeletons = set(
        _normalized_line_skeletons(calibration, values=normalization_values)
    )
    dev_skeletons = set(
        _normalized_line_skeletons(known_dev, values=normalization_values)
    )
    if fit_skeletons & calibration_skeletons:
        raise ValueError("fit/calibration normalized component overlap")
    if fit_skeletons & dev_skeletons or calibration_skeletons & dev_skeletons:
        raise ValueError("H4 normalized component overlap with known development")
    return {
        "known_development_role": "leakage_rejection_only_after_design_freeze",
        "open_values_disjoint": True,
        "fit_component_skeletons_disjoint": True,
        "calibration_component_skeletons_disjoint": True,
        "development_records_in_training": 0,
        "development_used_for_template_or_value_selection": False,
    }


def build_manifest(
    fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
    *,
    fit_traces: Sequence[_Trace],
    calibration_traces: Sequence[_Trace],
    generator_sha256: str,
    tokenizer_sha256: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build and validate the complete H4 pre-training identity."""

    for role, digest in (
        ("generator", generator_sha256),
        ("tokenizer", tokenizer_sha256),
        ("base checkpoint", base_checkpoint_sha256),
    ):
        _require_digest(digest, role)
    _assert_static_surface_isolation()
    if len(fit) != FIT_RECORDS or len(calibration) != CALIBRATION_RECORDS:
        raise ValueError("H4 record count changed")
    fit_worlds = {example.world_id for example in fit}
    calibration_worlds = {example.world_id for example in calibration}
    if len(fit_worlds) != FIT_WORLDS or len(calibration_worlds) != CALIBRATION_WORLDS:
        raise ValueError("H4 world count changed")
    if fit_worlds & calibration_worlds:
        raise ValueError("H4 world namespaces overlap")
    fit_ids = {example.example_id for example in fit}
    calibration_ids = {example.example_id for example in calibration}
    if len(fit_ids) != len(fit) or len(calibration_ids) != len(calibration):
        raise ValueError("duplicate H4 record IDs")
    if fit_ids & calibration_ids:
        raise ValueError("H4 record namespaces overlap")
    fit_transcripts = {example.transcript for example in fit}
    calibration_transcripts = {example.transcript for example in calibration}
    if fit_transcripts & calibration_transcripts:
        raise ValueError("H4 exact transcript overlap")
    lexicons = _lexicons()
    for field in _OPEN_FIELDS:
        if set(lexicons[FIT_PARTITION][field]) & set(
            lexicons[CALIBRATION_PARTITION][field]
        ):
            raise ValueError(f"H4 open-value overlap for {field.value}")
    open_values_casefolded = {
        value.casefold()
        for partition in PARTITIONS
        for field in _OPEN_FIELDS
        for value in lexicons[partition][field]
    }
    if open_values_casefolded & set(FORBIDDEN_HISTORICAL_SENTINELS):
        raise ValueError("historical sentinel leaked into H4")

    dev_audit = _known_development_audit(fit, calibration)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": GENERATOR_VERSION,
        "recipe": TRAINING_RECIPE_VERSION,
        "target_grammar": TARGET_GRAMMAR_VERSION,
        "generator_sha256": generator_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "training_identity": {
            "training_seeds": list(TRAINING_SEEDS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "steps_per_seed": STEPS_PER_SEED,
            "gradient_records": FIT_RECORDS,
            "calibration_records": CALIBRATION_RECORDS,
            "calibration_gradient_bearing": False,
        },
        "partitions": {
            FIT_PARTITION: _partition_identity(FIT_PARTITION, fit, fit_traces),
            CALIBRATION_PARTITION: _partition_identity(
                CALIBRATION_PARTITION, calibration, calibration_traces
            ),
        },
        "isolation": {
            "world_ids_disjoint": True,
            "record_ids_disjoint": True,
            "exact_transcripts_disjoint": True,
            "open_values_disjoint_except_severity": True,
            "exact_templates_disjoint": True,
            "normalized_component_skeletons_disjoint": True,
            **dev_audit,
            "gold_built_from_annotated_turn_offsets": True,
            "deterministic_baseline_used_for_labels": False,
            "historical_sentinels_present": False,
            "historical_fresh_v0_read": False,
            "sealed_confirmation_read": False,
        },
    }


def _write_no_clobber(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise OSError(f"could not create H4 dataset file: {path.name}") from exc


def write_dataset(
    output_dir: Path,
    *,
    tokenizer_sha256: str,
    base_checkpoint_sha256: str,
) -> Mapping[str, Any]:
    """Create the frozen H4 fit/calibration family without overwriting."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"H4 output directory already exists: {output}")
    generator_snapshot = Path(__file__).read_bytes()
    generator_sha256 = _sha256(generator_snapshot)
    fit, fit_traces = _generate_with_trace(FIT_PARTITION)
    calibration, calibration_traces = _generate_with_trace(CALIBRATION_PARTITION)
    manifest = build_manifest(
        fit,
        calibration,
        fit_traces=fit_traces,
        calibration_traces=calibration_traces,
        generator_sha256=generator_sha256,
        tokenizer_sha256=tokenizer_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    if _sha256(Path(__file__).read_bytes()) != generator_sha256:
        raise RuntimeError("H4 generator source changed during generation")
    output.mkdir(parents=True, exist_ok=False)
    _write_no_clobber(output / "fit.jsonl", records_bytes(fit))
    _write_no_clobber(output / "calibration.jsonl", records_bytes(calibration))
    _write_no_clobber(output / "manifest.json", canonical_json_bytes(manifest))
    return manifest


def load_records(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[StateSpanExample, ...]:
    snapshot = Path(path).read_bytes()
    if expected_sha256 is not None and _sha256(snapshot) != expected_sha256:
        raise ValueError(f"H4 dataset digest mismatch: {path.name}")
    records: list[StateSpanExample] = []
    for line_number, line in enumerate(snapshot.splitlines(), 1):
        try:
            value = json.loads(line)
            records.append(StateSpanExample.from_dict(value))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid H4 record at {path.name}:{line_number}") from exc
    if not records:
        raise ValueError(f"H4 dataset is empty: {path.name}")
    return tuple(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Nano H4 surface-transfer data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    write_dataset(
        args.output_dir,
        tokenizer_sha256=args.tokenizer_sha256,
        base_checkpoint_sha256=args.base_checkpoint_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_SIZE",
    "CALIBRATION_GENERATOR_SEED",
    "CALIBRATION_PARTITION",
    "CALIBRATION_RECORDS",
    "CALIBRATION_WORLDS",
    "EPOCHS",
    "FIT_GENERATOR_SEED",
    "FIT_PARTITION",
    "FIT_RECORDS",
    "FIT_WORLDS",
    "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "STEPS_PER_EPOCH",
    "STEPS_PER_SEED",
    "TRAINING_RECIPE_VERSION",
    "TRAINING_SEEDS",
    "build_manifest",
    "generate_partition",
    "load_records",
    "partition_lexicons",
    "records_bytes",
    "transcript_multiset_sha256",
    "write_dataset",
]
