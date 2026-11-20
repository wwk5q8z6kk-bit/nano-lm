from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FIELD_ORDER, FieldName, FieldState
from nano_ai.training import pointer_data
from nano_ai.training.pointer_data import (
    IGNORE_SPAN_INDEX,
    POINTER_PROMPT_INSTRUCTION,
    STATE_ORDER,
    PointerDataError,
    character_span_to_token_span,
    encode_pointer_example,
    encode_pointer_partition,
    load_pointer_tokenizer,
    token_span_to_evidence,
)
from nano_ai.training.state_span_data import (
    StateSpanExample,
    canonical_json_bytes,
    generate_split,
)


def _tokenizer_path() -> Path:
    return Path(__file__).resolve().parents[2] / "sft" / "tokenizer.json"


@pytest.fixture(scope="module")
def frozen_tokenizer() -> Tokenizer:
    return load_pointer_tokenizer(_tokenizer_path())


def test_segmented_chatml_keeps_exact_transcript_ids_and_offsets(
    frozen_tokenizer: Tokenizer,
) -> None:
    example = generate_split("dev", worlds=5)[0]
    encoded = encode_pointer_example(frozen_tokenizer, example)
    transcript_encoding = frozen_tokenizer.encode(
        example.transcript,
        add_special_tokens=False,
    )
    start = encoded.transcript_token_start
    end = encoded.transcript_token_end

    assert encoded.token_ids[start:end] == tuple(transcript_encoding.ids)
    assert encoded.token_offsets[start:end] == tuple(transcript_encoding.offsets)
    assert all(offset is None for offset in encoded.token_offsets[:start])
    assert all(offset is None for offset in encoded.token_offsets[end:])
    assert encoded.transcript_mask == tuple(
        start <= index < end for index in range(len(encoded.token_ids))
    )
    assert all(encoded.attention_mask)
    assert len(encoded.token_ids) <= 512

    instruction_ids = frozen_tokenizer.encode(
        f"\n{POINTER_PROMPT_INSTRUCTION}",
        add_special_tokens=False,
    ).ids
    assert list(encoded.token_ids[end : end + len(instruction_ids)]) == instruction_ids


def test_state_and_two_slot_labels_are_exact_and_inactive_slots_are_explicit(
    frozen_tokenizer: Tokenizer,
) -> None:
    examples = generate_split("dev", worlds=5)[:4]
    encoded_examples = encode_pointer_partition(
        frozen_tokenizer,
        examples,
        expected_split="dev",
    )

    for source, encoded in zip(examples, encoded_examples, strict=True):
        proposals = parse_state_span_summary(source.target, source.transcript)
        assert encoded.state_labels == tuple(
            STATE_ORDER.index(proposal.state) for proposal in proposals
        )
        for field_index, proposal in enumerate(proposals):
            assert sum(encoded.span_mask[field_index]) == len(proposal.spans)
            for slot in range(2):
                if slot >= len(proposal.spans):
                    assert encoded.span_mask[field_index][slot] is False
                    assert encoded.span_starts[field_index][slot] == IGNORE_SPAN_INDEX
                    assert encoded.span_ends[field_index][slot] == IGNORE_SPAN_INDEX
                    continue
                reconstructed = token_span_to_evidence(
                    source.transcript,
                    encoded.token_offsets,
                    encoded.span_starts[field_index][slot],
                    encoded.span_ends[field_index][slot],
                )
                assert reconstructed == proposal.spans[slot]

    conflicting = encoded_examples[3]
    target_field_index = FIELD_ORDER.index(examples[3].target_field)
    assert examples[3].variant == "conflicting"
    assert conflicting.state_labels[target_field_index] == STATE_ORDER.index(
        FieldState.CONFLICTING
    )
    assert conflicting.span_mask[target_field_index] == (True, True)


def test_outer_unicode_whitespace_is_trimmed_without_normalizing_content(
    frozen_tokenizer: Tokenizer,
) -> None:
    transcript = (
        "Doctor: What brings you in?\nPatient: \u2009migraine\u2009\nDoctor: Thank you."
    )
    example = StateSpanExample(
        split="dev",
        example_id="dev-whitespace-normal",
        world_id="dev-world-whitespace",
        variant="normal",
        target_field=FieldName.CHIEF_COMPLAINT,
        target_state=None,
        transcript=transcript,
        target="CC:S[migraine]|DUR:M|SEV:M|MED:M|ALG:M",
    )
    encoded = encode_pointer_example(frozen_tokenizer, example)
    start = encoded.span_starts[0][0]
    end = encoded.span_ends[0][0]

    evidence = token_span_to_evidence(
        transcript,
        encoded.token_offsets,
        start,
        end,
    )
    assert evidence.text == "migraine"
    assert transcript[evidence.start : evidence.end] == "migraine"
    assert all(encoded.pointer_mask[start : end + 1])


def test_pointer_mask_and_reconstruction_reject_doctor_or_prompt_tokens(
    frozen_tokenizer: Tokenizer,
) -> None:
    example = generate_split("dev", worlds=5)[0]
    encoded = encode_pointer_example(frozen_tokenizer, example)
    doctor_character = example.transcript.index("Doctor")
    doctor_token = next(
        index
        for index, offset in enumerate(encoded.token_offsets)
        if offset is not None and offset[0] <= doctor_character < offset[1]
    )

    assert encoded.pointer_mask[doctor_token] is False
    with pytest.raises(PointerDataError, match="Patient evidence"):
        token_span_to_evidence(
            example.transcript,
            encoded.token_offsets,
            doctor_token,
            doctor_token,
        )
    with pytest.raises(PointerDataError, match="inside the transcript"):
        token_span_to_evidence(
            example.transcript,
            encoded.token_offsets,
            0,
            0,
        )


def test_source_identity_and_partition_order_are_preserved(
    frozen_tokenizer: Tokenizer,
) -> None:
    examples = generate_split("train", worlds=5)[:7]
    encoded = encode_pointer_partition(
        frozen_tokenizer,
        examples,
        expected_split="train",
    )

    assert [item.example_id for item in encoded] == [
        item.example_id for item in examples
    ]
    assert [item.world_id for item in encoded] == [item.world_id for item in examples]
    assert [item.source_record_sha256 for item in encoded] == [
        hashlib.sha256(canonical_json_bytes(item.to_dict())).hexdigest()
        for item in examples
    ]


def test_partition_isolation_rejects_mixups_and_duplicates(
    frozen_tokenizer: Tokenizer,
) -> None:
    train = generate_split("train", worlds=5)
    dev = generate_split("dev", worlds=5)

    with pytest.raises(PointerDataError, match="cross-split"):
        encode_pointer_partition(
            frozen_tokenizer,
            (train[0], dev[0]),
            expected_split="train",
        )
    with pytest.raises(PointerDataError, match="duplicate example IDs"):
        encode_pointer_partition(
            frozen_tokenizer,
            (train[0], train[0]),
            expected_split="train",
        )
    forged = replace(train[0], example_id="dev-forged")
    with pytest.raises(PointerDataError, match="source identity"):
        encode_pointer_partition(
            frozen_tokenizer,
            (forged,),
            expected_split="train",
        )


def test_supervision_rejects_state_arity_mismatch_and_pointer_mask_holes(
    frozen_tokenizer: Tokenizer,
) -> None:
    encoded = encode_pointer_partition(
        frozen_tokenizer,
        generate_split("dev", worlds=5),
        expected_split="dev",
    )
    record = encoded[0]
    field_index = next(
        index for index, slots in enumerate(record.span_mask) if any(slots)
    )
    changed_states = list(record.state_labels)
    changed_states[field_index] = STATE_ORDER.index(FieldState.MISSING)
    with pytest.raises(PointerDataError, match="state and active evidence-span count"):
        replace(record, state_labels=tuple(changed_states))

    multi_token = next(
        (
            item,
            item.span_starts[field][slot],
            item.span_ends[field][slot],
        )
        for item in encoded
        for field in range(len(FIELD_ORDER))
        for slot in range(2)
        if item.span_mask[field][slot]
        and item.span_ends[field][slot] - item.span_starts[field][slot] >= 2
    )
    item, start, end = multi_token
    changed_mask = list(item.pointer_mask)
    changed_mask[start + (end - start) // 2] = False
    with pytest.raises(PointerDataError, match="wholly inside Patient content"):
        replace(item, pointer_mask=tuple(changed_mask))


def test_non_frozen_tokenizer_and_context_overflow_fail_closed(
    frozen_tokenizer: Tokenizer,
) -> None:
    changed = Tokenizer.from_str(frozen_tokenizer.to_str())
    changed.add_tokens(["not-a-frozen-token"])
    example = generate_split("dev", worlds=5)[0]
    with pytest.raises(PointerDataError, match="identity"):
        encode_pointer_example(changed, example)

    oversized = replace(
        example,
        transcript="Patient: " + " ".join(["word"] * 700),
    )
    with pytest.raises(PointerDataError, match="512-token context"):
        encode_pointer_example(frozen_tokenizer, oversized)


def test_all_65000_frozen_recipe_spans_round_trip_exactly(
    frozen_tokenizer: Tokenizer,
) -> None:
    span_count = 0
    for split in ("train", "dev"):
        for example in generate_split(split):
            transcript_encoding = frozen_tokenizer.encode(
                example.transcript,
                add_special_tokens=False,
            )
            offsets = tuple(transcript_encoding.offsets)
            for proposal in parse_state_span_summary(
                example.target,
                example.transcript,
            ):
                for source_evidence in proposal.spans:
                    start, end = character_span_to_token_span(
                        example.transcript,
                        offsets,
                        source_evidence,
                    )
                    assert (
                        token_span_to_evidence(
                            example.transcript,
                            offsets,
                            start,
                            end,
                        )
                        == source_evidence
                    )
                    span_count += 1
    assert span_count == 65_000


def test_pointer_data_source_has_no_benchmark_dependency() -> None:
    source = Path(pointer_data.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        module.startswith(("nano_ai.benchmark", "nano_ai.benchmarks"))
        for module in imported_modules
    )
