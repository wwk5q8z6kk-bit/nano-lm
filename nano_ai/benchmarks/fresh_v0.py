"""Construction and sealed loading for Nano's first fresh v0 benchmark.

This module owns benchmark data only.  It never invokes a Nano solver and it
does not write a partition to disk.  Callers must deliberately serialize and
seal a generated document before the runner can consume it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import stat
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from nano_ai.benchmarking import BENCHMARK_CASE_SCHEMA_VERSION
from nano_ai.contract import (
    CONTRACT_VERSION,
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.fixtures import FixtureCase

BENCHMARK_ID = "nano-v0-test-20260802"
MANIFEST_SCHEMA_VERSION = "nano.benchmark-manifest.v0"
PARTITION_SCHEMA_VERSION = "nano.benchmark-partition.v0"
BENCHMARK_PROVENANCE_SCHEMA_VERSION = BENCHMARK_CASE_SCHEMA_VERSION
GOLD_SOLVER_ID = "benchmark-gold-v0"
FRESH_SEED = 20260802
GENERATOR_PATH = "scribe/build_scribe_data.py"
GENERATOR_SHA256 = "3c4631f918bf6b0532641bb004c617f513d684b66d7eb3314fc7f87da5dc3df2"

TOTAL_RECORDS = 220
NORMAL_RECORDS = 160
NORMAL_SEEN_RECORDS = 80
NORMAL_HELD_RECORDS = 80
CHALLENGE_RECORDS = 60
PAIRED_NORMAL_RECORDS = 60
STATE_CHALLENGE_RECORDS = 20

HISTORICAL_RECORDS = 8217
HISTORICAL_UNIQUE_TRANSCRIPTS = 6689
HISTORICAL_EXACT_MULTISET_SHA256 = (
    "6231146627fa6232e6d4d7e098452f752347123de4ede1eada688d98185bc07f"
)
HISTORICAL_NORMALIZED_MULTISET_SHA256 = (
    "d285aa393cf1388340a644df3ac486fdc17618973efd040b6bb418de2185d566"
)
REGENERATED_V1_TRAIN_RECORDS = 8000

_INSTRUCTION_SUFFIX = "\nSummarize the visit."
_MANIFEST_STATUSES = frozenset({"fresh_unmeasured", "historical_regression"})
_VALUE_BANDS = frozenset({"seen", "held"})
_TARGET_STATES = (
    FieldState.MISSING,
    FieldState.UNCERTAIN,
    FieldState.CONFLICTING,
)

_HISTORICAL_SOURCE_SPECS: tuple[tuple[str, int], ...] = (
    ("nano_ai/fixtures/v0/manifest.json", 3),
    ("benchmarks/adapters/lm_eval/fixtures/held_value_sentinel_n4.json", 4),
    ("scribe/scribe_eval.json", 40),
    ("trajectory/scribe_dev.json", 10),
    ("trajectory/scribe_eval_T.json", 40),
    *((f"trajectory/scribe_eval_m{index}.json", 200) for index in range(5)),
    *((f"trajectory/c3_eval/c3_m{index}.json", 400) for index in range(5)),
    ("trajectory/e4/data/rstar_train.json", 800),
    ("trajectory/e4/data/rstar_dev.json", 100),
    ("trajectory/e4/data/rstar_eval.json", 220),
    *((f"trajectory/interference_eval/if_m{index}.json", 200) for index in range(5)),
    *(
        (f"trajectory/sweep_eval/d{depth}_m{index}.json", 200)
        for depth in (5, 20, 80)
        for index in range(5)
    ),
)
HISTORICAL_SOURCE_PATHS = tuple(path for path, _ in _HISTORICAL_SOURCE_SPECS)

_DEFINITION_NAMES = (
    "CC_TRAIN",
    "CC_HELD",
    "MED_TRAIN",
    "MED_HELD",
    "ALG_TRAIN",
    "ALG_HELD",
    "SEV",
    "D_OPEN_TRAIN",
    "D_OPEN_HELD",
    "P_CC_TRAIN",
    "P_CC_HELD",
    "D_DUR_TRAIN",
    "D_DUR_HELD",
    "P_DUR_TRAIN",
    "P_DUR_HELD",
    "D_SEV_TRAIN",
    "D_SEV_HELD",
    "P_SEV_TRAIN",
    "P_SEV_HELD",
    "D_MED_TRAIN",
    "D_MED_HELD",
    "P_MED_YES_TRAIN",
    "P_MED_YES_HELD",
    "P_MED_NO_TRAIN",
    "P_MED_NO_HELD",
    "D_ALG_TRAIN",
    "D_ALG_HELD",
    "P_ALG_YES_TRAIN",
    "P_ALG_YES_HELD",
    "P_ALG_NO_TRAIN",
    "P_ALG_NO_HELD",
    "DISTRACT",
)

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "contract_version",
        "status",
        "seed",
        "generator",
        "partition",
        "composition",
        "collision_index",
    }
)
_GENERATOR_KEYS = frozenset({"path", "sha256"})
_PARTITION_SPEC_KEYS = frozenset({"path", "records", "sha256"})
_COMPOSITION_KEYS = frozenset(
    {
        "normal",
        "normal_seen",
        "normal_held",
        "state_challenge",
        "missing",
        "uncertain",
        "conflicting",
        "paired_normal",
    }
)
_COLLISION_KEYS = frozenset(
    {
        "sources",
        "record_count",
        "unique_transcript_count",
        "exact_multiset_sha256",
        "normalized_multiset_sha256",
        "regenerated_v1_train_count",
    }
)
_PARTITION_KEYS = frozenset(
    {"schema_version", "benchmark_id", "seed", "generator_sha256", "cases"}
)
_CASE_KEYS = frozenset(
    {"case_id", "transcript", "transcript_sha256", "gold", "benchmark"}
)
_BENCHMARK_KEYS = frozenset(
    {
        "schema_version",
        "family",
        "variant",
        "value_band",
        "target_state",
        "target_field",
        "pair_id",
    }
)


class BenchmarkIntegrityError(ValueError):
    """A benchmark artifact or source failed its frozen integrity contract."""


@dataclass(frozen=True, slots=True)
class FrozenBenchmark:
    """One validated benchmark snapshot ready for inference by a runner."""

    benchmark_id: str
    status: str
    manifest_sha256: str
    partition_sha256: str
    cases: tuple[FixtureCase, ...]


@dataclass(frozen=True, slots=True)
class V1Definitions:
    """Literal-only definitions recovered from the pinned v1 generator."""

    values: Mapping[str, tuple[Any, ...]]

    def __getitem__(self, name: str) -> tuple[Any, ...]:
        return self.values[name]


@dataclass(frozen=True, slots=True)
class CollisionIndex:
    sources: tuple[str, ...]
    record_count: int
    unique_transcript_count: int
    exact_multiset_sha256: str
    normalized_multiset_sha256: str
    regenerated_v1_train_count: int
    exact_hashes: frozenset[str]
    normalized_hashes: frozenset[str]
    held_complaints: frozenset[str]
    held_medications: frozenset[str]
    held_allergies: frozenset[str]

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "record_count": self.record_count,
            "unique_transcript_count": self.unique_transcript_count,
            "exact_multiset_sha256": self.exact_multiset_sha256,
            "normalized_multiset_sha256": self.normalized_multiset_sha256,
            "regenerated_v1_train_count": self.regenerated_v1_train_count,
        }


@dataclass(frozen=True, slots=True)
class _TupleRecord:
    cc: tuple[str, str]
    n: int
    unit: str
    sev: str
    med: str | None
    alg: str | None


@dataclass(frozen=True, slots=True)
class _NormalRecord:
    transcript: str
    fields: tuple[FieldOutput, ...]
    payload_ranges: Mapping[FieldName, tuple[int, int]]
    value_band: str
    pair_id: str | None = None
    target_state: FieldState | None = None
    target_field: FieldName | None = None


@dataclass(frozen=True, slots=True)
class _ChallengeRecord:
    transcript: str
    fields: tuple[FieldOutput, ...]
    value_band: str
    target_state: FieldState
    target_field: FieldName
    pair_id: str
    base: _NormalRecord


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only accepted byte representation of a sealed JSON artifact."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _normalized_transcript(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _multiset_digest(values: Iterable[str]) -> str:
    hashes = sorted(_sha256_text(value) for value in values)
    return _sha256_text("\n".join(hashes))


def _require_exact_keys(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise BenchmarkIntegrityError(f"invalid {label} schema")
    return value


def _require_int(value: Any, label: str, *, expected: int | None = None) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        suffix = "" if expected is None else f" equal to {expected}"
        raise BenchmarkIntegrityError(f"{label} must be an integer{suffix}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BenchmarkIntegrityError(
            f"{label} must be a non-empty edge-trimmed string"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BenchmarkIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BenchmarkIntegrityError(f"non-finite JSON value: {value}")


def _parse_json_snapshot(snapshot: bytes, label: str) -> Any:
    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except BenchmarkIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkIntegrityError(f"cannot parse {label}: {exc}") from exc


def _read_regular_snapshot(path: Path, label: str) -> bytes:
    """Read a regular, non-symlink file once."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BenchmarkIntegrityError(f"cannot open {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchmarkIntegrityError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise BenchmarkIntegrityError(f"cannot read {label}: {exc}") from exc
    finally:
        os.close(descriptor)


def _repo_file(repository_root: Path, relative: str) -> Path:
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise BenchmarkIntegrityError(f"unsafe repository path: {relative}")
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise BenchmarkIntegrityError("repository root is unavailable") from exc
    if not root.is_dir():
        raise BenchmarkIntegrityError("repository root is not a directory")
    cursor = root
    for part in relative_path.parts:
        cursor /= part
        try:
            if stat.S_ISLNK(cursor.lstat().st_mode):
                raise BenchmarkIntegrityError(
                    f"repository path uses a symlink: {relative}"
                )
        except FileNotFoundError as exc:
            raise BenchmarkIntegrityError(
                f"repository file is unavailable: {relative}"
            ) from exc
    try:
        cursor.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            f"repository path escapes root: {relative}"
        ) from exc
    return cursor


def _read_repo_snapshot(repository_root: Path, relative: str, label: str) -> bytes:
    return _read_regular_snapshot(_repo_file(repository_root, relative), label)


def _artifact_sibling(manifest_path: Path, relative: Any) -> Path:
    value = _require_string(relative, "partition path")
    relative_path = PurePosixPath(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise BenchmarkIntegrityError("partition path must be a safe relative path")
    parent = manifest_path.parent.resolve(strict=True)
    candidate = parent.joinpath(*relative_path.parts)
    try:
        candidate.resolve(strict=True).relative_to(parent)
    except (OSError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            "partition path escapes the manifest directory"
        ) from exc
    cursor = parent
    for part in relative_path.parts:
        cursor /= part
        if stat.S_ISLNK(cursor.lstat().st_mode):
            raise BenchmarkIntegrityError("partition path uses a symlink")
    return candidate


def load_v1_definitions(
    repository_root: Path,
    *,
    expected_sha256: str = GENERATOR_SHA256,
) -> V1Definitions:
    """Safely parse pinned generator literals without importing or executing it."""

    _require_sha256(expected_sha256, "generator digest")
    snapshot = _read_repo_snapshot(repository_root, GENERATOR_PATH, "v1 generator")
    if _sha256_bytes(snapshot) != expected_sha256:
        raise BenchmarkIntegrityError("v1 generator hash mismatch")
    try:
        tree = ast.parse(snapshot.decode("utf-8"), filename=GENERATOR_PATH)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise BenchmarkIntegrityError("cannot parse the pinned v1 generator") from exc
    literals: dict[str, tuple[Any, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in _DEFINITION_NAMES:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise BenchmarkIntegrityError(
                f"generator definition is not literal: {target.id}"
            ) from exc
        if not isinstance(value, list) or not value:
            raise BenchmarkIntegrityError(
                f"generator definition is invalid: {target.id}"
            )
        literals[target.id] = tuple(
            tuple(item) if isinstance(item, tuple) else item for item in value
        )
    if frozenset(literals) != frozenset(_DEFINITION_NAMES):
        raise BenchmarkIntegrityError("generator literal definitions are incomplete")
    return V1Definitions(values=literals)


def _sample_tuple(
    rng: random.Random, definitions: V1Definitions, held: bool
) -> _TupleRecord:
    cc = rng.choice(
        definitions["CC_HELD"]
        if held and rng.random() < 0.7
        else definitions["CC_TRAIN"]
    )
    unit = rng.choice(("days", "weeks"))
    n = rng.randint(2, 14) if unit == "days" else rng.randint(1, 6)
    sev = rng.choice(definitions["SEV"])
    med = None
    if rng.random() < 0.6:
        med = rng.choice(
            definitions["MED_HELD"]
            if held and rng.random() < 0.5
            else definitions["MED_TRAIN"]
        )
    alg = None
    if rng.random() < 0.5:
        alg = rng.choice(
            definitions["ALG_HELD"]
            if held and rng.random() < 0.5
            else definitions["ALG_TRAIN"]
        )
    return _TupleRecord(cc=cc, n=n, unit=unit, sev=sev, med=med, alg=alg)


def _v1_render(
    rng: random.Random,
    definitions: V1Definitions,
    values: _TupleRecord,
    held: bool,
) -> str:
    def pick(train: str, held_name: str) -> str:
        return rng.choice(definitions[held_name] if held else definitions[train])

    lines = ["Doctor: " + pick("D_OPEN_TRAIN", "D_OPEN_HELD")]
    lines.append("Patient: " + pick("P_CC_TRAIN", "P_CC_HELD").format(cc=values.cc[0]))
    if rng.random() < 0.4:
        lines.append("Patient: " + rng.choice(definitions["DISTRACT"]))
    lines.append("Doctor: " + pick("D_DUR_TRAIN", "D_DUR_HELD"))
    lines.append(
        "Patient: "
        + pick("P_DUR_TRAIN", "P_DUR_HELD").format(n=values.n, unit=values.unit)
    )
    lines.append("Doctor: " + pick("D_SEV_TRAIN", "D_SEV_HELD"))
    lines.append("Patient: " + pick("P_SEV_TRAIN", "P_SEV_HELD").format(sev=values.sev))
    lines.append("Doctor: " + pick("D_MED_TRAIN", "D_MED_HELD"))
    if values.med:
        patient_med = pick("P_MED_YES_TRAIN", "P_MED_YES_HELD").format(med=values.med)
    else:
        patient_med = pick("P_MED_NO_TRAIN", "P_MED_NO_HELD")
    lines.append("Patient: " + patient_med)
    lines.append("Doctor: " + pick("D_ALG_TRAIN", "D_ALG_HELD"))
    if values.alg:
        patient_alg = pick("P_ALG_YES_TRAIN", "P_ALG_YES_HELD").format(alg=values.alg)
    else:
        patient_alg = pick("P_ALG_NO_TRAIN", "P_ALG_NO_HELD")
    lines.append("Patient: " + patient_alg)
    return "\n".join(lines)


def _strip_instruction(record: Any) -> str:
    try:
        message = record["convo"][0]
        if message["role"] != "user" or not isinstance(message["content"], str):
            raise BenchmarkIntegrityError(
                "historical record has an invalid user message"
            )
        content = message["content"]
    except BenchmarkIntegrityError:
        raise
    except (KeyError, IndexError, TypeError) as exc:
        raise BenchmarkIntegrityError("historical record has an invalid shape") from exc
    if not content.endswith(_INSTRUCTION_SUFFIX):
        raise BenchmarkIntegrityError(
            "historical record lacks the exact instruction suffix"
        )
    return content[: -len(_INSTRUCTION_SUFFIX)]


def _historical_transcripts(repository_root: Path) -> list[str]:
    transcripts: list[str] = []
    for path, expected_records in _HISTORICAL_SOURCE_SPECS:
        value = _parse_json_snapshot(
            _read_repo_snapshot(repository_root, path, f"historical source {path}"),
            f"historical source {path}",
        )
        if path == "nano_ai/fixtures/v0/manifest.json":
            try:
                rows = value["contract_smoke"]
                source_transcripts = [row["source"]["transcript"] for row in rows]
            except (KeyError, TypeError) as exc:
                raise BenchmarkIntegrityError(
                    "contract-smoke history has an invalid shape"
                ) from exc
            if any(
                not isinstance(item, str) or not item for item in source_transcripts
            ):
                raise BenchmarkIntegrityError(
                    "contract-smoke history contains invalid transcripts"
                )
            current = source_transcripts
        else:
            if not isinstance(value, list):
                raise BenchmarkIntegrityError(
                    f"historical source is not an array: {path}"
                )
            current = [_strip_instruction(record) for record in value]
        if len(current) != expected_records:
            raise BenchmarkIntegrityError(
                f"historical source record-count mismatch: {path}"
            )
        transcripts.extend(current)
    return transcripts


def build_collision_index(repository_root: Path) -> CollisionIndex:
    """Verify the explicit history and add a deterministic v1-train reconstruction."""

    definitions = load_v1_definitions(repository_root)
    historical = _historical_transcripts(repository_root)
    if len(historical) != HISTORICAL_RECORDS:
        raise BenchmarkIntegrityError("historical aggregate record-count mismatch")
    if len(set(historical)) != HISTORICAL_UNIQUE_TRANSCRIPTS:
        raise BenchmarkIntegrityError("historical aggregate unique-count mismatch")
    exact_digest = _multiset_digest(historical)
    normalized_digest = _multiset_digest(map(_normalized_transcript, historical))
    if exact_digest != HISTORICAL_EXACT_MULTISET_SHA256:
        raise BenchmarkIntegrityError("historical exact multiset digest mismatch")
    if normalized_digest != HISTORICAL_NORMALIZED_MULTISET_SHA256:
        raise BenchmarkIntegrityError("historical normalized multiset digest mismatch")

    rng = random.Random(7)
    regenerated = [
        _v1_render(rng, definitions, values, False)
        for values in (
            _sample_tuple(rng, definitions, False)
            for _ in range(REGENERATED_V1_TRAIN_RECORDS)
        )
    ]
    # The generator expression above deliberately preserves v1's sample-then-render
    # sequence because each value is consumed before the next sample is requested.
    all_transcripts = historical + regenerated
    return CollisionIndex(
        sources=HISTORICAL_SOURCE_PATHS,
        record_count=len(historical),
        unique_transcript_count=len(set(historical)),
        exact_multiset_sha256=exact_digest,
        normalized_multiset_sha256=normalized_digest,
        regenerated_v1_train_count=len(regenerated),
        exact_hashes=frozenset(_sha256_text(item) for item in all_transcripts),
        normalized_hashes=frozenset(
            _sha256_text(_normalized_transcript(item)) for item in all_transcripts
        ),
        held_complaints=frozenset(item[1] for item in definitions["CC_HELD"]),
        held_medications=frozenset(definitions["MED_HELD"]),
        held_allergies=frozenset(definitions["ALG_HELD"]),
    )


def _patient_line(
    lines: list[tuple[str, str, FieldName | None, str | None]],
    text: str,
    field: FieldName | None,
    evidence: str | None,
) -> None:
    lines.append(("Patient", text, field, evidence))


def _fresh_normal(
    rng: random.Random,
    definitions: V1Definitions,
    values: _TupleRecord,
    value_band: str,
) -> _NormalRecord:
    """Render held dialogue templates with no optional distractor turns."""

    lines: list[tuple[str, str, FieldName | None, str | None]] = []
    lines.append(("Doctor", rng.choice(definitions["D_OPEN_HELD"]), None, None))
    cc_text = rng.choice(definitions["P_CC_HELD"]).format(cc=values.cc[0])
    _patient_line(lines, cc_text, FieldName.CHIEF_COMPLAINT, values.cc[0])
    lines.append(("Doctor", rng.choice(definitions["D_DUR_HELD"]), None, None))
    duration = f"{values.n} {values.unit}"
    dur_text = rng.choice(definitions["P_DUR_HELD"]).format(
        n=values.n, unit=values.unit
    )
    _patient_line(lines, dur_text, FieldName.DURATION, duration)
    lines.append(("Doctor", rng.choice(definitions["D_SEV_HELD"]), None, None))
    sev_text = rng.choice(definitions["P_SEV_HELD"]).format(sev=values.sev)
    _patient_line(lines, sev_text, FieldName.SEVERITY, values.sev)
    lines.append(("Doctor", rng.choice(definitions["D_MED_HELD"]), None, None))
    if values.med is None:
        med_text = rng.choice(definitions["P_MED_NO_HELD"])
        med_evidence = med_text
    else:
        med_text = rng.choice(definitions["P_MED_YES_HELD"]).format(med=values.med)
        med_evidence = values.med
    _patient_line(lines, med_text, FieldName.MEDICATION, med_evidence)
    lines.append(("Doctor", rng.choice(definitions["D_ALG_HELD"]), None, None))
    if values.alg is None:
        alg_text = rng.choice(definitions["P_ALG_NO_HELD"])
        alg_evidence = alg_text
    else:
        alg_text = rng.choice(definitions["P_ALG_YES_HELD"]).format(alg=values.alg)
        alg_evidence = values.alg
    _patient_line(lines, alg_text, FieldName.ALLERGY, alg_evidence)

    rendered = [f"{speaker}: {text}" for speaker, text, _, _ in lines]
    transcript = "\n".join(rendered)
    spans: dict[FieldName, EvidenceSpan] = {}
    payload_ranges: dict[FieldName, tuple[int, int]] = {}
    offset = 0
    for speaker, text, field, evidence in lines:
        line = f"{speaker}: {text}"
        if field is not None and evidence is not None:
            payload_start = offset + len("Patient: ")
            evidence_start = payload_start + text.index(evidence)
            spans[field] = EvidenceSpan(
                start=evidence_start,
                end=evidence_start + len(evidence),
                text=evidence,
            )
            payload_ranges[field] = (payload_start, payload_start + len(text))
        offset += len(line) + 1

    fields = (
        FieldOutput(
            field=FieldName.CHIEF_COMPLAINT,
            state=FieldState.SUPPORTED,
            value=values.cc[1],
            evidence=(spans[FieldName.CHIEF_COMPLAINT],),
        ),
        FieldOutput(
            field=FieldName.DURATION,
            state=FieldState.SUPPORTED,
            value=duration,
            evidence=(spans[FieldName.DURATION],),
        ),
        FieldOutput(
            field=FieldName.SEVERITY,
            state=FieldState.SUPPORTED,
            value=values.sev,
            evidence=(spans[FieldName.SEVERITY],),
        ),
        FieldOutput(
            field=FieldName.MEDICATION,
            state=FieldState.ABSENT if values.med is None else FieldState.SUPPORTED,
            value=values.med,
            evidence=(spans[FieldName.MEDICATION],),
        ),
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.ABSENT if values.alg is None else FieldState.SUPPORTED,
            value=values.alg,
            evidence=(spans[FieldName.ALLERGY],),
        ),
    )
    request = NanoInput(item_id="generation-check", transcript=transcript)
    NanoOutput(
        item_id=request.item_id, solver_id=GOLD_SOLVER_ID, fields=fields
    ).validate_against(request)
    return _NormalRecord(
        transcript=transcript,
        fields=fields,
        payload_ranges=payload_ranges,
        value_band=value_band,
    )


def _has_held_value(values: _TupleRecord, definitions: V1Definitions) -> bool:
    return (
        values.cc in definitions["CC_HELD"]
        or values.med in definitions["MED_HELD"]
        or values.alg in definitions["ALG_HELD"]
    )


def _alternate_value(
    field: FieldName, original: FieldOutput, definitions: V1Definitions
) -> str:
    candidates: Sequence[str]
    if field is FieldName.CHIEF_COMPLAINT:
        candidates = tuple(
            item[1] for item in definitions["CC_TRAIN"] + definitions["CC_HELD"]
        )
    elif field is FieldName.DURATION:
        candidates = ("15 days", "7 weeks")
    elif field is FieldName.SEVERITY:
        candidates = definitions["SEV"]
    elif field is FieldName.MEDICATION:
        candidates = definitions["MED_TRAIN"] + definitions["MED_HELD"]
    else:
        candidates = definitions["ALG_TRAIN"] + definitions["ALG_HELD"]
    original_normalized = (
        "" if original.value is None else _normalized_transcript(original.value)
    )
    return next(
        item
        for item in candidates
        if _normalized_transcript(item) != original_normalized
    )


def _challenge(
    normal: _NormalRecord,
    state: FieldState,
    field: FieldName,
    pair_id: str,
    definitions: V1Definitions,
) -> _ChallengeRecord:
    field_index = FIELD_ORDER.index(field)
    original = normal.fields[field_index]
    fields = list(normal.fields)
    if state in {FieldState.MISSING, FieldState.UNCERTAIN}:
        start, end = normal.payload_ranges[field]
        if state is FieldState.MISSING:
            line_start = start - len("Patient: ")
            replacement = " " * (end - line_start)
            transcript = (
                normal.transcript[:line_start] + replacement + normal.transcript[end:]
            )
            target = FieldOutput(field=field, state=state)
        else:
            token = "Unsure."
            if len(token) > end - start:
                raise BenchmarkIntegrityError(
                    "target payload is too short for uncertainty marker"
                )
            replacement = token + " " * (end - start - len(token))
            target = FieldOutput(
                field=field,
                state=state,
                evidence=(
                    EvidenceSpan(start=start, end=start + len(token), text=token),
                ),
            )
            transcript = (
                normal.transcript[:start] + replacement + normal.transcript[end:]
            )
    else:
        alternate = _alternate_value(field, original, definitions)
        question_names = {
            FieldName.CHIEF_COMPLAINT: "D_OPEN_HELD",
            FieldName.DURATION: "D_DUR_HELD",
            FieldName.SEVERITY: "D_SEV_HELD",
            FieldName.MEDICATION: "D_MED_HELD",
            FieldName.ALLERGY: "D_ALG_HELD",
        }
        question = definitions[question_names[field]][0]
        answer_names = {
            FieldName.CHIEF_COMPLAINT: "P_CC_HELD",
            FieldName.DURATION: "P_DUR_HELD",
            FieldName.SEVERITY: "P_SEV_HELD",
            FieldName.MEDICATION: "P_MED_YES_HELD",
            FieldName.ALLERGY: "P_ALG_YES_HELD",
        }
        answer_template = definitions[answer_names[field]][0]
        if field is FieldName.DURATION:
            number, unit = alternate.split(" ", 1)
            payload = answer_template.format(n=number, unit=unit)
        else:
            placeholder = {
                FieldName.CHIEF_COMPLAINT: "cc",
                FieldName.SEVERITY: "sev",
                FieldName.MEDICATION: "med",
                FieldName.ALLERGY: "alg",
            }[field]
            payload = answer_template.format(**{placeholder: alternate})
        addition = f"\nDoctor: {question}\nPatient: {payload}"
        transcript = normal.transcript + addition
        evidence_start = (
            len(normal.transcript)
            + len(f"\nDoctor: {question}\nPatient: ")
            + payload.index(alternate)
        )
        alternate_span = EvidenceSpan(
            start=evidence_start,
            end=evidence_start + len(alternate),
            text=alternate,
        )
        target = FieldOutput(
            field=field,
            state=state,
            evidence=original.evidence + (alternate_span,),
        )
    fields[field_index] = target
    request = NanoInput(item_id="generation-check", transcript=transcript)
    NanoOutput(
        item_id=request.item_id, solver_id=GOLD_SOLVER_ID, fields=tuple(fields)
    ).validate_against(request)
    return _ChallengeRecord(
        transcript=transcript,
        fields=tuple(fields),
        value_band=normal.value_band,
        target_state=state,
        target_field=field,
        pair_id=pair_id,
        base=normal,
    )


def _generate_records(
    definitions: V1Definitions,
    collision_index: CollisionIndex,
    *,
    seed: int,
) -> tuple[_NormalRecord | _ChallengeRecord, ...]:
    rng = random.Random(seed)
    normals: list[_NormalRecord] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()

    for band, target_count in (
        ("seen", NORMAL_SEEN_RECORDS),
        ("held", NORMAL_HELD_RECORDS),
    ):
        band_count = 0
        attempts = 0
        while band_count < target_count:
            attempts += 1
            if attempts > 100_000:
                raise BenchmarkIntegrityError(
                    f"could not construct unique {band} records"
                )
            values = _sample_tuple(rng, definitions, band == "held")
            if band == "held" and not _has_held_value(values, definitions):
                continue
            normal = _fresh_normal(rng, definitions, values, band)
            exact = _sha256_text(normal.transcript)
            normalized = _sha256_text(_normalized_transcript(normal.transcript))
            if (
                exact in collision_index.exact_hashes
                or normalized in collision_index.normalized_hashes
                or exact in seen_exact
                or normalized in seen_normalized
            ):
                continue
            seen_exact.add(exact)
            seen_normalized.add(normalized)
            normals.append(normal)
            band_count += 1

    challenges: list[_ChallengeRecord] = []
    pair_number = 0
    for band in ("seen", "held"):
        band_indices = [
            index for index, item in enumerate(normals) if item.value_band == band
        ]
        rng.shuffle(band_indices)
        cursor = 0
        for state in _TARGET_STATES:
            for field in FIELD_ORDER:
                for _ in range(2):
                    normal_index = band_indices[cursor]
                    cursor += 1
                    pair_id = f"pair-{pair_number:04d}"
                    pair_number += 1
                    paired = replace(
                        normals[normal_index],
                        pair_id=pair_id,
                        target_state=state,
                        target_field=field,
                    )
                    normals[normal_index] = paired
                    challenges.append(
                        _challenge(paired, state, field, pair_id, definitions)
                    )

    combined: list[_NormalRecord | _ChallengeRecord] = [*normals, *challenges]
    rng.shuffle(combined)
    return tuple(combined)


def _benchmark_metadata(record: _NormalRecord | _ChallengeRecord) -> dict[str, Any]:
    if isinstance(record, _ChallengeRecord):
        return {
            "schema_version": BENCHMARK_PROVENANCE_SCHEMA_VERSION,
            "family": "state_challenge",
            "variant": "challenge",
            "value_band": record.value_band,
            "target_state": record.target_state.value,
            "target_field": record.target_field.value,
            "pair_id": record.pair_id,
        }
    return {
        "schema_version": BENCHMARK_PROVENANCE_SCHEMA_VERSION,
        "family": "normal",
        "variant": "normal",
        "value_band": record.value_band,
        "target_state": (
            record.target_state.value if record.target_state is not None else None
        ),
        "target_field": (
            record.target_field.value if record.target_field is not None else None
        ),
        "pair_id": record.pair_id,
    }


def _partition_document_from_records(
    records: Sequence[_NormalRecord | _ChallengeRecord],
    *,
    seed: int = FRESH_SEED,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        case_id = f"nano-v0-test-{index:04d}"
        output = NanoOutput(
            item_id=case_id, solver_id=GOLD_SOLVER_ID, fields=record.fields
        )
        request = NanoInput(item_id=case_id, transcript=record.transcript)
        output.validate_against(request)
        cases.append(
            {
                "case_id": case_id,
                "transcript": record.transcript,
                "transcript_sha256": _sha256_text(record.transcript),
                "gold": output.to_dict(),
                "benchmark": _benchmark_metadata(record),
            }
        )
    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "seed": seed,
        "generator_sha256": GENERATOR_SHA256,
        "cases": cases,
    }


def build_fresh_partition_document(repository_root: Path) -> dict[str, Any]:
    """Build the fresh partition in memory without running inference or writing it."""

    definitions = load_v1_definitions(repository_root)
    collision_index = build_collision_index(repository_root)
    document = _partition_document_from_records(
        _generate_records(definitions, collision_index, seed=FRESH_SEED)
    )
    _validate_partition(document, collision_index)
    return document


def _composition() -> dict[str, int]:
    return {
        "normal": NORMAL_RECORDS,
        "normal_seen": NORMAL_SEEN_RECORDS,
        "normal_held": NORMAL_HELD_RECORDS,
        "state_challenge": CHALLENGE_RECORDS,
        "missing": STATE_CHALLENGE_RECORDS,
        "uncertain": STATE_CHALLENGE_RECORDS,
        "conflicting": STATE_CHALLENGE_RECORDS,
        "paired_normal": PAIRED_NORMAL_RECORDS,
    }


def build_sealed_manifest(
    partition_document: Mapping[str, Any],
    *,
    partition_path: str,
    repository_root: Path,
    status: str = "fresh_unmeasured",
) -> dict[str, Any]:
    """Create a manifest for a separately persisted canonical partition snapshot."""

    if status not in _MANIFEST_STATUSES:
        raise BenchmarkIntegrityError("invalid benchmark status")
    relative = PurePosixPath(partition_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise BenchmarkIntegrityError("partition path must be a safe relative path")
    collision_index = build_collision_index(repository_root)
    _validate_partition(dict(partition_document), collision_index)
    partition_bytes = canonical_json_bytes(partition_document)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "seed": FRESH_SEED,
        "generator": {"path": GENERATOR_PATH, "sha256": GENERATOR_SHA256},
        "partition": {
            "path": partition_path,
            "records": TOTAL_RECORDS,
            "sha256": _sha256_bytes(partition_bytes),
        },
        "composition": _composition(),
        "collision_index": collision_index.manifest_dict(),
    }


def _load_case(row: Any, index: int) -> FixtureCase:
    case = _require_exact_keys(row, _CASE_KEYS, f"case {index}")
    case_id = _require_string(case["case_id"], f"case {index} ID")
    expected_id = f"nano-v0-test-{index:04d}"
    if case_id != expected_id:
        raise BenchmarkIntegrityError(
            "benchmark case IDs are not neutral and sequential"
        )
    transcript = case["transcript"]
    if not isinstance(transcript, str) or not transcript.strip():
        raise BenchmarkIntegrityError(f"invalid transcript: {case_id}")
    digest = _require_sha256(case["transcript_sha256"], f"transcript digest {case_id}")
    if _sha256_text(transcript) != digest:
        raise BenchmarkIntegrityError(f"transcript digest mismatch: {case_id}")
    benchmark = _require_exact_keys(
        case["benchmark"], _BENCHMARK_KEYS, f"benchmark metadata {case_id}"
    )
    if benchmark["schema_version"] != BENCHMARK_PROVENANCE_SCHEMA_VERSION:
        raise BenchmarkIntegrityError(
            f"benchmark provenance version mismatch: {case_id}"
        )
    try:
        request = NanoInput(item_id=case_id, transcript=transcript)
        output = NanoOutput.from_dict(case["gold"])
        if output.solver_id != GOLD_SOLVER_ID:
            raise BenchmarkIntegrityError(f"invalid gold solver ID: {case_id}")
        fixture = FixtureCase(
            case_id=case_id,
            partition=BENCHMARK_ID,
            request=request,
            gold=output,
            provenance={"benchmark": dict(benchmark), "transcript_sha256": digest},
        )
    except BenchmarkIntegrityError:
        raise
    except ValueError as exc:
        raise BenchmarkIntegrityError(
            f"invalid case contract: {case_id}: {exc}"
        ) from exc
    return fixture


def _validate_partition(
    value: Any,
    collision_index: CollisionIndex,
) -> tuple[FixtureCase, ...]:
    root = _require_exact_keys(value, _PARTITION_KEYS, "partition")
    if root["schema_version"] != PARTITION_SCHEMA_VERSION:
        raise BenchmarkIntegrityError("partition schema version mismatch")
    if root["benchmark_id"] != BENCHMARK_ID:
        raise BenchmarkIntegrityError("partition benchmark ID mismatch")
    _require_int(root["seed"], "partition seed", expected=FRESH_SEED)
    if root["generator_sha256"] != GENERATOR_SHA256:
        raise BenchmarkIntegrityError("partition generator digest mismatch")
    rows = root["cases"]
    if not isinstance(rows, list) or len(rows) != TOTAL_RECORDS:
        raise BenchmarkIntegrityError("partition must contain exactly 220 cases")
    cases = tuple(_load_case(row, index) for index, row in enumerate(rows))

    exact_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    pair_members: dict[str, list[FixtureCase]] = {}
    normal_seen = normal_held = normal_count = challenge_count = paired_normal = 0
    state_counts = {state.value: 0 for state in _TARGET_STATES}
    quota = {
        (state.value, field.value, band): 0
        for state in _TARGET_STATES
        for field in FIELD_ORDER
        for band in _VALUE_BANDS
    }
    for case in cases:
        transcript = case.request.transcript
        exact = _sha256_text(transcript)
        normalized = _sha256_text(_normalized_transcript(transcript))
        if exact in exact_hashes or normalized in normalized_hashes:
            raise BenchmarkIntegrityError(
                "partition transcripts are not internally unique"
            )
        if (
            exact in collision_index.exact_hashes
            or normalized in collision_index.normalized_hashes
        ):
            raise BenchmarkIntegrityError(
                "partition transcript collides with historical data"
            )
        exact_hashes.add(exact)
        normalized_hashes.add(normalized)

        metadata = case.provenance["benchmark"]
        family = metadata["family"]
        variant = metadata["variant"]
        band = metadata["value_band"]
        pair_id = metadata["pair_id"]
        if band not in _VALUE_BANDS:
            raise BenchmarkIntegrityError("invalid benchmark value band")
        if pair_id is not None:
            if not isinstance(pair_id, str) or not pair_id.startswith("pair-"):
                raise BenchmarkIntegrityError("invalid pair ID")
            pair_members.setdefault(pair_id, []).append(case)
        if family == "normal":
            metadata_nulls = sum(
                item is None
                for item in (
                    metadata["target_state"],
                    metadata["target_field"],
                    pair_id,
                )
            )
            if variant != "normal" or metadata_nulls not in {0, 3}:
                raise BenchmarkIntegrityError("invalid normal-case metadata")
            if metadata_nulls == 0:
                try:
                    normal_target_state = FieldState(metadata["target_state"])
                    FieldName(metadata["target_field"])
                except (TypeError, ValueError) as exc:
                    raise BenchmarkIntegrityError(
                        "invalid paired-normal target"
                    ) from exc
                if normal_target_state not in _TARGET_STATES:
                    raise BenchmarkIntegrityError("invalid paired-normal target state")
            if any(
                field.state not in {FieldState.SUPPORTED, FieldState.ABSENT}
                for field in case.gold.fields
            ):
                raise BenchmarkIntegrityError(
                    "normal case contains an abstention state"
                )
            patient_lines = [
                line for line in transcript.splitlines() if line.startswith("Patient: ")
            ]
            if len(patient_lines) != len(FIELD_ORDER):
                raise BenchmarkIntegrityError(
                    "normal case must contain exactly five patient answer turns"
                )
            held_present = (
                case.gold.field(FieldName.CHIEF_COMPLAINT).value
                in collision_index.held_complaints
                or case.gold.field(FieldName.MEDICATION).value
                in collision_index.held_medications
                or case.gold.field(FieldName.ALLERGY).value
                in collision_index.held_allergies
            )
            if held_present != (band == "held"):
                raise BenchmarkIntegrityError(
                    "normal value band disagrees with held vocabularies"
                )
            normal_count += 1
            normal_seen += band == "seen"
            normal_held += band == "held"
            paired_normal += pair_id is not None
        elif family == "state_challenge":
            if variant != "challenge" or pair_id is None:
                raise BenchmarkIntegrityError("invalid challenge-case metadata")
            try:
                state = FieldState(metadata["target_state"])
                field = FieldName(metadata["target_field"])
            except (TypeError, ValueError) as exc:
                raise BenchmarkIntegrityError("invalid challenge target") from exc
            if state not in _TARGET_STATES or case.gold.field(field).state is not state:
                raise BenchmarkIntegrityError(
                    "challenge target state disagrees with gold"
                )
            if sum(item.state in _TARGET_STATES for item in case.gold.fields) != 1:
                raise BenchmarkIntegrityError(
                    "challenge must change exactly one field state"
                )
            challenge_count += 1
            state_counts[state.value] += 1
            quota[(state.value, field.value, band)] += 1
        else:
            raise BenchmarkIntegrityError("invalid benchmark family")

    if (
        normal_count != NORMAL_RECORDS
        or normal_seen != NORMAL_SEEN_RECORDS
        or normal_held != NORMAL_HELD_RECORDS
        or challenge_count != CHALLENGE_RECORDS
        or paired_normal != PAIRED_NORMAL_RECORDS
        or any(value != STATE_CHALLENGE_RECORDS for value in state_counts.values())
        or any(value != 2 for value in quota.values())
    ):
        raise BenchmarkIntegrityError(
            "partition composition or target quotas are invalid"
        )
    if len(pair_members) != CHALLENGE_RECORDS or any(
        len(items) != 2 for items in pair_members.values()
    ):
        raise BenchmarkIntegrityError("benchmark pairs are incomplete or duplicated")

    for pair_id, members in pair_members.items():
        normal = next(
            (
                item
                for item in members
                if item.provenance["benchmark"]["family"] == "normal"
            ),
            None,
        )
        challenge = next(
            (
                item
                for item in members
                if item.provenance["benchmark"]["family"] == "state_challenge"
            ),
            None,
        )
        if normal is None or challenge is None:
            raise BenchmarkIntegrityError(
                f"pair lacks one normal and one challenge: {pair_id}"
            )
        normal_meta = normal.provenance["benchmark"]
        challenge_meta = challenge.provenance["benchmark"]
        if normal_meta["value_band"] != challenge_meta["value_band"]:
            raise BenchmarkIntegrityError(f"pair value bands disagree: {pair_id}")
        if (
            normal_meta["target_state"] != challenge_meta["target_state"]
            or normal_meta["target_field"] != challenge_meta["target_field"]
        ):
            raise BenchmarkIntegrityError(f"pair target metadata disagrees: {pair_id}")
        target_field = FieldName(challenge_meta["target_field"])
        target_state = FieldState(challenge_meta["target_state"])
        for field in FIELD_ORDER:
            if (
                field is not target_field
                and normal.gold.field(field).to_dict()
                != challenge.gold.field(field).to_dict()
            ):
                raise BenchmarkIntegrityError(
                    f"challenge changed unaffected gold: {pair_id}"
                )
        if target_state in {FieldState.MISSING, FieldState.UNCERTAIN}:
            if len(normal.request.transcript) != len(challenge.request.transcript):
                raise BenchmarkIntegrityError(
                    f"replacement challenge changed transcript length: {pair_id}"
                )
            target_spans = normal.gold.field(target_field).evidence
            if not target_spans:
                raise BenchmarkIntegrityError(
                    f"normal target lacks evidence: {pair_id}"
                )
            first = min(span.start for span in target_spans)
            line_start = normal.request.transcript.rfind("\n", 0, first) + 1
            line_end = normal.request.transcript.find("\n", first)
            if line_end < 0:
                line_end = len(normal.request.transcript)
            prefix = "Patient: "
            if not normal.request.transcript.startswith(prefix, line_start):
                raise BenchmarkIntegrityError(
                    f"target evidence is not in a patient answer: {pair_id}"
                )
            allowed_start = (
                line_start
                if target_state is FieldState.MISSING
                else line_start + len(prefix)
            )
            differences = [
                index
                for index, (left, right) in enumerate(
                    zip(normal.request.transcript, challenge.request.transcript)
                )
                if left != right
            ]
            if (
                not differences
                or min(differences) < allowed_start
                or max(differences) >= line_end
                or not any(
                    span.start <= index < span.end
                    for span in target_spans
                    for index in differences
                )
            ):
                raise BenchmarkIntegrityError(
                    f"replacement challenge changed more than the target answer: {pair_id}"
                )
        else:
            prefix = normal.request.transcript + "\nDoctor: "
            if not challenge.request.transcript.startswith(prefix):
                raise BenchmarkIntegrityError(
                    f"conflict challenge is not append-only: {pair_id}"
                )
            appended = challenge.request.transcript[
                len(normal.request.transcript) + 1 :
            ]
            appended_lines = appended.splitlines()
            if (
                len(appended_lines) != 2
                or not appended_lines[0].startswith("Doctor: ")
                or not appended_lines[1].startswith("Patient: ")
            ):
                raise BenchmarkIntegrityError(
                    f"conflict challenge must append one question and answer: {pair_id}"
                )
            normal_target = normal.gold.field(target_field)
            challenge_target = challenge.gold.field(target_field)
            if (
                tuple(challenge_target.evidence[: len(normal_target.evidence)])
                != normal_target.evidence
            ):
                raise BenchmarkIntegrityError(
                    f"conflict challenge lost original evidence: {pair_id}"
                )
    return cases


def _validate_manifest(value: Any, collision_index: CollisionIndex) -> dict[str, Any]:
    root = _require_exact_keys(value, _MANIFEST_KEYS, "benchmark manifest")
    if root["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise BenchmarkIntegrityError("manifest schema version mismatch")
    if (
        root["benchmark_id"] != BENCHMARK_ID
        or root["contract_version"] != CONTRACT_VERSION
    ):
        raise BenchmarkIntegrityError("manifest benchmark or contract version mismatch")
    if root["status"] not in _MANIFEST_STATUSES:
        raise BenchmarkIntegrityError("manifest status is invalid")
    _require_int(root["seed"], "manifest seed", expected=FRESH_SEED)
    generator = _require_exact_keys(
        root["generator"], _GENERATOR_KEYS, "generator declaration"
    )
    if generator != {"path": GENERATOR_PATH, "sha256": GENERATOR_SHA256}:
        raise BenchmarkIntegrityError("manifest generator declaration mismatch")
    partition = _require_exact_keys(
        root["partition"], _PARTITION_SPEC_KEYS, "partition declaration"
    )
    _require_string(partition["path"], "partition path")
    _require_int(partition["records"], "partition records", expected=TOTAL_RECORDS)
    _require_sha256(partition["sha256"], "partition digest")
    composition = _require_exact_keys(
        root["composition"], _COMPOSITION_KEYS, "composition"
    )
    if composition != _composition() or any(
        type(item) is not int for item in composition.values()
    ):
        raise BenchmarkIntegrityError("manifest composition mismatch")
    declared_collision = _require_exact_keys(
        root["collision_index"], _COLLISION_KEYS, "collision index"
    )
    if declared_collision != collision_index.manifest_dict():
        raise BenchmarkIntegrityError("manifest collision index mismatch")
    return root


def load_sealed_benchmark(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    repository_root: Path,
) -> FrozenBenchmark:
    """Load exactly one externally sealed manifest and partition byte snapshot."""

    expected = _require_sha256(expected_manifest_sha256, "expected manifest digest")
    manifest_path = Path(manifest_path)
    manifest_snapshot = _read_regular_snapshot(manifest_path, "benchmark manifest")
    manifest_digest = _sha256_bytes(manifest_snapshot)
    if manifest_digest != expected:
        raise BenchmarkIntegrityError("benchmark manifest seal mismatch")
    manifest_value = _parse_json_snapshot(manifest_snapshot, "benchmark manifest")
    if canonical_json_bytes(manifest_value) != manifest_snapshot:
        raise BenchmarkIntegrityError("benchmark manifest is not canonical JSON")

    collision_index = build_collision_index(Path(repository_root))
    manifest = _validate_manifest(manifest_value, collision_index)
    generator_snapshot = _read_repo_snapshot(
        repository_root, GENERATOR_PATH, "v1 generator"
    )
    if _sha256_bytes(generator_snapshot) != manifest["generator"]["sha256"]:
        raise BenchmarkIntegrityError("benchmark generator source changed")

    partition_path = _artifact_sibling(manifest_path, manifest["partition"]["path"])
    partition_snapshot = _read_regular_snapshot(partition_path, "benchmark partition")
    partition_digest = _sha256_bytes(partition_snapshot)
    if partition_digest != manifest["partition"]["sha256"]:
        raise BenchmarkIntegrityError("benchmark partition seal mismatch")
    partition_value = _parse_json_snapshot(partition_snapshot, "benchmark partition")
    if canonical_json_bytes(partition_value) != partition_snapshot:
        raise BenchmarkIntegrityError("benchmark partition is not canonical JSON")
    cases = _validate_partition(partition_value, collision_index)
    return FrozenBenchmark(
        benchmark_id=manifest["benchmark_id"],
        status=manifest["status"],
        manifest_sha256=manifest_digest,
        partition_sha256=partition_digest,
        cases=cases,
    )


def load_sealed_partition(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    repository_root: Path,
) -> tuple[FixtureCase, ...]:
    """Compatibility wrapper returning only cases from a sealed benchmark."""

    return load_sealed_benchmark(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        repository_root=repository_root,
    ).cases


__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_PROVENANCE_SCHEMA_VERSION",
    "FRESH_SEED",
    "GENERATOR_PATH",
    "GENERATOR_SHA256",
    "GOLD_SOLVER_ID",
    "HISTORICAL_EXACT_MULTISET_SHA256",
    "HISTORICAL_NORMALIZED_MULTISET_SHA256",
    "HISTORICAL_RECORDS",
    "HISTORICAL_SOURCE_PATHS",
    "HISTORICAL_UNIQUE_TRANSCRIPTS",
    "MANIFEST_SCHEMA_VERSION",
    "PARTITION_SCHEMA_VERSION",
    "BenchmarkIntegrityError",
    "CollisionIndex",
    "FrozenBenchmark",
    "build_collision_index",
    "build_fresh_partition_document",
    "build_sealed_manifest",
    "canonical_json_bytes",
    "load_sealed_benchmark",
    "load_sealed_partition",
    "load_v1_definitions",
]
