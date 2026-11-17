"""Offline, annotation-owned P1 development comparison; never frozen evaluation.

Primary scoring admits natural-language annotations independently of the
product's denial vocabulary. Exact spans and normalized content are distinct
metrics; neither substitutes for human clinical evaluation. Run --preflight
without a checkpoint, or supply --checkpoint and a new --output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import (
    FIELD_ORDER, EvidenceSpan, FieldOutput, FieldState, NanoInput, NanoOutput,
    normalize_value,
)
from nano_ai.evaluation import evaluate_solver
from nano_ai.fixtures import FixtureCase
from nano_ai.solver import SolverDescriptor, SolverKind
from nanoscribe.compact import FEWSHOT_PROMPT, STRUCTURED_PROMPT, LocalCompactPredictor, compact_solver
from nanoscribe.scribe import scribe

LABELS = dict(zip(("CC", "DUR", "SEV", "MED", "ALG"), FIELD_ORDER, strict=True))
STATES = dict(zip("SAMUC", (FieldState.SUPPORTED, FieldState.ABSENT, FieldState.MISSING,
                           FieldState.UNCERTAIN, FieldState.CONFLICTING), strict=True))
ASSERTED = {"supported", "absent"}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_unique_object)


def save(output, name, value):
    """Atomically replace this run's progress files, never another run directory."""
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify_lock(data, expected=None):
    """Validate every preregistered annotation input and the lock's own bytes."""
    data = Path(data).resolve()
    lock = read_json(data / "annotation_lock.json")
    files = lock.get("files") if isinstance(lock, dict) else None
    if not isinstance(files, dict) or not {"sources.json", "gold.json"} <= set(files):
        raise ValueError("annotation_lock.json must hash sources.json and gold.json in files")
    observed = {"annotation_lock.json": digest(data / "annotation_lock.json")}
    for name, sha256 in files.items():
        if (Path(name).is_absolute() or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError(f"invalid annotation lock path: {name}")
        path = (data / name).resolve()
        if not path.is_relative_to(data) or name == "annotation_lock.json" or not path.is_file():
            raise ValueError(f"invalid annotation lock path: {name}")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"invalid annotation lock digest: {name}")
        observed[name] = digest(path)
        if observed[name] != sha256:
            raise ValueError(f"annotation lock hash mismatch: {name}")
    if expected is not None and observed != expected:
        raise ValueError("annotation inputs or lock changed after admission")
    return observed


@dataclass(frozen=True)
class AnnotationCase:
    case_id: str
    stratum: str
    request: NanoInput
    fields: tuple[dict, ...]
    transport_errors: tuple[dict, ...]
    fixture: FixtureCase | None


def _evidence(transcript, text):
    if isinstance(text, dict):
        if (set(text) != {"text", "start"} or type(text["start"]) is not int
                or not isinstance(text["text"], str)):
            raise ValueError("explicit annotation evidence requires text and integer start")
        span = EvidenceSpan(text["start"], text["start"] + len(text["text"]), text["text"])
        span.validate_against(transcript)
        return span
    if not isinstance(text, str) or not text or text.strip() != text:
        raise ValueError("annotation evidence must be nonempty edge-trimmed text")
    # Count overlapping occurrences, so 'aa' in 'aaa' is also ambiguous.
    positions = [i for i in range(len(transcript)) if transcript.startswith(text, i)]
    if len(positions) != 1:
        raise ValueError(f"annotation evidence must occur exactly once: {text!r}")
    span = EvidenceSpan(positions[0], positions[0] + len(text), text)
    span.validate_against(transcript)
    return span


def _field_annotation(label, item, transcript):
    if not isinstance(item, dict) or not {"state", "spans"} <= set(item):
        raise ValueError(f"{label} requires state and spans")
    if not isinstance(item["state"], str) or item["state"] not in STATES or not isinstance(item["spans"], list):
        raise ValueError(f"invalid {label} annotation state or spans")
    state = STATES[item["state"]]
    if state is FieldState.ABSENT and label not in ("MED", "ALG"):
        raise ValueError("A annotations are restricted to MED/ALG by the annotation rubric")
    spans = tuple(_evidence(transcript, text) for text in item["spans"])
    if len({(s.start, s.end) for s in spans}) != len(spans):
        raise ValueError("duplicate annotation evidence")
    if state is FieldState.MISSING and spans:
        raise ValueError("M annotations cannot have evidence")
    if state in (FieldState.SUPPORTED, FieldState.ABSENT) and not spans:
        raise ValueError("S/A annotations require source evidence")
    if state is FieldState.CONFLICTING and len({normalize_value(s.text) for s in spans}) < 2:
        raise ValueError("C annotations require two textually distinct observations")
    return {"field": LABELS[label].value, "state": state.value,
            "value": normalize_value(spans[0].text) if state is FieldState.SUPPORTED else None,
            "evidence": [span.to_dict() for span in spans]}


def admit(data):
    """Use annotations, never extraction rules, as evaluation truth.

    Natural explicit denials outside the existing product contract remain in
    the primary denominator. Transport failures are reported, not relabeled.
    """
    hashes = verify_lock(data)
    rows, gold = read_json(Path(data) / "sources.json"), read_json(Path(data) / "gold.json")
    if not isinstance(rows, list) or not rows or not isinstance(gold, list):
        raise ValueError("sources and gold must be nonempty lists")
    if any(not isinstance(row, dict) or not {"case_id", "stratum", "transcript"} <= set(row) for row in rows):
        raise ValueError("source records require case_id, stratum and transcript")
    if any(not isinstance(row, dict) or not {"case_id", "fields"} <= set(row) for row in gold):
        raise ValueError("gold records require case_id and fields")
    source_ids, gold_ids = [r["case_id"] for r in rows], [r["case_id"] for r in gold]
    if len(set(source_ids)) != len(rows) or len(set(gold_ids)) != len(gold):
        raise ValueError("duplicate annotation case ID")
    if set(source_ids) != set(gold_ids):
        raise ValueError("source and gold IDs must match exactly")
    if len({r["transcript"] for r in rows}) != len(rows):
        raise ValueError("duplicate source transcript")
    gold_by_id = {r["case_id"]: r for r in gold}
    cases = []
    for row in rows:
        request = NanoInput(row["case_id"], row["transcript"])
        if not isinstance(row["stratum"], str) or not row["stratum"].strip():
            raise ValueError("source stratum must be nonempty text")
        annotations = gold_by_id[row["case_id"]]["fields"]
        if not isinstance(annotations, dict) or set(annotations) != set(LABELS):
            raise ValueError("gold requires exactly CC, DUR, SEV, MED, ALG")
        fields = tuple(_field_annotation(label, annotations[label], request.transcript) for label in LABELS)
        transported, errors = [], []
        for field in fields:
            try:
                transported.append(FieldOutput.from_dict(field))
            except ValueError as exc:
                errors.append({"field": field["field"], "error": str(exc)})
        fixture = None
        if not errors:
            fixture = FixtureCase(request.item_id, "development", request,
                                  NanoOutput(request.item_id, "independent-agent-annotations-v1", tuple(transported)),
                                  {"role": "source_annotation_transport", "stratum": row["stratum"]})
        cases.append(AnnotationCase(request.item_id, row["stratum"], request, fields, tuple(errors), fixture))
    verify_lock(data, hashes)
    return rows, tuple(cases), hashes


def raw_fields(raw, transcript, *, policy="context"):
    """Parse exact source syntax without imposing the product's denial whitelist."""
    if policy == "grounded":
        from nanoscribe.grounded import parse_grounded_summary
        _, proposals, errors, strict_error = parse_grounded_summary(raw, transcript)
        if strict_error is not None or any(errors) or any(p is None for p in proposals):
            raise ValueError(strict_error or "malformed grounded summary")
        parsed = proposals
    else:
        parsed = parse_state_span_summary(raw, transcript)
    return tuple({"field": p.field.value, "state": p.state.value,
                  "value": normalize_value(p.spans[0].text) if p.state is FieldState.SUPPORTED else None,
                  "evidence": [span.to_dict() for span in p.spans]}
                 for p in parsed)


def field_score(predicted, gold):
    if predicted is None:
        return {"state_correct": False, "normalized_content_correct": False,
                "exact_evidence_aligned": False, "grounded_exact_correct": False,
                "complete_evidence_exact_correct": False}
    same_state = predicted["state"] == gold["state"]
    content = same_state and (gold["state"] != "supported" or
                             normalize_value(predicted["value"]) == normalize_value(gold["value"]))
    def keys(field):
        return {(s["start"], s["end"], s["text"], s["speaker"]) for s in field["evidence"]}
    predicted_keys, gold_keys = keys(predicted), keys(gold)
    aligned = same_state and (
        gold["state"] in ("missing", "uncertain") or
        (gold_keys <= predicted_keys if gold["state"] == "conflicting"
         else bool(predicted_keys & gold_keys)))
    return {"state_correct": same_state, "normalized_content_correct": content,
            "exact_evidence_aligned": aligned, "grounded_exact_correct": content and aligned,
            "complete_evidence_exact_correct": content and keys(predicted) == keys(gold)}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _note_field(draft, field):
    if draft.get("note") is None:
        return None
    claim = next(c for c in draft["note"]["claims"] if c["field"] == field)
    spans = {s["evidence_id"]: s for s in draft["encounter"]["evidence"]}
    return {"field": field, "state": claim["state"], "value": claim["value"],
            "evidence": [{key: spans[ref][key] for key in ("start", "end", "text", "speaker")}
                         for ref in claim["evidence_ids"]]}


def _group_scores(details, grouping):
    result = {}
    for group in sorted({row[grouping] for row in details}):
        rows = [row for row in details if row[grouping] == group]
        result[group] = {path: {"fields": len(rows), **{
            name: sum(row[path][key] for row in rows) / len(rows)
            for name, key in (("state_accuracy", "state_correct"),
                              ("grounded_exact_field_accuracy", "grounded_exact_correct"))}}
                        for path in ("raw", "checked")}
    return result


def compare(cases, encounters):
    """Paired, full-denominator scores using the same generation for each path."""
    by_id = {r["case_id"]: r for r in encounters}
    if len(by_id) != len(encounters) or set(by_id) != {c.case_id for c in cases}:
        raise ValueError("comparison requires exactly one encounter for every admitted case")
    counters = {path: Counter() for path in ("raw", "checked")}
    confusions = {path: {} for path in counters}
    transitions, final_note, details = Counter(), Counter(), []
    malformed = checked_failures = avoidable_abstentions = usable_cases = 0
    for case in cases:
        encounter = by_id[case.case_id]
        malformed += encounter["raw_parse_error"] is not None
        checked_failures += encounter["draft"].get("extraction") is None
        usable = any(g["state"] in ASSERTED for g in case.fields)
        usable_cases += usable
        avoidable_abstentions += usable and encounter["draft"].get("disposition") == "abstain"
        outputs = {"raw": encounter.get("raw_fields"), "checked":
                   (encounter["draft"].get("extraction") or {}).get("fields")}
        for path, fields in outputs.items():
            if fields is not None and [f["field"] for f in fields] != [f.value for f in FIELD_ORDER]:
                raise ValueError(f"{path} fields must use the complete canonical order")
        for index, gold in enumerate(case.fields):
            pair = {"case_id": case.case_id, "stratum": case.stratum, "field": gold["field"], "gold": gold}
            for path in counters:
                predicted = outputs[path][index] if outputs[path] is not None else None
                score = field_score(predicted, gold)
                pair[path] = {"prediction": predicted, **score}
                count = counters[path]
                count["fields"] += 1
                for metric, value in score.items():
                    count[metric] += value
                asserted = predicted is not None and predicted["state"] in ASSERTED
                count["emitted_asserted_fields"] += asserted
                count["asserted_state_or_content_mismatch"] += asserted and not score["normalized_content_correct"]
                count["asserted_exact_evidence_mismatch"] += asserted and not score["exact_evidence_aligned"]
                count["asserted_exact_field_mismatch"] += asserted and not score["grounded_exact_correct"]
                count["gold_asserted_fields"] += gold["state"] in ASSERTED
                count["correct_asserted_fields"] += gold["state"] in ASSERTED and score["grounded_exact_correct"]
                count["gold_conflict_fields"] += gold["state"] == "conflicting"
                count["conflict_state_recalled"] += gold["state"] == "conflicting" and score["state_correct"]
                count["conflict_exact_recalled"] += gold["state"] == "conflicting" and score["grounded_exact_correct"]
                confusion = confusions[path].setdefault(gold["state"], Counter())
                confusion[predicted["state"] if predicted is not None else "failure"] += 1
            raw, checked = pair["raw"], pair["checked"]
            transitions["exact_fields_salvaged"] += not raw["grounded_exact_correct"] and checked["grounded_exact_correct"]
            transitions["exact_fields_lost"] += raw["grounded_exact_correct"] and not checked["grounded_exact_correct"]
            transitions["correct_asserted_fields_lost"] += gold["state"] in ASSERTED and raw["grounded_exact_correct"] and not checked["grounded_exact_correct"]
            transitions["correct_asserted_fields_salvaged"] += gold["state"] in ASSERTED and not raw["grounded_exact_correct"] and checked["grounded_exact_correct"]
            transitions["correct_asserted_fields_retained"] += gold["state"] in ASSERTED and raw["grounded_exact_correct"] and checked["grounded_exact_correct"]
            for kind, key in (("normalized_content", "normalized_content_correct"), ("exact", "grounded_exact_correct")):
                mismatches = {path: pair[path]["prediction"] is not None and pair[path]["prediction"]["state"] in ASSERTED
                              and not pair[path][key] for path in ("raw", "checked")}
                transitions[f"asserted_{kind}_mismatches_removed"] += mismatches["raw"] and not mismatches["checked"]
                transitions[f"asserted_{kind}_mismatches_introduced"] += not mismatches["raw"] and mismatches["checked"]
            note = _note_field(encounter["draft"], gold["field"])
            note_score = field_score(note, gold)
            pair["final_note"] = {"prediction": note, **note_score}
            note_asserted = note is not None and note["state"] in ASSERTED
            final_note["emitted_asserted_fields"] += note_asserted
            final_note["asserted_state_or_content_mismatch"] += note_asserted and not note_score["normalized_content_correct"]
            final_note["asserted_exact_field_mismatch"] += note_asserted and not note_score["grounded_exact_correct"]
            final_note["gold_asserted_fields"] += gold["state"] in ASSERTED
            final_note["correct_asserted_fields_emitted"] += gold["state"] in ASSERTED and note_score["grounded_exact_correct"]
            details.append(pair)
    metrics = {}
    for path, count in counters.items():
        metrics[path] = {**dict(count), "state_accuracy": _rate(count["state_correct"], count["fields"]),
                         "normalized_content_accuracy": _rate(count["normalized_content_correct"], count["fields"]),
                         "grounded_exact_field_accuracy": _rate(count["grounded_exact_correct"], count["fields"]),
                         "complete_evidence_exact_field_accuracy": _rate(count["complete_evidence_exact_correct"], count["fields"]),
                         "conflict_state_recall": _rate(count["conflict_state_recalled"], count["gold_conflict_fields"]),
                         "conflict_exact_recall": _rate(count["conflict_exact_recalled"], count["gold_conflict_fields"]),
                         "state_confusion": confusions[path]}
    return {"cases": len(cases), "fields": len(cases) * len(FIELD_ORDER), "metrics": metrics,
            "by_field": _group_scores(details, "field"), "by_stratum": _group_scores(details, "stratum"),
            "final_note": dict(final_note),
            "paired_changes": dict(transitions), "malformed_raw_cases": malformed,
            "checked_inference_failure_cases": checked_failures,
            "gold_usable_cases": usable_cases,
            "whole_note_abstention_with_gold_usable_content": avoidable_abstentions,
            "whole_note_abstention_with_gold_usable_content_rate": _rate(avoidable_abstentions, usable_cases),
            "definitions": {
                "state_accuracy": "Correct state; all fields of malformed/missing outputs count incorrect.",
                "normalized_content_accuracy": "Correct state and, for S, normalized first annotated value; span boundaries ignored.",
                "grounded_exact_field_accuracy": "Correct normalized content plus at least one exact annotated span for S/A or every annotated span for C; U/M need only the state.",
                "complete_evidence_exact_field_accuracy": "Correct normalized content and identical complete evidence sets, including U/C evidence.",
                "checked": "Checked extraction scores precede note withholding; final_note separately measures actually emitted claims.",
                "asserted_exact_field_mismatch": "Exact annotation mismatch, not an adjudicated clinical error count.",
                "whole_note_abstention": "ABSTAIN disposition on a source with at least one gold S/A; it does not establish the safety of a draft.",
                "independence": "Agent annotation independence must be documented separately; not human or clinician evaluation.",
            }}, details


class Replay:
    def __init__(self, cases):
        self.outputs = {c.case_id: c.gold for c in cases}
        self.descriptor = SolverDescriptor("control/annotation-oracle", SolverKind.REFERENCE)

    def infer(self, request):
        return NanoOutput(request.item_id, self.descriptor.solver_id, self.outputs[request.item_id].fields)


def preflight(data):
    rows, cases, hashes = admit(data)
    oracle, negative = [], []
    for case in cases:
        for collection, fields in ((oracle, case.fields), (negative, tuple(
                {"field": f.value, "state": "missing", "value": None, "evidence": []} for f in FIELD_ORDER))):
            collection.append({"case_id": case.case_id, "raw_fields": fields, "raw_parse_error": None,
                               "draft": {"extraction": {"fields": fields}, "disposition": "review"}})
    oracle_score, _ = compare(cases, oracle)
    negative_score, _ = compare(cases, negative)
    assert oracle_score["metrics"]["raw"]["grounded_exact_field_accuracy"] == 1.0
    missing = sum(f["state"] == "missing" for c in cases for f in c.fields)
    assert negative_score["metrics"]["raw"]["state_correct"] == missing
    transportable = tuple(c.fixture for c in cases if c.fixture is not None)
    secondary = evaluate_solver(Replay(transportable), transportable).to_dict() if transportable else None
    controls = {"input_hashes": hashes, "primary_oracle": oracle_score, "primary_all_missing": negative_score,
                "field_state_counts": {f.value: dict(Counter(c.fields[i]["state"] for c in cases)) for i, f in enumerate(FIELD_ORDER)},
                "contract_transport_errors": [{"case_id": c.case_id, **e} for c in cases for e in c.transport_errors],
                "secondary_transportable_case_count": len(transportable), "secondary_transportable_oracle": secondary}
    return rows, cases, controls


class OncePredictor:
    """One call per transcript; inference never receives evaluator annotations."""
    def __init__(self, predictor):
        self.predictor, self.records = predictor, {}

    def __call__(self, transcript):
        if transcript in self.records:
            raise RuntimeError("duplicate model invocation forbidden in paired comparison")
        self.records[transcript] = {}
        before = time.perf_counter()
        try:
            raw = self.predictor(transcript)
            self.records[transcript] = {**getattr(self.predictor, "records", {}).get(transcript, {}), "raw": raw}
            return raw
        except Exception as exc:
            self.records[transcript] = {"generation_error": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            self.records[transcript]["wall_seconds"] = time.perf_counter() - before


def collect(cases, predictor, output, *, version="independent-development-v1", policy="context"):
    """Evaluate checked extraction and note, then parse the saved same raw output."""
    once = OncePredictor(predictor)
    solver = compact_solver(once, version=version, binder=policy)
    encounters = []
    for case in cases:
        before = time.perf_counter()
        draft = scribe(case.request.transcript, encounter_id=case.case_id, solver=solver, context_policy=policy)
        record = once.records.get(case.request.transcript, {})
        try:
            raw = raw_fields(record.get("raw"), case.request.transcript, policy=policy)
            error = None
        except (TypeError, ValueError) as exc:
            raw, error = None, str(exc)
        transport_errors = []
        for field in raw or ():
            try:
                FieldOutput.from_dict(field)
            except ValueError as exc:
                transport_errors.append({"field": field["field"], "error": str(exc)})
        encounters.append({"case_id": case.case_id, "stratum": case.stratum, "generation": record,
                           "raw_fields": raw, "raw_parse_error": error, "raw_contract_transport_errors": transport_errors,
                           "draft": draft, "total_seconds": time.perf_counter() - before})
        save(output, "encounters.json", encounters)
        print(case.case_id, draft["status"], draft["disposition"], record.get("generated_tokens"), flush=True)
    return encounters


def source_snapshot(output):
    paths = sorted({p for directory in ("nano", "nano_ai", "nanoscribe", "fabric") for p in (ROOT / directory).rglob("*.py")
                    if "__pycache__" not in p.parts} | {Path(__file__).resolve()})
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        hashes[relative] = digest(path)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--decoding", choices=("free", "constrained"), default="free")
    parser.add_argument("--policy", choices=("context", "grounded"), default="context")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight and (args.checkpoint is None or args.output is None):
        parser.error("a model run requires --checkpoint and a new --output directory")
    rows, cases, controls = preflight(args.data)
    if args.output is None:
        print(json.dumps(controls, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"schema_version": "p1.independent-development.v1", "development_only": True,
                  "frozen_evaluation": False, "incremental_paid_compute_usd": 0,
                  "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                  "platform": platform.platform(), "input_hashes": controls["input_hashes"],
                  "code_hashes": {}, "decoding": args.decoding, "binder": args.policy,
                  "prefix_cache": bool(args.prefix_cache),
                  "annotation_blinding": "See input_snapshot annotation protocol and independent annotation files.",
                  "status": "initializing"}
    save(args.output, "provenance.json", provenance)
    started = time.perf_counter()
    try:
        save(args.output, "controls.json", controls)
        for name in controls["input_hashes"]:
            target = args.output / "input_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((args.data / name).read_bytes())
            if digest(target) != controls["input_hashes"][name]:
                raise ValueError(f"annotation input changed while snapshotting: {name}")
        code_hashes = source_snapshot(args.output)
        provenance.update(code_hashes=code_hashes)
        if args.preflight:
            verify_lock(args.data, controls["input_hashes"])
            provenance.update(status="preflight_complete")
            print(json.dumps({"cases": len(cases), "contract_transport_errors": controls["contract_transport_errors"]}))
            return 0
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.decoding == "constrained" and args.policy == "grounded":
            from nanoscribe.grounded import GROUNDED_SYSTEM_PROMPT
            prompt = GROUNDED_SYSTEM_PROMPT
            prompt_style = "grounded-english"
            pred_kwargs = {"candidate_policy": "grounded", "prefix_cache": bool(args.prefix_cache)}
        else:
            prompt = STRUCTURED_PROMPT if args.decoding == "constrained" else FEWSHOT_PROMPT
            prompt_style = "structured" if args.decoding == "constrained" else "fewshot"
            pred_kwargs = {"prefix_cache": bool(args.prefix_cache)} if args.prefix_cache else {}
        predictor = LocalCompactPredictor(str(args.checkpoint), system_prompt=prompt, decoding=args.decoding, **pred_kwargs)
        checkpoint = Path(predictor.checkpoint)
        provenance.update(checkpoint=str(checkpoint), checkpoint_revision=checkpoint.name,
                          checkpoint_hashes={p.name: digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          prompt_style=prompt_style,
                          max_new_tokens=predictor.max_new_tokens, do_sample=False, local_files_only=True,
                          status="running")
        save(args.output, "prompt.json", {"system": prompt})
        save(args.output, "provenance.json", provenance)
        verify_lock(args.data, controls["input_hashes"])
        encounters = collect(cases, predictor, args.output, version=checkpoint.name, policy=args.policy)
        verify_lock(args.data, controls["input_hashes"])
        if any(digest(ROOT / path) != sha for path, sha in code_hashes.items()):
            raise ValueError("evaluation source code changed during model run")
        comparison, details = compare(cases, encounters)
        save(args.output, "comparison.json", comparison)
        save(args.output, "paired_fields.json", details)
        import torch
        import transformers
        provenance.update(status="complete", torch=torch.__version__, transformers=transformers.__version__,
                          device=predictor._loaded[2] if predictor._loaded else None,
                          input_lock_verified_after_inference=True)
    except Exception as exc:
        provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        provenance.update(total_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save(args.output, "provenance.json", provenance)
    print(json.dumps({path: comparison["metrics"][path]["grounded_exact_field_accuracy"] for path in ("raw", "checked")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
