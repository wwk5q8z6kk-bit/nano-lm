#!/usr/bin/env python3
"""Frozen, CPU-only generated relational evaluation instrument.

No language model is loaded. Symbolic controls test this instrument, not model
quality. A real checkpoint requires a separately implemented CandidateAdapter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Protocol
import unittest

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "generative_reasoning_protocol_v6.json"
FROZEN_PROTOCOL_SHA256 = "1d86c0b00d4c6db8974fd2bb3f739ba54809daedaa8a839109521c2184fd49b6"
FAMILIES = ("direct_retrieval", "two_hop")
CONDITIONS = ("full_a", "full_b", "permuted_a", "no_context", "broken_chain")
SPLITS = ("development", "evaluation")
VALUE_RE = re.compile(r"value_[0-9a-f]{12}\Z", re.ASCII)
ENTITY_RE = re.compile(r"entity_[0-9a-f]{12}\Z", re.ASCII)
HASH_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
ID_RE = re.compile(r"[0-9a-f]{24}\Z", re.ASCII)
PUBLIC_KEYS = {"id", "prompt"}
ORACLE_KEYS = {
    "id", "world_id", "split", "family", "condition", "query_entity", "facts",
    "expected_answer", "a_target", "prompt_sha256", "prompt_utf8_bytes",
    "context_utf8_bytes",
}
MODEL_KEYS = {
    "checkpoint_sha256", "tokenizer_sha256", "model_config_sha256",
    "adapter_source_sha256", "context_limit",
}
RUN_KEYS = {
    "schema_version", "run_id", "evidence_kind", "protocol_sha256",
    "dataset_manifest_sha256", "model", "decoding", "contamination_status",
}
CONTROL_RECEIPT_KEYS = {
    "kind", "prompt_sha256", "actual_model_tokenizer", "stop_reason",
    "truncated",
}
MODEL_RECEIPT_KEYS = {
    "kind", "prompt_sha256", "model", "actual_model_tokenizer",
    "input_token_ids", "output_token_ids", "input_tokens", "output_tokens",
    "context_limit", "max_new_tokens", "truncated", "reset_state_per_case",
    "stop_reason", "error", "generation_attempted",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha(value: str) -> str:
    return sha(value.encode("utf-8"))


def canonical(value) -> bytes:
    finite_json(value)
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def finite_json(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        require(math.isfinite(value), "nonfinite number")
    elif type(value) is list or type(value) is tuple:
        for item in value:
            finite_json(item)
    elif type(value) is dict:
        require(all(type(k) is str for k in value), "non-string JSON key")
        for item in value.values():
            finite_json(item)
    else:
        raise ValueError("not a JSON value")


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def strict_loads(raw):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    result = json.loads(raw, object_pairs_hook=strict_object,
                        parse_constant=reject_constant)
    finite_json(result)
    return result


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            label + " has invalid keys")


def positive_int(value, label):
    require(type(value) is int and value > 0, label + " must be a positive integer")


def hex_hash(value, label):
    require(type(value) is str and HASH_RE.fullmatch(value) is not None,
            label + " is not a SHA256")


def token_ids(value, label, allow_empty=False):
    require(type(value) is list, label + " must be an actual token-ID list")
    require(allow_empty or bool(value), label + " must not be empty")
    require(all(type(x) is int and x >= 0 for x in value), label + " has invalid IDs")


def protocol():
    raw = PROTOCOL_PATH.read_bytes()
    require(sha(raw) == FROZEN_PROTOCOL_SHA256, "frozen protocol SHA256 mismatch")
    return strict_loads(raw)


def source_sha():
    return sha(Path(__file__).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_new_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value))


def jsonl(rows):
    return b"".join(canonical(row) for row in rows)


def read_jsonl(path: Path):
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        require(bool(line.strip()), f"blank JSONL record at line {number}")
        row = strict_loads(line)
        require(type(row) is dict, f"non-object JSONL record at line {number}")
        rows.append(row)
    return rows


def fact_tuple(fact):
    return fact["subject"], fact["relation"], fact["object"]


def validate_facts(facts):
    require(type(facts) is list, "facts must be a list")
    keys = set()
    for fact in facts:
        exact_keys(fact, {"subject", "relation", "object"}, "fact")
        require(type(fact["subject"]) is str and
                ENTITY_RE.fullmatch(fact["subject"]) is not None, "invalid fact subject")
        relation = fact["relation"]
        require(relation in ("value", "points_to"), "invalid relation")
        matcher = VALUE_RE if relation == "value" else ENTITY_RE
        require(type(fact["object"]) is str and
                matcher.fullmatch(fact["object"]) is not None, "invalid fact object")
        key = fact["subject"], relation
        require(key not in keys, "duplicate or conflicting fact key")
        keys.add(key)


def semantic_answer(facts, family, entity):
    """Structured-fact oracle; this function never supplies a model input."""
    validate_facts(facts)
    require(family in FAMILIES, "unknown family")
    require(type(entity) is str and ENTITY_RE.fullmatch(entity) is not None,
            "invalid query entity")
    assignments = {(f["subject"], f["relation"]): f["object"] for f in facts}
    if family == "two_hop":
        entity = assignments.get((entity, "points_to"))
        if entity is None:
            return "UNKNOWN"
    return assignments.get((entity, "value"), "UNKNOWN")


def fact_line(fact):
    verb = "has value" if fact["relation"] == "value" else "points to"
    return f'{fact["subject"]} {verb} {fact["object"]}.'


def context_text(facts):
    return "\n".join(fact_line(f) for f in facts) if facts else "(none)"


def render_prompt(facts, family, entity):
    p = protocol()["prompt"]
    query = p["direct_query" if family == "direct_retrieval" else "two_hop_query"]
    query = query.format(entity=entity)
    return (
        p["preamble"] + "\n\nFacts:\n" + context_text(facts) +
        "\n\nQuery: " + query + "\nAnswer:"
    )


def parse_visible_prompt(prompt):
    """Parse the frozen grammar; returned facts are reconstructed from prompt bytes."""
    require(type(prompt) is str, "prompt is not text")
    prefix = protocol()["prompt"]["preamble"] + "\n\nFacts:\n"
    require(prompt.startswith(prefix) and prompt.endswith("\nAnswer:"),
            "prompt does not use frozen framing")
    body = prompt[len(prefix):-len("\nAnswer:")]
    require(body.count("\n\nQuery: ") == 1, "invalid query boundary")
    fact_text, query = body.split("\n\nQuery: ")
    direct = re.fullmatch(r"What is the value of (entity_[0-9a-f]{12})\?", query)
    hop = re.fullmatch(r"What is the value of the entity reached from (entity_[0-9a-f]{12})\?", query)
    require(bool(direct or hop), "invalid query")
    family, entity = ("direct_retrieval", direct.group(1)) if direct else ("two_hop", hop.group(1))
    facts = []
    if fact_text != "(none)":
        for line in fact_text.split("\n"):
            match = re.fullmatch(
                r"(entity_[0-9a-f]{12}) (has value|points to) ((?:entity|value)_[0-9a-f]{12})\.",
                line,
            )
            require(match is not None, "invalid fact line")
            facts.append({"subject": match.group(1),
                          "relation": "value" if match.group(2) == "has value" else "points_to",
                          "object": match.group(3)})
    validate_facts(facts)
    return facts, family, entity


@dataclass(frozen=True, slots=True)
class CandidateView:
    """The only per-case object a candidate may receive."""
    prompt: str


def symbolic_candidate(view: CandidateView):
    """Independent prompt parser/control. Does not receive private oracle data."""
    facts, family, entity = parse_visible_prompt(view.prompt)
    values, destinations = {}, {}
    for fact in facts:
        if fact["relation"] == "value":
            values[fact["subject"]] = fact["object"]
        else:
            destinations[fact["subject"]] = fact["object"]
    if family == "two_hop":
        if entity not in destinations:
            return "UNKNOWN"
        entity = destinations[entity]
    return values[entity] if entity in values else "UNKNOWN"


def value_control(view: CandidateView, last=False):
    facts, _, _ = parse_visible_prompt(view.prompt)
    values = [f["object"] for f in facts if f["relation"] == "value"]
    return values[-1 if last else 0] if values else "UNKNOWN"


def make_world(split, family, index):
    p = protocol()
    seed = p["splits"][split]["seed"]
    world_id = text_sha(f'{p["experiment_id"]}|{split}|{seed}|{family}|{index}')
    def name(kind, slot):
        return kind + "_" + text_sha(world_id + "|" + kind + "|" + str(slot))[:12]
    origin = name("entity", 0)
    destination = name("entity", 1) if family == "two_hop" else origin
    a_target, b_target = name("value", 0), name("value", 1)
    facts = [{"subject": destination, "relation": "value", "object": a_target}]
    distractors = [name("entity", 2 + j) for j in range(p["world"]["distractor_entities"])]
    for j, entity in enumerate(distractors):
        facts.append({"subject": entity, "relation": "value", "object": name("value", 2 + j)})
    if family == "two_hop":
        facts.append({"subject": origin, "relation": "points_to", "object": destination})
        for j in range(3):
            facts.append({"subject": distractors[j], "relation": "points_to",
                          "object": distractors[j + 1]})
    facts.sort(key=lambda f: text_sha(world_id + "|order|" + "|".join(fact_tuple(f))))
    full_b = [
        {**f, "object": b_target}
        if (f["subject"], f["relation"]) == (destination, "value") else dict(f)
        for f in facts
    ]
    removed = (origin, "points_to") if family == "two_hop" else (origin, "value")
    variants = {
        "full_a": facts,
        "full_b": full_b,
        "permuted_a": facts[1:] + facts[:1],
        "no_context": [],
        "broken_chain": [f for f in facts if (f["subject"], f["relation"]) != removed],
    }
    public, oracle = [], []
    for condition in CONDITIONS:
        visible = variants[condition]
        prompt = render_prompt(visible, family, origin)
        identifier = text_sha(world_id + "|case|" + condition)[:24]
        public.append({"id": identifier, "prompt": prompt})
        oracle.append({
            "id": identifier, "world_id": world_id, "split": split, "family": family,
            "condition": condition, "query_entity": origin,
            "facts": [dict(f) for f in visible],
            "expected_answer": semantic_answer(visible, family, origin),
            "a_target": a_target, "prompt_sha256": text_sha(prompt),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "context_utf8_bytes": len(context_text(visible).encode("utf-8")),
        })
    return public, oracle


def validate_dataset(public, oracle):
    require(type(public) is list and type(oracle) is list, "dataset rows must be lists")
    require(public and oracle, "dataset must not be empty")
    by_id = {}
    for row in public:
        exact_keys(row, PUBLIC_KEYS, "public record")
        require(type(row["id"]) is str and ID_RE.fullmatch(row["id"]), "invalid public ID")
        require(row["id"] not in by_id, "duplicate public ID")
        require(type(row["prompt"]) is str and row["prompt"], "empty public prompt")
        by_id[row["id"]] = row
    oracle_by_id, groups = {}, defaultdict(dict)
    split_prompts, split_worlds = defaultdict(set), defaultdict(set)
    split_facts, split_vocab = defaultdict(set), defaultdict(set)
    for row in oracle:
        exact_keys(row, ORACLE_KEYS, "oracle record")
        identifier = row["id"]
        require(identifier in by_id and identifier not in oracle_by_id, "unknown or duplicate oracle ID")
        oracle_by_id[identifier] = row
        require(row["split"] in SPLITS and row["family"] in FAMILIES and
                row["condition"] in CONDITIONS, "invalid oracle strata")
        hex_hash(row["world_id"], "world ID")
        prompt = by_id[identifier]["prompt"]
        parsed_facts, parsed_family, parsed_entity = parse_visible_prompt(prompt)
        require(parsed_facts == row["facts"] and parsed_family == row["family"] and
                parsed_entity == row["query_entity"], "public/oracle visible semantics mismatch")
        require(row["prompt_sha256"] == text_sha(prompt), "prompt hash mismatch")
        require(type(row["prompt_utf8_bytes"]) is int and
                row["prompt_utf8_bytes"] == len(prompt.encode("utf-8")), "prompt byte count mismatch")
        require(type(row["context_utf8_bytes"]) is int and
                row["context_utf8_bytes"] == len(context_text(parsed_facts).encode("utf-8")),
                "context byte count mismatch")
        require(row["expected_answer"] == semantic_answer(parsed_facts, parsed_family, parsed_entity),
                "oracle gold does not follow visible facts")
        require(type(row["a_target"]) is str and VALUE_RE.fullmatch(row["a_target"]), "invalid A target")
        group = groups[row["world_id"]]
        require(row["condition"] not in group, "duplicate condition in world")
        group[row["condition"]] = row
        split = row["split"]
        split_prompts[split].add(row["prompt_sha256"])
        split_worlds[split].add(row["world_id"])
        split_vocab[split].add(row["query_entity"])
        for fact in parsed_facts:
            split_vocab[split].update((fact["subject"], fact["object"]))
        if row["condition"] in ("full_a", "full_b"):
            split_facts[split].add(sha(canonical(sorted(fact_tuple(f) for f in parsed_facts))))
    require(set(by_id) == set(oracle_by_id), "public/oracle ID coverage mismatch")
    for world, group in groups.items():
        require(set(group) == set(CONDITIONS), "incomplete world conditions")
        a, b, perm, nc, broken = (group[c] for c in CONDITIONS)
        for row in group.values():
            require((row["split"], row["family"], row["query_entity"]) ==
                    (a["split"], a["family"], a["query_entity"]), "world metadata mismatch")
            require(row["a_target"] == a["expected_answer"], "A-target mismatch")
        require(a["expected_answer"] != b["expected_answer"] and
                VALUE_RE.fullmatch(a["expected_answer"]) and VALUE_RE.fullmatch(b["expected_answer"]),
                "counterfactual targets must differ")
        af, bf = a["facts"], b["facts"]
        require(len(af) == len(bf), "counterfactual fact count changed")
        changed = [(x, y) for x, y in zip(af, bf) if x != y]
        require(len(changed) == 1, "counterfactual must change exactly one fact")
        left, right = changed[0]
        require(left["relation"] == right["relation"] == "value" and
                left["subject"] == right["subject"], "counterfactual changed relation/key")
        require(sorted(map(fact_tuple, af)) == sorted(map(fact_tuple, perm["facts"])) and
                af != perm["facts"], "permutation did not preserve facts/change order")
        require(perm["expected_answer"] == a["expected_answer"], "permutation changed gold")
        require(nc["facts"] == [] and nc["expected_answer"] == "UNKNOWN", "invalid no-context case")
        removed = (a["query_entity"], "points_to" if a["family"] == "two_hop" else "value")
        expected_broken = [f for f in af if (f["subject"], f["relation"]) != removed]
        require(len(expected_broken) == len(af) - 1 and broken["facts"] == expected_broken and
                broken["expected_answer"] == "UNKNOWN", "invalid broken-chain case")
    for name, sets in (("world", split_worlds), ("prompt", split_prompts),
                       ("fact set", split_facts), ("vocabulary", split_vocab)):
        require(not sets["development"].intersection(sets["evaluation"]),
                "cross-split " + name + " overlap")
    p = protocol()
    counts = {}
    for split in SPLITS:
        rows = [r for r in oracle if r["split"] == split]
        worlds = {r["world_id"] for r in rows}
        for family in FAMILIES:
            count = len({r["world_id"] for r in rows if r["family"] == family})
            require(count == p["splits"][split]["worlds_per_family"],
                    "world count does not match frozen recipe")
        counts[split] = {
            "worlds": len(worlds), "records": len(rows),
            "planned_prompt_utf8_bytes": sum(r["prompt_utf8_bytes"] for r in rows),
            "planned_context_utf8_bytes": sum(r["context_utf8_bytes"] for r in rows),
        }
    return {
        "worlds": len(groups), "records": len(public), "splits": counts,
        "cross_split_overlap": {"world_ids": 0, "prompts": 0, "full_fact_sets": 0,
                                "entity_and_value_vocabulary": 0},
        "templates": "shared; template transfer not evaluated",
    }


def build_dataset():
    public, oracle = [], []
    for split in SPLITS:
        for family in FAMILIES:
            for index in range(protocol()["splits"][split]["worlds_per_family"]):
                new_public, new_oracle = make_world(split, family, index)
                public.extend(new_public)
                oracle.extend(new_oracle)
    # Opaque IDs scramble presentation; conditions are not exposed in the public file.
    public.sort(key=lambda row: row["id"])
    oracle.sort(key=lambda row: row["id"])
    validate_dataset(public, oracle)
    return public, oracle


@dataclass
class Dataset:
    public: list
    oracle: list
    manifest: dict
    manifest_sha256: str


def generate_dataset(destination: Path):
    require(not destination.exists(), "destination exists; immutable dataset will not be overwritten")
    public, oracle = build_dataset()
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, rows in (("public.jsonl", public), ("oracle.jsonl", oracle)):
        raw = jsonl(rows)
        with (destination / filename).open("xb") as handle:
            handle.write(raw)
        files[filename] = {"sha256": sha(raw), "bytes": len(raw), "records": len(rows)}
    manifest = {
        "schema_version": 1,
        "experiment_id": protocol()["experiment_id"],
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol_frozen_at_utc": protocol()["local_recipe_frozen_at_utc"],
        "generator_source_sha256": source_sha(),
        "generated_at_utc": now(),
        "runtime": {"python": platform.python_version()},
        "files": files,
        "summary": validate_dataset(public, oracle),
        "contamination_status": "NOT_CHECKED",
    }
    write_new_json(destination / "manifest.json", manifest)
    return load_dataset(destination)


def load_dataset(destination: Path):
    raw_manifest = (destination / "manifest.json").read_bytes()
    manifest = strict_loads(raw_manifest)
    exact_keys(manifest, {
        "schema_version", "experiment_id", "protocol_sha256", "protocol_frozen_at_utc",
        "generator_source_sha256", "generated_at_utc", "runtime", "files",
        "summary", "contamination_status",
    }, "dataset manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1 and
            manifest["experiment_id"] == protocol()["experiment_id"],
            "wrong dataset schema/experiment")
    require(manifest["protocol_sha256"] == FROZEN_PROTOCOL_SHA256 and
            manifest["protocol_frozen_at_utc"] == protocol()["local_recipe_frozen_at_utc"],
            "protocol receipt mismatch")
    require(manifest["generator_source_sha256"] == source_sha(), "generator code hash mismatch")
    require(manifest["contamination_status"] == "NOT_CHECKED", "unsupported contamination claim")
    exact_keys(manifest["files"], {"public.jsonl", "oracle.jsonl"}, "dataset files")
    records = {}
    for filename in ("public.jsonl", "oracle.jsonl"):
        metadata = manifest["files"][filename]
        exact_keys(metadata, {"sha256", "bytes", "records"}, "dataset file receipt")
        positive_int(metadata["bytes"], "dataset bytes")
        positive_int(metadata["records"], "dataset records")
        raw = (destination / filename).read_bytes()
        require(metadata["sha256"] == sha(raw) and metadata["bytes"] == len(raw),
                "dataset file hash/byte mismatch: " + filename)
        records[filename] = read_jsonl(destination / filename)
        require(metadata["records"] == len(records[filename]), "dataset record count mismatch")
    public, oracle = records["public.jsonl"], records["oracle.jsonl"]
    require(manifest["summary"] == validate_dataset(public, oracle), "dataset summary mismatch")
    return Dataset(public, oracle, manifest, sha(raw_manifest))


def decoding_policy():
    return {
        "do_sample": False, "max_new_tokens": protocol()["generation"]["max_new_tokens"],
        "stop_policy": "first_newline_or_eos", "reset_state_per_case": True,
    }


def make_run_manifest(dataset: Dataset, run_id, evidence_kind="instrument_control", model=None):
    manifest = {
        "schema_version": 1, "run_id": run_id, "evidence_kind": evidence_kind,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "model": model, "decoding": decoding_policy(),
        "contamination_status": "NOT_CHECKED",
    }
    validate_run_manifest(dataset, manifest)
    return manifest


def validate_run_manifest(dataset, manifest):
    finite_json(manifest)
    exact_keys(manifest, RUN_KEYS, "run manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "wrong run schema")
    require(type(manifest["run_id"]) is str and bool(manifest["run_id"]), "empty run ID")
    require(manifest["protocol_sha256"] == FROZEN_PROTOCOL_SHA256 and
            manifest["dataset_manifest_sha256"] == dataset.manifest_sha256, "run/dataset binding mismatch")
    exact_keys(manifest["decoding"], set(decoding_policy()), "decoding policy")
    require(manifest["decoding"] == decoding_policy() and
            manifest["decoding"]["do_sample"] is False and
            manifest["decoding"]["reset_state_per_case"] is True and
            type(manifest["decoding"]["max_new_tokens"]) is int,
            "unregistered decoding policy")
    require(manifest["contamination_status"] == "NOT_CHECKED", "unsupported contamination claim")
    kind = manifest["evidence_kind"]
    require(kind in ("instrument_control", "model_checkpoint_evaluation"), "invalid evidence kind")
    if kind == "instrument_control":
        require(manifest["model"] is None, "instrument control must not assert a learned model identity")
    else:
        exact_keys(manifest["model"], MODEL_KEYS, "model identity")
        for key in MODEL_KEYS - {"context_limit"}:
            hex_hash(manifest["model"][key], key)
        positive_int(manifest["model"]["context_limit"], "context limit")


def control_predictions(dataset, candidate):
    predictions = []
    for row in dataset.public:
        text = candidate(CandidateView(prompt=row["prompt"]))
        require(type(text) is str, "candidate output is not a string")
        predictions.append({
            "id": row["id"], "status": "generated", "raw_text": text,
            "receipt": {
                "kind": "instrument_control", "prompt_sha256": text_sha(row["prompt"]),
                "actual_model_tokenizer": False, "stop_reason": "symbolic", "truncated": False,
            },
        })
    return predictions


@dataclass(frozen=True)
class Completion:
    raw_text: str
    input_token_ids: list[int]
    output_token_ids: list[int]
    stop_reason: str
    truncated: bool = False


class CandidateAdapter(Protocol):
    """Implement externally for a real native checkpoint; no binding is supplied.

    identity has MODEL_KEYS. reset clears all case-dependent state. encode_prompt
    and generate must use the identical actual model tokenizer/BOS sequence.
    generate follows decoding_policy(), including first-newline/EOS stopping.
    """
    identity: dict
    def reset(self) -> None: ...
    def encode_prompt(self, candidate: CandidateView) -> list[int]: ...
    def generate(self, candidate: CandidateView) -> Completion: ...


def model_receipt(prompt, manifest, ids, output=None, stop_reason=None, error=None,
                  generation_attempted=False):
    output = [] if output is None else output
    return {
        "kind": "model_checkpoint_evaluation", "prompt_sha256": text_sha(prompt),
        "model": dict(manifest["model"]), "actual_model_tokenizer": ids is not None,
        "input_token_ids": ids, "output_token_ids": output,
        "input_tokens": len(ids) if ids is not None else None, "output_tokens": len(output),
        "context_limit": manifest["model"]["context_limit"],
        "max_new_tokens": manifest["decoding"]["max_new_tokens"], "truncated": False,
        "reset_state_per_case": True, "stop_reason": stop_reason, "error": error,
        "generation_attempted": generation_attempted or stop_reason is not None,
    }


def run_candidate(dataset: Dataset, adapter: CandidateAdapter, run_manifest):
    """Generic prompt-only runner. The caller supplies the real checkpoint adapter.

    Nothing here reads private oracle data or passes a case ID to the adapter.
    Failures and context-ineligible records are retained for scoring.
    """
    validate_run_manifest(dataset, run_manifest)
    require(run_manifest["evidence_kind"] == "model_checkpoint_evaluation",
            "adapter runner requires a model identity receipt")
    require(adapter.identity == run_manifest["model"], "adapter/run identity mismatch")
    predictions = []
    for row in dataset.public:
        view = CandidateView(prompt=row["prompt"])
        ids = None
        try:
            adapter.reset()
            ids = adapter.encode_prompt(view)
            token_ids(ids, "adapter input")
        except Exception as exc:
            predictions.append({
                "id": row["id"], "status": "error", "raw_text": "",
                "receipt": model_receipt(view.prompt, run_manifest, None,
                                         error="encoding/reset: " + type(exc).__name__),
            })
            continue
        receipt = model_receipt(view.prompt, run_manifest, ids)
        if len(ids) + receipt["max_new_tokens"] > receipt["context_limit"]:
            predictions.append({"id": row["id"], "status": "context_ineligible",
                                "raw_text": "", "receipt": receipt})
            continue
        try:
            completion = adapter.generate(view)
            require(type(completion) is Completion, "adapter returned invalid completion")
            require(completion.input_token_ids == ids, "adapter changed encoded prompt")
            require(completion.truncated is False, "adapter silently truncated prompt")
            receipt = model_receipt(view.prompt, run_manifest, ids,
                                    completion.output_token_ids, completion.stop_reason)
            prediction = {"id": row["id"], "status": "generated",
                          "raw_text": completion.raw_text, "receipt": receipt}
            validate_prediction(prediction, row, run_manifest)
            predictions.append(prediction)
        except Exception as exc:
            predictions.append({
                "id": row["id"], "status": "error", "raw_text": "",
                "receipt": model_receipt(view.prompt, run_manifest, ids,
                                         error="generation: " + type(exc).__name__,
                                         generation_attempted=True),
            })
    return predictions


def validate_prediction(prediction, public, manifest):
    finite_json(prediction)
    exact_keys(prediction, {"id", "status", "raw_text", "receipt"}, "prediction")
    require(prediction["id"] == public["id"], "prediction ID mismatch")
    require(type(prediction["raw_text"]) is str, "raw_text must be a string")
    status = prediction["status"]
    require(status in ("generated", "context_ineligible", "error"), "invalid prediction status")
    receipt = prediction["receipt"]
    kind = manifest["evidence_kind"]
    if kind == "instrument_control":
        exact_keys(receipt, CONTROL_RECEIPT_KEYS, "control receipt")
        require(status == "generated" and receipt["kind"] == kind and
                receipt["actual_model_tokenizer"] is False and
                receipt["stop_reason"] == "symbolic" and receipt["truncated"] is False,
                "invalid instrument control receipt")
    else:
        exact_keys(receipt, MODEL_RECEIPT_KEYS, "model receipt")
        require(receipt["kind"] == kind and receipt["model"] == manifest["model"],
                "model identity mismatch")
        require(receipt["truncated"] is False and receipt["reset_state_per_case"] is True,
                "truncation/state reset contract violated")
        require(type(receipt["generation_attempted"]) is bool, "invalid generation-attempt receipt")
        positive_int(receipt["context_limit"], "receipt context limit")
        positive_int(receipt["max_new_tokens"], "receipt output allowance")
        require(receipt["context_limit"] == manifest["model"]["context_limit"] and
                receipt["max_new_tokens"] == manifest["decoding"]["max_new_tokens"],
                "context/decoding mismatch")
        ids = receipt["input_token_ids"]
        if ids is None:
            require(status == "error" and receipt["input_tokens"] is None and
                    receipt["actual_model_tokenizer"] is False and
                    receipt["generation_attempted"] is False, "unknown actual tokenizer counts")
        else:
            token_ids(ids, "input")
            require(type(receipt["input_tokens"]) is int and receipt["input_tokens"] == len(ids) and
                    receipt["actual_model_tokenizer"] is True, "actual input count mismatch")
        token_ids(receipt["output_token_ids"], "output", allow_empty=True)
        require(type(receipt["output_tokens"]) is int and
                receipt["output_tokens"] == len(receipt["output_token_ids"]), "output count mismatch")
        if status == "generated":
            require(receipt["generation_attempted"] is True, "generated without attempt receipt")
            require(ids is not None and len(ids) + receipt["max_new_tokens"] <= receipt["context_limit"],
                    "generated with ineligible/unknown context")
            require(0 < receipt["output_tokens"] <= receipt["max_new_tokens"], "invalid generation length")
            require(receipt["stop_reason"] in ("eos", "newline", "max_new_tokens") and
                    receipt["error"] is None, "invalid generation stop/error")
            if receipt["stop_reason"] == "max_new_tokens":
                require(receipt["output_tokens"] == receipt["max_new_tokens"], "invalid budget stop")
            if receipt["stop_reason"] == "newline":
                require(prediction["raw_text"].endswith("\n") and
                        prediction["raw_text"].count("\n") == 1, "first-newline stop violated")
        elif status == "context_ineligible":
            require(receipt["generation_attempted"] is False, "ineligible prompt reached generator")
            require(ids is not None and len(ids) + receipt["max_new_tokens"] > receipt["context_limit"],
                    "false context-ineligible claim")
            require(receipt["error"] is None and receipt["stop_reason"] is None and
                    receipt["output_token_ids"] == [] and prediction["raw_text"] == "",
                    "ineligible context contains generation")
        else:
            if receipt["generation_attempted"]:
                require(ids is not None and len(ids) + receipt["max_new_tokens"] <= receipt["context_limit"],
                        "failed generation attempted with ineligible/unknown context")
            require(type(receipt["error"]) is str and receipt["error"] and
                    receipt["stop_reason"] is None and receipt["output_token_ids"] == [] and
                    prediction["raw_text"] == "", "invalid error receipt")
    require(receipt["prompt_sha256"] == text_sha(public["prompt"]), "prediction prompt hash mismatch")


def normalized_answer(text):
    answer = text.strip(" \t\r\n")
    return answer if answer == "UNKNOWN" or VALUE_RE.fullmatch(answer) else None


def predicted_answer(prediction):
    if prediction is None or prediction["status"] != "generated":
        return None
    if prediction["receipt"]["stop_reason"] == "max_new_tokens":
        return None
    return normalized_answer(prediction["raw_text"])


def rate(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int and
            0 <= numerator <= denominator and denominator > 0, "invalid metric counts")
    return {"correct": numerator, "denominator": denominator, "rate": numerator / denominator}


def score(dataset: Dataset, predictions, run_manifest):
    validate_dataset(dataset.public, dataset.oracle)
    validate_run_manifest(dataset, run_manifest)
    require(type(predictions) is list, "predictions must be a list")
    public = {r["id"]: r for r in dataset.public}
    by_id = {}
    for prediction in predictions:
        require(type(prediction) is dict, "non-object prediction")
        identifier = prediction.get("id")
        require(type(identifier) is str and identifier in public, "unknown prediction ID")
        require(identifier not in by_id, "duplicate prediction ID")
        validate_prediction(prediction, public[identifier], run_manifest)
        by_id[identifier] = prediction
    answer = {identifier: predicted_answer(by_id.get(identifier)) for identifier in public}
    groups = defaultdict(dict)
    for row in dataset.oracle:
        groups[row["world_id"]][row["condition"]] = row
    metrics = {}
    for split in SPLITS:
        metrics[split] = {}
        for family in ("all", *FAMILIES):
            worlds = [g for g in groups.values()
                      if g["full_a"]["split"] == split and
                      (family == "all" or g["full_a"]["family"] == family)]
            n = len(worlds)
            def correct(row):
                return answer[row["id"]] == row["expected_answer"]
            joint = sum(correct(g["full_a"]) and correct(g["full_b"]) for g in worlds)
            permutation_joint = sum(correct(g["full_a"]) and correct(g["permuted_a"]) for g in worlds)
            permutation_same = sum(
                answer[g["full_a"]["id"]] is not None and
                answer[g["full_a"]["id"]] == answer[g["permuted_a"]["id"]]
                for g in worlds
            )
            full_correct = sum(correct(g["full_a"]) for g in worlds)
            nc_original = sum(answer[g["no_context"]["id"]] == g["full_a"]["expected_answer"] for g in worlds)
            conditions = {
                condition: rate(sum(correct(g[condition]) for g in worlds), n)
                for condition in CONDITIONS
            }
            metrics[split][family] = {
                "worlds": n, "records": n * len(CONDITIONS),
                "primary_counterfactual_joint": rate(joint, n),
                "permutation_joint_correct": rate(permutation_joint, n),
                "permutation_same_valid_answer": rate(permutation_same, n),
                "original_target_contextual_gain": {
                    "full_a_correct": full_correct, "no_context_original_target_correct": nc_original,
                    "denominator_worlds": n, "paired_difference": (full_correct - nc_original) / n,
                },
                "condition_policy_accuracy": conditions,
                "unknown_abstention": rate(
                    sum(correct(g[c]) for g in worlds for c in ("no_context", "broken_chain")), 2 * n),
            }
    statuses = Counter(by_id[i]["status"] if i in by_id else "missing" for i in public)
    for key in ("generated", "context_ineligible", "error", "missing"):
        statuses.setdefault(key, 0)
    censored = sum(p["receipt"]["stop_reason"] == "max_new_tokens" for p in predictions)
    invalid_format = sum(p["status"] == "generated" and normalized_answer(p["raw_text"]) is None
                         for p in predictions)
    generated_ids = {i for i, p in by_id.items() if p["status"] == "generated"}
    model_kind = run_manifest["evidence_kind"] == "model_checkpoint_evaluation"
    attempted_ids = ({i for i, p in by_id.items() if p["receipt"]["generation_attempted"]}
                     if model_kind else generated_ids)
    oracle_by_id = {r["id"]: r for r in dataset.oracle}
    report = {
        "schema_version": 1, "experiment_id": protocol()["experiment_id"],
        "run_manifest": run_manifest, "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "dataset_manifest_sha256": dataset.manifest_sha256, "scorer_source_sha256": source_sha(),
        "evidence_kind": run_manifest["evidence_kind"],
        "model_execution_status": "EXTERNAL_PREDICTIONS_SCORED" if model_kind else "NOT_RUN",
        "contamination_status": "NOT_CHECKED",
        "denominators": {"registered_records": len(public), "received_predictions": len(predictions),
                         "status_counts": dict(statuses), "budget_exhausted": censored,
                         "format_invalid": invalid_format},
        "exposure": {
            "kind": "declared_model_receipts" if model_kind else "symbolic_instrument_controls",
            "generated_prompt_utf8_bytes": sum(oracle_by_id[i]["prompt_utf8_bytes"] for i in generated_ids),
            "generated_context_utf8_bytes": sum(oracle_by_id[i]["context_utf8_bytes"] for i in generated_ids),
            "attempted_prompt_utf8_bytes": sum(oracle_by_id[i]["prompt_utf8_bytes"] for i in attempted_ids),
            "attempted_context_utf8_bytes": sum(oracle_by_id[i]["context_utf8_bytes"] for i in attempted_ids),
            "actual_model_input_tokens": sum(by_id[i]["receipt"]["input_tokens"] for i in attempted_ids)
                                         if model_kind else None,
            "observed_completed_model_output_tokens": sum(by_id[i]["receipt"]["output_tokens"] for i in generated_ids)
                                          if model_kind else None,
            "failed_generation_output_token_counts_unknown": len(attempted_ids - generated_ids) if model_kind else None,
            "model_tokenizer_context_eligibility": "RECEIPTS_VALIDATED" if model_kind else "NOT_CHECKED",
            "no_silent_truncation": True,
        },
        "metrics": metrics,
        "limitations": [
            "Instrument/control results are not learned-model results.",
            "Receipts are consistency checks, not independent execution attestation.",
            "Within-world conditions are dependent; no inferential efficacy claim.",
            "Same grammar templates across splits; no broad language-transfer claim.",
            "Training-corpus contamination and actual pretraining exposure remain NOT_CHECKED.",
        ],
    }
    finite_json(report)
    return report


def instrument_controls(dataset):
    controls = {
        "symbolic_visible_facts_oracle": symbolic_candidate,
        "always_unknown": lambda view: "UNKNOWN",
        "first_visible_value": lambda view: value_control(view),
        "last_visible_value": lambda view: value_control(view, last=True),
        "constant_value": lambda view: "value_000000000000",
    }
    reports = {}
    for name, candidate in controls.items():
        run = make_run_manifest(dataset, name)
        reports[name] = score(dataset, control_predictions(dataset, candidate), run)
    return {
        "schema_version": 1, "classification": "INSTRUMENT_CONTROLS_ONLY",
        "experiment_id": protocol()["experiment_id"], "created_at_utc": now(),
        "protocol_sha256": FROZEN_PROTOCOL_SHA256, "protocol_frozen_at_utc": protocol()["local_recipe_frozen_at_utc"],
        "generator_and_scorer_source_sha256": source_sha(),
        "test_source_sha256": sha((HERE / "test_generative_reasoning_v6.py").read_bytes()),
        "dataset_manifest": dataset.manifest, "dataset_manifest_sha256": dataset.manifest_sha256,
        "model_execution_status": "NOT_RUN", "contamination_status": "NOT_CHECKED",
        "controls": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--out", required=True, type=Path)
    for name in ("controls", "checks", "score"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
        if name == "score":
            command.add_argument("--predictions", required=True, type=Path)
            command.add_argument("--run-manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        dataset = generate_dataset(args.out)
        print(json.dumps({"dataset": str(args.out), "manifest_sha256": dataset.manifest_sha256,
                          "summary": dataset.manifest["summary"]}, indent=2))
        return
    dataset = load_dataset(args.dataset)
    if args.command == "score":
        predictions = read_jsonl(args.predictions)
        run = strict_loads(args.run_manifest.read_bytes())
        report = score(dataset, predictions, run)
    else:
        report = instrument_controls(dataset)
        if args.command == "checks":
            stream = io.StringIO()
            suite = unittest.defaultTestLoader.discover(str(HERE),
                                                       pattern="test_generative_reasoning_v6.py")
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            report["unit_validation"] = {
                "tests_run": result.testsRun, "failures": len(result.failures),
                "errors": len(result.errors), "skipped": len(result.skipped),
                "successful": result.wasSuccessful(), "output": stream.getvalue(),
                "scope": "Only test_generative_reasoning_v6.py; synthetic adapters are mechanism fixtures.",
            }
            if not result.wasSuccessful():
                print(stream.getvalue(), file=sys.stderr)
                raise SystemExit("instrument checks failed; no success report written")
    write_new_json(args.out, report)
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out.read_bytes()),
                      "evidence_kind": report.get("classification", report.get("evidence_kind")),
                      "model_execution_status": report["model_execution_status"]}, indent=2))


if __name__ == "__main__":
    main()

