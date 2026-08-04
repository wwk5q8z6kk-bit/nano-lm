"""Generate synthetic mini-corpus + gold for Wedge v1 Phase 2.

I*/X* frozen in inclusion_predicates.md. Seed fixed before scoring.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "corpus"
GOLD = ROOT / "data" / "gold"
MANIFESTS = ROOT / "data" / "manifests"
SEED = 20260731


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _span(doc: str, needle: str) -> dict | None:
    i = doc.find(needle)
    if i < 0:
        return None
    return {"start": i, "end": i + len(needle), "text": needle}


DOCS: list[dict] = [
    {
        "doc_id": "tech_note_cache",
        "title": "Cache Invalidation Notes",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "year": 2024,
        "doc_type": "note",
        "doi": None,
        "body": """# Cache Invalidation Notes

Authors: Ada Lovelace, Grace Hopper
Year: 2024
Contact: ada@example.org

## Summary

We define cache TTL as 300 seconds. The peak QPS is 12000.

## Definition

Latency is the end-to-end response time measured at p50.

## Results

Sample size n=240. Throughput improved by 18%.

## Notes

See https://example.org/cache-rfc and citation key CACHE2024.
""",
    },
    {
        "doc_id": "tech_note_cache_v2",
        "title": "Cache Invalidation Follow-up",
        "authors": ["Ada Lovelace"],
        "year": 2025,
        "doc_type": "note",
        "doi": None,
        "body": """# Cache Invalidation Follow-up

Authors: Ada Lovelace
Year: 2025

## Summary

We revise cache TTL as 600 seconds. Peak QPS remains 12000.

## Results

Sample size n=240. No change in error rate.
""",
    },
    {
        "doc_id": "bio_abs_metformin",
        "title": "Metformin Kinetics in Synthetic Cohort",
        "authors": ["Alan Turing", "Joan Clarke"],
        "year": 2023,
        "doc_type": "abstract",
        "doi": "10.0000/fake.metformin.2023",
        "body": """# Metformin Kinetics in Synthetic Cohort

Authors: Alan Turing, Joan Clarke
Year: 2023
DOI: 10.0000/fake.metformin.2023
Affiliation: Institute for Imaginary Medicine

## Abstract

Patients received metformin 500 mg twice daily. Allergy mentions include ragweed pollen
in the screening form (not a drug). Compound list: metformin, placebo.

Dose schedule: 500 mg BID. Sample size n=512.

## Captions

Figure 1: Plasma concentration over 24h.
Table 1: Baseline covariates.
""",
    },
    {
        "doc_id": "bio_abs_metformin_conflict",
        "title": "Metformin Kinetics Replication",
        "authors": ["Alan Turing"],
        "year": 2024,
        "doc_type": "abstract",
        "doi": "10.0000/fake.metformin.2024",
        "body": """# Metformin Kinetics Replication

Authors: Alan Turing
Year: 2024
DOI: 10.0000/fake.metformin.2024

## Abstract

Patients received metformin 850 mg twice daily. Sample size n=480.

Compound list: metformin.
""",
    },
    {
        "doc_id": "semi_structured_lab",
        "title": "Lab Instrument Log",
        "authors": ["Katherine Johnson"],
        "year": 2022,
        "doc_type": "note",
        "doi": None,
        "body": """# Lab Instrument Log

Authors: Katherine Johnson
Year: 2022

device: spectrometer-7
wavelength_nm: 532
operator: KJ
status: calibrated

ibuprofen appears here as a reagent label only (not an allergy field).
Missing: patient_id intentionally absent.
""",
    },
    {
        "doc_id": "tableish_throughput",
        "title": "Throughput Table Dump",
        "authors": ["Claude Shannon"],
        "year": 2021,
        "doc_type": "table_dump",
        "doi": None,
        "body": """# Throughput Table Dump

Authors: Claude Shannon
Year: 2021

region | qps | error_rate
us-east | 1000 | 0.01
eu-west | 800 | 0.02
ap-south | 600 | 0.015
""",
    },
    {
        "doc_id": "noisy_ocr_line",
        "title": "OCR Recovery Fixture",
        "authors": ["Noisy Bot"],
        "year": 2020,
        "doc_type": "note",
        "doi": None,
        "body": """# OCR Recovery Fixture

Authors: Noisy Bot
Year: 2020

TTL  i5  250  secands
dose: 5O0 mg
""",
    },
    {
        "doc_id": "binding_coref",
        "title": "Entity Binding Note",
        "authors": ["Binding Author"],
        "year": 2024,
        "doc_type": "note",
        "doi": None,
        "body": """# Entity Binding Note

Authors: Binding Author
Year: 2024

Metformin was administered in the morning. It reduced fasting glucose.
Placebo was given at night. It had no effect.
""",
    },
]


def build() -> dict:
    CORPUS.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    corpus_meta = []
    for d in DOCS:
        path = CORPUS / f"{d['doc_id']}.md"
        path.write_text(d["body"], encoding="utf-8")
        corpus_meta.append(
            {
                "doc_id": d["doc_id"],
                "path": str(path.relative_to(ROOT)),
                "title": d["title"],
                "authors": d["authors"],
                "year": d["year"],
                "doc_type": d["doc_type"],
                "doi": d["doi"],
                "sha256": _sha(d["body"]),
                "n_chars": len(d["body"]),
            }
        )

    # Gold atoms / task expectations (process-planted, not score-cherry-picked)
    gold = {
        "seed": SEED,
        "docs": {d["doc_id"]: {
            "title": d["title"],
            "authors": d["authors"],
            "year": d["year"],
            "doc_type": d["doc_type"],
            "doi": d["doi"],
        } for d in DOCS},
        "planted": {
            "B1_numeric_contradiction": {
                "field": "metformin_dose_mg",
                "docs": {
                    "bio_abs_metformin": 500,
                    "bio_abs_metformin_conflict": 850,
                },
            },
            "B1b_ttl_contradiction": {
                "field": "ttl_seconds",
                "docs": {"tech_note_cache": 300, "tech_note_cache_v2": 600},
            },
            "B2_entity_collision": {
                "string": "ibuprofen",
                "as_reagent_doc": "semi_structured_lab",
                "note": "same string must not auto-type as allergy",
            },
            "B3_missing_field": {
                "doc": "semi_structured_lab",
                "field": "patient_id",
                "expected": "ABSTAIN",
            },
            "B4_paraphrastic": {
                "query": "How long before cached entries expire?",
                "target_doc": "tech_note_cache",
                "lexical_overlap_forbidden_terms": ["expire", "expiration"],
                "answer_span": "300 seconds",
            },
            "dosages": [
                {"doc_id": "bio_abs_metformin", "text": "500 mg"},
                {"doc_id": "bio_abs_metformin_conflict", "text": "850 mg"},
            ],
            "compounds": ["metformin", "placebo"],
            "sample_sizes": {
                "tech_note_cache": 240,
                "bio_abs_metformin": 512,
                "bio_abs_metformin_conflict": 480,
            },
            "emails": {"tech_note_cache": "ada@example.org"},
            "urls": {"tech_note_cache": "https://example.org/cache-rfc"},
            "citation_keys": {"tech_note_cache": "CACHE2024"},
            "kv": {
                "semi_structured_lab": {
                    "device": "spectrometer-7",
                    "wavelength_nm": "532",
                    "operator": "KJ",
                    "status": "calibrated",
                }
            },
            "table_rows": {
                "tableish_throughput": [
                    {"region": "us-east", "qps": "1000", "error_rate": "0.01"},
                    {"region": "eu-west", "qps": "800", "error_rate": "0.02"},
                    {"region": "ap-south", "qps": "600", "error_rate": "0.015"},
                ]
            },
            "mentions": {
                "metformin": ["bio_abs_metformin", "bio_abs_metformin_conflict", "binding_coref"],
            },
            "definition": {
                "doc_id": "tech_note_cache",
                "term": "Latency",
                "text": "Latency is the end-to-end response time measured at p50.",
            },
        },
        "probe_flags": {"B1": True, "B2": True, "B3": True, "B4": True},
    }

    gold_path = GOLD / "gold.json"
    gold_path.write_text(json.dumps(gold, indent=2), encoding="utf-8")

    # Evidence spans for key planted strings
    spans = {}
    for d in DOCS:
        body = d["body"]
        spans[d["doc_id"]] = {}
        for key, needle in [
            ("title_line", d["title"]),
            ("year", str(d["year"])),
        ]:
            sp = _span(body, needle)
            if sp:
                spans[d["doc_id"]][key] = sp
        if d["doc_id"] == "tech_note_cache":
            for key, needle in [
                ("ttl", "300 seconds"),
                ("email", "ada@example.org"),
                ("url", "https://example.org/cache-rfc"),
                ("n", "n=240"),
                ("def", "Latency is the end-to-end response time measured at p50."),
            ]:
                spans[d["doc_id"]][key] = _span(body, needle)
        if d["doc_id"] == "bio_abs_metformin":
            spans[d["doc_id"]]["dose"] = _span(body, "500 mg")
            spans[d["doc_id"]]["n"] = _span(body, "n=512")

    (GOLD / "spans.json").write_text(json.dumps(spans, indent=2), encoding="utf-8")

    manifest = {
        "seed": SEED,
        "n_docs": len(corpus_meta),
        "docs": corpus_meta,
        "gold_sha256": _sha(gold_path.read_text(encoding="utf-8")),
        "inclusion": "wedge_v1/inclusion_predicates.md",
        "track": "clean",
        "probe_flags": gold["probe_flags"],
        "probe_ok": sum(1 for v in gold["probe_flags"].values() if v) >= 2,
    }
    man_path = MANIFESTS / "corpus_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "n_docs": len(corpus_meta), "probe_ok": manifest["probe_ok"], "manifest": str(man_path)}, indent=2))
    return manifest


if __name__ == "__main__":
    build()
