"""Failure gallery exporters — decision-useful buckets (Active Frontier).

Not Layer-1 evidence. Empty buckets mean unobserved on this corpus only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wedge_v1.runtime import ask
from wedge_v1.arch.failure_codes import FINE_BUCKET_TO_CODE

ROOT = Path(__file__).resolve().parent
DEFAULT_DOGFOOD = ROOT / "results_wedge_v1_dogfood.json"

# Canonical fine buckets (always listed; zero = unobserved here, not "solved")
FINE_BUCKETS = (
    "evidence_absent",
    "retrieval_miss",
    "wrong_span_retrieval",
    "verifier_rejection",
    "correct_abstention",
    "over_abstention",
    "entity_type_collision",
    "multi_document_contradiction",
    "unsupported_composition",
    "ingestion_layout_failure",
    "ok_supported",
    "other",
)

REPRO = {
    "ok_supported": "python -m wedge_v1 ask \"…\" --corpus CORPUS",
    "other": "python -m wedge_v1 gallery --from PATH",

    "evidence_absent": 'python -m wedge_v1 ask "…" --corpus CORPUS  # check claims[].evidence',
    "retrieval_miss": 'python -m wedge_v1 find NEEDLE --corpus CORPUS',
    "wrong_span_retrieval": 'python -m wedge_v1 report ask --corpus CORPUS "…"',
    "verifier_rejection": 'python -m wedge_v1 ask "…" --corpus CORPUS  # status ABSTAIN/unsupported',
    "correct_abstention": 'python -m wedge_v1 ask "OOS clinical question" --corpus CORPUS',
    "over_abstention": 'python -m wedge_v1 review --corpus CORPUS --interactive  # label OVER_ABSTENTION',
    "entity_type_collision": 'python -m wedge_v1 scan --corpus CORPUS',
    "multi_document_contradiction": 'python -m wedge_v1 compare TERM --corpus CORPUS',
    "unsupported_composition": 'python -m wedge_v1 ask "multi-hop …" --corpus CORPUS',
    "ingestion_layout_failure": 'python -m wedge_v1 ingest --corpus CORPUS',
}


def classify_fine(result_or_row: dict) -> str:
    """Map ask()/dogfood row → fine bucket."""
    if "ok" in result_or_row and "got_status" in result_or_row:
        row = result_or_row
        got = str(row.get("got_status") or "")
        expect = set(row.get("expect_status") or [])
        fail = row.get("fail_kind")
        if row.get("ok"):
            if got == "CONTRADICTED":
                return "multi_document_contradiction"
            if got == "ABSTAIN" and expect <= {"ABSTAIN"} or expect == {"ABSTAIN"}:
                return "correct_abstention"
            if got == "ABSTAIN":
                return "correct_abstention"
            return "ok_supported"
        if fail == "over_abstain" or (got == "ABSTAIN" and "ABSTAIN" not in expect):
            return "over_abstention"
        if fail == "over_answer" or (
            got in {"SUPPORTED", "CONTRADICTED"} and expect == {"ABSTAIN"}
        ):
            return "verifier_rejection"
        if fail == "wrong_span_or_miss" or not row.get("ok_needles", True):
            return "wrong_span_retrieval"
        if got == "NO_CORPUS":
            return "ingestion_layout_failure"
        return "other"

    result = result_or_row
    status = result.get("answer_status")
    claims = result.get("claims") or []
    banner = str(result.get("contradiction_banner") or "").lower()
    notes = " ".join(str(c.get("notes") or "") for c in claims).lower()
    if status == "NO_CORPUS":
        return "ingestion_layout_failure"
    if status == "CONTRADICTED" or "contradict" in banner:
        if "collision" in notes or "entity" in notes:
            return "entity_type_collision"
        return "multi_document_contradiction"
    if status == "ABSTAIN":
        # heuristic: expected support signals absent
        if result.get("bm25_review"):
            return "retrieval_miss"
        if not claims:
            return "evidence_absent"
        return "correct_abstention"  # default; review can re-label over_abstention
    if status == "SUPPORTED":
        if not claims:
            return "silent_miss" if False else "evidence_absent"
        if any(not (c.get("evidence") or []) for c in claims):
            return "wrong_span_retrieval"
        if result.get("bm25_review"):
            return "retrieval_miss"
        return "ok_supported"
    return "other"


def classify_outcome(result_or_row: dict) -> str:
    """Backward-compatible coarse class + fine alias."""
    fine = classify_fine(result_or_row)
    # keep old names for existing tests where possible
    if "ok" in result_or_row and "got_status" in result_or_row:
        row = result_or_row
        if row.get("ok"):
            got = str(row.get("got_status") or "")
            if got == "CONTRADICTED":
                return "ok_contradicted"
            if got == "ABSTAIN":
                return "ok_abstain"
            return "ok_supported"
        expect = set(row.get("expect_status") or [])
        got = str(row.get("got_status") or "")
        if "ABSTAIN" in expect and got in {"SUPPORTED", "CONTRADICTED"}:
            return "under_abstain"
        if got == "ABSTAIN" and "ABSTAIN" not in expect:
            return "over_abstain"
        if got in {"SUPPORTED", "CONTRADICTED"} and not row.get("ok_needles", True):
            return "wrong_or_miss_needle"
        if not row.get("ok_status"):
            return "status_mismatch"
        return "fail_other"
    return fine


def run_gallery(questions: list[str], corpus_dir: Path | None = None) -> dict:
    items = []
    tallies: dict[str, int] = {}
    fine_tallies = {k: 0 for k in FINE_BUCKETS}
    for q in questions:
        r = ask(q, corpus_dir=corpus_dir)
        kind = classify_outcome(r)
        fine = classify_fine(r)
        tallies[kind] = tallies.get(kind, 0) + 1
        fine_tallies[fine] = fine_tallies.get(fine, 0) + 1
        items.append(
            {
                "query": q,
                "kind": kind,
                "fine_bucket": fine,
                "failure_codes": (r.get("trace") or {}).get("failure_codes") or r.get("failure_codes") or [],
                "answer_status": r.get("answer_status"),
                "n_claims": len(r.get("claims") or []),
                "claims": r.get("claims") or [],
                "contradiction_banner": r.get("contradiction_banner"),
                "trace": r.get("trace"),
                "repro": REPRO.get(fine, "").replace("CORPUS", str(corpus_dir or "CORPUS")),
            }
        )
    return {
        "schema": "nano-lm.wedge_v1.failure_gallery.v1",
        "n": len(items),
        "tallies": tallies,
        "fine_tallies": fine_tallies,
        "items": items,
        "note": "Empty fine buckets = unobserved on this corpus, not proof of capability.",
    }


def to_markdown(gallery: dict) -> str:
    if "buckets" in gallery or "fine_buckets" in gallery:
        return gallery_to_markdown(gallery)
    lines = ["# Failure gallery", "", f"n={gallery.get('n')}", "", "## Tallies"]
    for k, v in sorted((gallery.get("tallies") or {}).items()):
        lines.append(f"- **{k}**: {v}")
    if gallery.get("fine_tallies"):
        lines += ["", "## Fine buckets (zeros = unobserved here)"]
        for k in FINE_BUCKETS:
            lines.append(f"- **{k}**: {gallery['fine_tallies'].get(k, 0)}")
    lines += ["", "## Items"]
    for it in gallery.get("items") or []:
        lines.append(f"### {it.get('fine_bucket') or it['kind']} — {it['query']}")
        lines.append(f"- status: `{it['answer_status']}`")
        if it.get("repro"):
            lines.append(f"- repro: `{it['repro']}`")
        lines.append("")
    return "\n".join(lines)


def build_gallery(
    dogfood: dict[str, Any] | None = None, path: Path | None = None
) -> dict:
    path = path or DEFAULT_DOGFOOD
    if dogfood is None:
        if not Path(path).is_file():
            return {
                "schema": "nano-lm.wedge_v1.failure_gallery.v1",
                "source": str(path),
                "error": "dogfood results missing — run: python -m wedge_v1 dogfood",
                "buckets": {},
                "fine_buckets": {k: [] for k in FINE_BUCKETS},
                "rows": [],
            }
        dogfood = json.loads(Path(path).read_text(encoding="utf-8"))
    rows_out: list[dict] = []
    buckets: dict[str, list[str]] = {}
    fine_buckets: dict[str, list[str]] = {k: [] for k in FINE_BUCKETS}
    examples: dict[str, dict] = {}
    corpus_hint = dogfood.get("corpus") or "CORPUS"
    for row in dogfood.get("rows") or []:
        klass = classify_outcome(row)
        fine = classify_fine(row)
        item = {
            "id": row.get("id"),
            "class": klass,
            "fine_bucket": fine,
            "expect_status": row.get("expect_status"),
            "got_status": row.get("got_status"),
            "ok": row.get("ok"),
            "query": row.get("query"),
            "note": row.get("note"),
            "solver_path": row.get("solver_path"),
            "n_claims": row.get("n_claims"),
            "fail_kind": row.get("fail_kind"),
            "repro": REPRO.get(fine, "").replace("CORPUS", str(corpus_hint)),
        }
        rows_out.append(item)
        buckets.setdefault(klass, []).append(str(row.get("id")))
        fine_buckets.setdefault(fine, []).append(str(row.get("id")))
        if fine not in examples:
            examples[fine] = item
    return {
        "schema": "nano-lm.wedge_v1.failure_gallery.v1",
        "source": str(path),
        "n_tasks": dogfood.get("n_tasks"),
        "n_ok": dogfood.get("n_ok"),
        "accuracy": dogfood.get("accuracy"),
        "buckets": {k: sorted(v) for k, v in sorted(buckets.items())},
        "fine_buckets": {k: sorted(v) for k, v in fine_buckets.items()},
        "fine_counts": {k: len(v) for k, v in fine_buckets.items()},
        "failure_code_tallies": {
            FINE_BUCKET_TO_CODE[k].value: len(v)
            for k, v in fine_buckets.items()
            if k in FINE_BUCKET_TO_CODE and v
        },
        "examples": examples,
        "repro_commands": {k: REPRO.get(k, "python -m wedge_v1 gallery").replace("CORPUS", str(corpus_hint)) for k in FINE_BUCKETS},
        "rows": rows_out,
        "note": (
            "Product failure gallery; not Layer-1. "
            "Zero count ⇒ unobserved on this corpus, not solved."
        ),
    }


def gallery_to_markdown(gallery: dict) -> str:
    lines = [
        "# wedge_v1 failure gallery",
        "",
        f"**Source:** `{gallery.get('source', '')}`",
        f"**Accuracy:** {gallery.get('accuracy')} ({gallery.get('n_ok')}/{gallery.get('n_tasks')})",
        "",
        "> Empty fine buckets mean *unobserved on this corpus*, not that the mode is solved.",
        "",
        "## Fine buckets",
        "",
    ]
    fine_counts = gallery.get("fine_counts") or {
        k: len(v) for k, v in (gallery.get("fine_buckets") or {}).items()
    }
    for name in FINE_BUCKETS:
        n = fine_counts.get(name, 0)
        ids = (gallery.get("fine_buckets") or {}).get(name) or []
        id_s = ", ".join(f"`{i}`" for i in ids[:8])
        more = f" (+{len(ids)-8})" if len(ids) > 8 else ""
        lines.append(f"- **{name}** ({n}): {id_s}{more}" if ids else f"- **{name}** (0): _unobserved_")
    lines += ["", "## Representative examples", ""]
    for name, ex in (gallery.get("examples") or {}).items():
        lines.append(f"### {name}")
        lines.append(f"- id: `{ex.get('id')}` status=`{ex.get('got_status')}` ok={ex.get('ok')}")
        if ex.get("query"):
            lines.append(f"- query: {ex['query']}")
        if ex.get("repro"):
            lines.append(f"- repro: `{ex['repro']}`")
        lines.append("")
    lines += ["## Coarse buckets", ""]
    for name, ids in (gallery.get("buckets") or {}).items():
        lines.append(f"- **{name}** ({len(ids)}): " + ", ".join(f"`{i}`" for i in ids))
    if gallery.get("error"):
        lines += ["", f"**Error:** {gallery['error']}"]
    lines.append("")
    lines.append("_Not Evidence Core._")
    lines.append("")
    return "\n".join(lines)


def write_gallery(path: Path | None = None) -> dict:
    g = build_gallery(path=path)
    out_json = ROOT / "results_wedge_v1_failure_gallery.json"
    out_md = ROOT / "results_wedge_v1_failure_gallery.md"
    out_json.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(gallery_to_markdown(g), encoding="utf-8")
    return g


failure_gallery = run_gallery
