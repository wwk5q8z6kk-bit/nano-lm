"""Failure gallery exporter — product UX for wrong span / miss / over-abstain.

Not Layer-1 evidence. Classical dogfood / ask outcomes only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DOGFOOD = ROOT / "results_wedge_v1_dogfood.json"


def classify_outcome(row: dict[str, Any]) -> str:
    """Map a dogfood row into a failure/success class."""
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


def build_gallery(dogfood: dict[str, Any] | None = None, path: Path | None = None) -> dict:
    path = path or DEFAULT_DOGFOOD
    if dogfood is None:
        if not path.is_file():
            return {
                "schema": "nano-lm.wedge_v1.failure_gallery.v1",
                "source": str(path),
                "error": "dogfood results missing — run: python -m wedge_v1 dogfood",
                "buckets": {},
                "rows": [],
            }
        dogfood = json.loads(path.read_text(encoding="utf-8"))
    rows_out: list[dict] = []
    buckets: dict[str, list[str]] = {}
    for row in dogfood.get("rows") or []:
        klass = classify_outcome(row)
        item = {
            "id": row.get("id"),
            "class": klass,
            "expect_status": row.get("expect_status"),
            "got_status": row.get("got_status"),
            "ok": row.get("ok"),
            "note": row.get("note"),
            "solver_path": row.get("solver_path"),
            "n_claims": row.get("n_claims"),
        }
        rows_out.append(item)
        buckets.setdefault(klass, []).append(str(row.get("id")))
    return {
        "schema": "nano-lm.wedge_v1.failure_gallery.v1",
        "source": str(path),
        "n_tasks": dogfood.get("n_tasks"),
        "n_ok": dogfood.get("n_ok"),
        "accuracy": dogfood.get("accuracy"),
        "buckets": {k: sorted(v) for k, v in sorted(buckets.items())},
        "rows": rows_out,
        "note": "Product failure gallery; not Layer-1 ledger claim.",
    }


def gallery_to_markdown(gallery: dict) -> str:
    lines = [
        "# wedge_v1 failure gallery",
        "",
        f"**Source:** `{gallery.get('source', '')}`",
        f"**Accuracy:** {gallery.get('accuracy')} ({gallery.get('n_ok')}/{gallery.get('n_tasks')})",
        "",
        "## Buckets",
        "",
    ]
    buckets = gallery.get("buckets") or {}
    if not buckets:
        lines.append("_empty_")
    for name, ids in buckets.items():
        lines.append(f"- **{name}** ({len(ids)}): " + ", ".join(f"`{i}`" for i in ids))
    lines += ["", "## Rows", ""]
    for r in gallery.get("rows") or []:
        mark = "OK" if r.get("ok") else "FAIL"
        lines.append(
            f"- `{r.get('id')}` [{mark}] class=`{r.get('class')}` "
            f"expect={r.get('expect_status')} got=`{r.get('got_status')}`"
        )
    if gallery.get("error"):
        lines += ["", f"**Error:** {gallery['error']}"]
    lines.append("")
    lines.append("_Not Evidence Core. Classical product dogfood only._")
    lines.append("")
    return "\n".join(lines)


def write_gallery(path: Path | None = None) -> dict:
    g = build_gallery(path=path)
    out_json = ROOT / "results_wedge_v1_failure_gallery.json"
    out_md = ROOT / "results_wedge_v1_failure_gallery.md"
    out_json.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(gallery_to_markdown(g), encoding="utf-8")
    return g


if __name__ == "__main__":
    print(json.dumps(write_gallery(), indent=2))
