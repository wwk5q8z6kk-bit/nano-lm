# Dataset license and permitted-use record

**Status:** License authority record for corpus sources pinned in
`nano_ai/pretraining/sources.py`. Written 2026-08-04. This file records; it does
not approve. Approval fields live in `sources.py` (`authorization_state`,
`commercial_clearance`) and only the owner advances them.

## 1. `roneneldan/TinyStories` (pinned revision `f54c09fd…8f9c64`)

- **License (declared at pinned revision):** CDLA-Sharing-1.0.
- **Permitted use here:** internal Nano research and bounded local smoke
  training (`authorization_state: approved_for_bounded_local_smoke`). The
  prepared `smoke-1m` set (1.0M train / 0.1M validation tokens, manifest
  `1854b8a1…ed2191`) is the only materialized artifact.
- **Obligations:** CDLA-Sharing requires that if the *data* is published or
  redistributed, it carries the same license and attribution. No publication or
  redistribution of the data is planned; model weights trained on it are not
  the data.
- **Commercial clearance:** `not_reviewed` — irrelevant while use is internal
  smoke only; must be reviewed before any released model claims this corpus.

## 2. `HuggingFaceFW/fineweb-edu` (pinned revision `87f09149…f2b8f9`, config `sample-10BT`)

- **License (declared at pinned revision):** ODC-BY 1.0, subject additionally to
  the CommonCrawl Terms of Use (FineWeb-Edu is a CommonCrawl derivative).
- **Obligations:** ODC-BY requires attribution when the database or a
  substantial extract is publicly used or redistributed. CommonCrawl ToU
  restrict re-identification and require compliance with source-site rights.
  Internal training use with published *weights* is the normal, widely-practiced
  reading, but this program records rather than asserts that reading.
- **Permitted use here (proposed, NOT approved):** pretraining-scale corpus for
  future Nano base models. `authorization_state: proposal_only`;
  `commercial_clearance: blocked_pending_review`.
- **Note:** this is a *different dataset* from the `HuggingFaceFW/fineweb`
  (non-Edu) stream used by the July 2026 anchors — see
  `papers/PRETRAINING_PROVENANCE.md`. The Edu variant was chosen for its
  quality filter; the provenance break is deliberate and documented.

## 3. Blocking-gate status for `fineweb-edu` (as of 2026-08-04)

| Gate | Status | Evidence / next step |
|---|---|---|
| 1. Approve source-content and commercial-use risk | **OWNER-BLOCKED** | Question below. |
| 2. Freeze a contamination exclusion digest set | **Policy defined; computation deferred** | Policy: 13-gram normalized-text digests computed over `native-state-span-dev-v0`, `fresh_v1`, and all H-cycle calibration/fit values, produced by a dedicated tool during the authorized preparation run — *not before*, because hashing the sealed development partition is itself a read and stays behind its seal. `prepare.py` already consumes a digest set and excludes matching windows. |
| 3. Resolve and hash the exact pinned source-file inventory | **SATISFIED** | 14 parquet files, per-file bytes + sha256 in `sources.py`; total 28,518,193,415 bytes. |
| 4. Bounded local preparation run + manifest verify | **Blocked behind gate 1** | `prepare.py` refuses any source not `approved_for_bounded_local_smoke`; `test_fineweb_preparation_is_blocked_at_proposal_stage` pins this. |

## 4. The owner question (gate 1) — ANSWERED 2026-08-05

> Do you approve `HuggingFaceFW/fineweb-edu` (ODC-BY 1.0 + CommonCrawl ToU,
> pinned revision `87f09149…`) as a pretraining corpus for internal Nano
> research — yes (research-internal only), yes (including future released
> weights), or no?

**Owner answer (2026-08-05):** APPROVED for research **and** commercial use,
citing ODC-By v1.0's terms directly ("free to use… use it commercially…
share and distribute"). Standing obligations recorded:
1. **Attribution** to Hugging Face (HuggingFaceFW/fineweb-edu) in any public
   distribution of the dataset or heavily derived artifacts, noting changes.
2. **Common Crawl Terms of Use** compliance (no re-identification; respect
   source-site rights).

`commercial_clearance` may advance to `approved_with_attribution`. The owner
also locked the **$150 rung-1 budget** in the same authorization
(`papers/SCALE_PROGRAM_PREREG.md` gate 2 → cleared).
