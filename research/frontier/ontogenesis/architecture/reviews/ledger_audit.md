# Nano LM ledger and reproducibility audit

## Outcome

The research package now supports successive events for an existing decision and new decision IDs without changing the validator. Historical event bytes remain intact. The current implementation passes **35 behavioral regression tests** and **12 deterministic mathematical examples**. A separate full-prior-log comparison passes against the exact live PR #195 ledger at commit `3f9ef1a1b5643acda9ee4cc2a780d4fdcbf4bce8`.

These results establish artifact behavior. They do not establish language-model quality, mathematical novelty, reviewer authenticity, factual truth, or permission to run an experiment.

## Source identity and preserved history

The copied download and live GitHub branch were not byte-identical. Root recovered the authoritative current branch files before edits. The legacy log used here is exactly 7,421 bytes and 10 lines, with SHA-256 `aa55fc010e06563188c6d9b8899e58b20223807d8ecae1ab6d1259c85b7eb3a0`. No legacy event was rewritten. The frozen numerical results were not changed.

`ledger_history.json` records the source commit, exact historical-prefix checksum, event IDs and historical human decision IDs. This detects accidental changes to the pinned prefix. It is not an authenticated signature: changing the log and its manifest together can produce an internally consistent package. The external Git commit is the independent comparison anchor.

For subsequent releases, use `--baseline-ledger PATH` with the **entire exact prior committed JSONL file**. This verifies that all prior bytes, including events added after V2, remain the current prefix. Without that argument, `full_baseline_append_only` explicitly reads `NOT_CHECKED`; the permanent legacy prefix is still checked. A regression demonstrates why the full baseline matters: a coherent rewrite of a later event passes schema checks but fails comparison with the prior log.

## Defects and repairs

| Finding | Previous failure | Repair and evidence |
|---|---|---|
| Fixed decision whitelist | Every decision after D-033 failed validation. | Decisions are discovered from events and human rows; D-900 is appended successfully in a temporary fixture. |
| Decision identity confused with event identity | A second event for D-026 was treated as a duplicate. | Event IDs are unique; repeated decision IDs are valid when linked to that decision's latest prior event. Both legacy revisions and later revisions are exercised. |
| Incomplete schema enforcement | Many wrong types and undeclared fields could survive hand-written checks. | Full Draft 2020-12 validation rejects wrong types, unsupported fields, missing required fields and enum violations. Invalid scalar/array records produce reports instead of crashes. |
| Silent date-format gap | A default `FormatChecker` can lack optional date-format support. | An explicit UTC calendar validator is registered; invalid dates and non-UTC timestamps fail. Append receipt times cannot move backwards. |
| Unprotected history | Neither a log rewrite nor contradictory revision branches had a complete comparison gate. | Frozen legacy prefix hash; optional entire-prior-ledger byte comparison; latest-prior-event linkage; rejection of orphan/stale pointers and duplicate JSON keys. |
| Missing inputs crashed validation | Missing documents or malformed JSON could interrupt the script before a useful report. | Read failures become structured failure messages. New Markdown documents are discovered recursively. |
| Sources were weakly checked | Missing local evidence and invalid references could be overlooked. | Local files, package boundaries and ATX heading fragments are checked; remote URLs are checked syntactically. Optional local evidence artifacts are verified by SHA-256. |
| Curvature check was tautological | The old check compared two spellings of the same expression and never evaluated the function. | It evaluates a nonlinear function, the Jacobian step and actual Taylor remainder. An independent AST mutation removes curvature and makes the check fail. |
| Read-only claim lacked direct evidence | A permission-bit experiment can be misleading under elevated execution. | Tests compare the complete mapping of package paths to file hashes before and after each default script, with bytecode writes disabled. No file creation or byte change is allowed. |
| Output behavior was unclear | The transport script's docstring still advertised a fixed output path. | Its docstring now describes print-only default behavior and exclusive `--out` creation. Existing outputs are preserved by tests. |

The live branch had already corrected the Markdown regex escaping defect. That earlier repair was retained; this pass does not claim to have discovered or authored it.

## Event contract for new entries

The existing common fields remain required: decision and event IDs, UTC recorded time, decision, rationale, alternatives, evidence type, source references, status, next gate, superseded decision IDs, affected model, permission class, outcome and open questions.

Every newly appended event additionally requires:

```json
{
  "schema_version": 2,
  "previous_event_id": null,
  "provenance": {
    "source_commit": "3f9ef1a1b5643acda9ee4cc2a780d4fdcbf4bce8",
    "actor": "assistant review",
    "evidence_note": "Review of the pinned baseline and explicitly identified new work."
  }
}
```

For a new decision, `previous_event_id` is null. For a revision, it must equal the latest earlier event ID for that same decision. `supersedes_ids` names different earlier decisions; it cannot substitute for the same-decision event pointer. Human register rows remain unique by decision ID. Historical D-001–D-023 rows need no fabricated machine history, while later human decision rows require at least one machine event.

Optional `provenance.evidence_artifacts` entries have `path` and `sha256` fields. They should name preserved evidence files, not mutable narrative documents. A `confirmed` status requires a matching `confirmed` outcome, an evidence type other than `hypothesis`, and at least one checksum-verified local artifact. This is a structural minimum, not an automatic scientific confirmation: a reviewer must still assess what that artifact supports.

## Numerical examples added

1. **Actual optimizer displacement:** a gradient orthogonal to `J=(1,1)` becomes nonorthogonal after unequal diagonal preconditioning. Projecting the realized displacement restores the linear constraint. Raw-gradient projection alone is insufficient.
2. **Architecture closure:** `A=diag(2,1)` does not commute with RMS normalization. A nontrivial coordinate permutation does. For attention, `q'=A^{-T}q, k'=Ak` preserves the score; applying `A` to both generally does not.
3. **Backdated history:** an event inserted later with an earlier valid time can change a single-time historical answer. The bitemporal visibility example separately demonstrates why recorded-time cutoffs matter.
4. **Finite-step remainder:** the actual nonlinear change is positive despite zero first-order change, within the stated Lipschitz-Jacobian bound for the constructed function.

These are deliberately small checks and counterexamples. They do not sample an LM or prove the architecture achieves a useful trade-off.

## Commands and report

Run from the architecture directory:

```bash
python3 -B checks/test_research_package.py
python3 -B checks/repo_lightweight_checks.py
python3 -B checks/validate_research_package.py --baseline-ledger /absolute/path/to/prior-ledger.jsonl
python3 -B checks/formal_sanity.py
python3 -B checks/transport_anchor_pilot.py
```

The dependency contract is in `checks/requirements-checks.txt`. The tested runtime versions, command results and exact code hashes are recorded in `review/ledger_checks_report.json`. That report is an intermediate audit record; root should run final package validation after all architecture, evidence and ledger edits are complete, then retain the final results with the full source manifest.

## Remaining scope limits

- Markdown handling covers the inline/reference-link and ATX-heading forms used by this package; it is not a complete CommonMark rendering validator.
- Remote URLs and `source_commit` syntax are checked without fetching or authenticating their contents. Source verification belongs to the separate literature/repository audit.
- Permission labels do not grant authorization, and the tool does not verify who wrote a record.
- Human/machine decision-ID coverage is checked, but natural-language summaries are not semantically proven equivalent to the latest event.
- Full-history preservation requires the prior committed ledger supplied through `--baseline-ledger` or an independently reviewed Git diff. The fixed prefix checksum alone cannot protect future appended content indefinitely.

## Files owned by this repair

- Modified: `checks/validate_research_package.py`, `checks/repo_lightweight_checks.py`, `checks/transport_anchor_pilot.py`, `ledger_event.schema.json`.
- Added: `checks/test_research_package.py`, `checks/requirements-checks.txt`, `ledger_history.json`.
- Restored unchanged from the live baseline: `checks/formal_sanity.py`.
- Preserved: historical `ledger_events.jsonl` and all frozen numerical result JSON files.

