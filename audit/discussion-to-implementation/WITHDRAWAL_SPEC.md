# Withdrawal Spec — soft claims (post-α freeze)

**Authority:** Council-of-Five hybrid honesty surface, 2026-07-31  
**Freeze tag:** `post-alpha-evidence-freeze-2026-07-31`  
**Program posture:** `IDLE_AFTER_DOGFOOD` + `AUTHORIZED_NONEXECUTION_WORK: NONE` (E4 `EXECUTED` / `KILL`; science `IDLE_AFTER_E4_KILL`); product wedge: see `papers/EXECUTION_QUEUE.md` / `WEDGE_V1.md`  
**Companions:** `CLAIM_GLOSSARY.md`, `CANONICAL_STATUS_TABLE.md`, `EVIDENCE_MANIFEST.json`, `papers/EVIDENCE_LEDGER.md`  
**Machine-readable:** `WITHDRAWAL_SPEC.json`

Use this when reviewing prose, PRs, or papers. A withdrawn soft claim must not be
reintroduced via synonym, implication, or “everyone knows” framing.

| ID | Soft claim (withdraw) | Evidence bound | Expires when | Replace with | Forbidden rephrase |
|----|----------------------|----------------|--------------|--------------|--------------------|
| W-E3-HUMAN | “E3 human / clinician evaluation is complete” | `results_e3_human.json` rater=`agent-rubric-pass-1`; IAA null; CANONICAL_STATUS_TABLE E3 human arm `NOT_RUN` | Dual-clinician IAA study completed under amended prereg + RESULT | “Agent-applied rubric audit (n=100); EXACT_SURVIVES; dual-clinician IAA open” | “bounded human study done”; “human faithfulness validated”; “clinician pack closed” |
| W-IMMUTABLE-JSON | “Trajectory / lockfile JSONs are immutable forever / every edit is frozen science” | Freeze tag archives a commit; working-tree prose can drift; recipe fingerprints are logged not fail-closed | N/A (permanent packaging discipline) | “Content-addressed / tagged archival state at named freeze tags” | “immutable scientific record” (unqualified); “fail-closed recipe digest lock” |
| W-FABRIC-NS | “Fabric is NanoScribe” / “NanoScribe is implemented” | Fabric = scoped propose→verify→abstain slice; NanoScribe kernel/memory/UI absent (`NANOSCRIBE_VNEXT` boundary; fabric README) | Evidenced NanoScribe modules land + ledger update | “Fabric is a verification / regression harness; ≠ NanoScribe architecture” | “cognitive fabric OS”; “NanoScribe Verified Cognitive Fabric is shipped”; Intent→Control-as-implemented |
| W-50X-FLAT | “50× parameter scale leaves the gap flat” (parameter-only law) | Own-stack configs unequal token budgets (nano **32.8M** vs ~200M / 3.2B); descriptive non-monotonic gap | Equal-token / isolated-N prereg + RESULT overturns descriptive reading | “Across evaluated own-stack full-FT configs (unequal token budgets), gap did not collapse monotonically with N” | “scale does nothing”; “flatness proves capacity ceiling”; bare “50× flat” |
| W-BASELINES-PLURAL | “Non-generative baselines dominate” (plural) | Under frozen E1 \(U\), **M1** strictly dominates official M0; **M2** is within \(\delta=0.05\) of M0 but does **not** beat it | New utility / methods table where ≥2 non-LM methods each beat official generative refs | “M1 (hand-template/rules) wins under frozen \(U\); M2 is near but does not dominate M0” | “classical methods all win”; “templates and dicts beat LMs” (unscoped) |
| W-KILL-UNIVERSAL | “E1 KILL falsifies generative LMs / verification / all scribe products universally” | KILL is under frozen closed-task \(U\) on this instrument; new \(U\) or regime may re-rank | New preregistered utility/regime measurement (e.g. authorized E4) changes decision | “KILL H-substrate for this closed scribe task under frozen \(U\) (\(\delta=0.05\))” | “LMs can’t extract”; “verification is useless”; “product is dead forever” |
| W-RSTAR-NEXT | “E4 is authorized to execute / next stage running / ∃R★ generative value is expected” / “generative wins on tested R★” | E4 RESULT exists: `results_e4_utility.json` → **KILL**; Gate 4 logged | N/A (expired as design-block claim; do not revive as “still pending”) | “E4 `EXECUTED` / **KILL** on tested R★ (`IDLE_AFTER_E4_KILL`); revision budget ≤1 needs fresh owner auth; generative+verify did **not** beat classical under frozen \(U_{R★}\)” | “E4 still design-blocked”; “no results_e4_*”; “product path unlocked by E4”; “Stage 4 queued”; Path-B as shipping work; “generative value expected on this R★” |

## Cross-cuts (always true at this freeze)

- **ρ in E1 \(U\)** = review load, not hallucination.
- **Tag ≠ prereg proof:** `post-alpha-evidence-freeze-2026-07-31` archives state; it does not retroactively prove pre-run preregistration chronology.
- **Paper α** (`paper-alpha-v1`) ≠ **post-α evidence bundle** (E1/E3 primaries at freeze tag) — see `EVIDENCE_MANIFEST.json`.

## Open REPRODUCIBILITY_LIMITATIONs (not withdrawals; still open)

1. Official M0 adapter weight binaries / CUDA bit-identical retrain not in archive.
2. Raw RunPod host logs optional; structured L/C in `trajectory/results_e1_runtime_components.json`.
3. Large Paper-α `*.jsonl` gitignored; local hashed archive under `artifacts/local_raw_archive/`.
4. E3 dual-clinician IAA **NOT_RUN**.
5. Freeze tag ≠ retroactive prereg chronology proof.
