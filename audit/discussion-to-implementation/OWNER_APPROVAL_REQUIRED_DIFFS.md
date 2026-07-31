# Owner Approval Required — Exact Proposed Diffs

> **Status 2026-07-31 (freeze ruling):** Owner discovery ruling authorized applying the full
> claim-synchronization patch set. DIFF A–J / H′ / C′ treated as **applied** under that
> authorization during `post-alpha-evidence-freeze-2026-07-31` preparation. Remaining owner
> action: review local commits + tag, then `git push && git push --tags` when ready.
> E4 remains BLOCKED.

---

## DIFF A — `papers/EMPIRICAL_FOUNDATION.md` operating state (§11–23)

### Replace residual doubt / kill threat lines that say E3 “human pending”

**From:**
```
- **Residual doubt mass (ordered):** external validity / synthetic world; oracle
  verifier → open-world overclaim; human faithfulness (E3 human arm pending);
  LoRA mechanism unidentified (E2 blocked); seed/factorial underpower; morphology
  residual; train nondeterminism. Classical baseline **executed** (E1).
- **Kill threats status:** T5 wrong substrate **FIRED (E1 KILL)** · T3 exact-match
  construct **provisionally stable** under normalize (E3 auto; human pending) ·
  H4 wrong objective open · LoRA U1–U4 unidentified (E2 blocked) · oracle verifier
  overclaim open.
```

**To:**
```
- **Residual doubt mass (ordered):** external validity / synthetic world; oracle
  verifier → open-world overclaim; agent-rubric audit completed for E3 exact-error
  pack (dual-clinician IAA + synonym ontology still open); LoRA mechanism
  unidentified (E2 GATED/STOP, no RESULT); seed/factorial underpower; morphology
  residual (exploratory); train nondeterminism. Classical baseline **executed** (E1).
- **Kill threats status:** T5 wrong substrate **FIRED (E1 KILL)** · T3 exact-match
  construct **EXACT_SURVIVES** under normalize + agent-rubric audit (clinician IAA
  open) · H4 wrong objective open · LoRA U1–U4 unidentified (E2 GATED/STOP) ·
  oracle verifier overclaim open.
```

Also replace any remaining “E3 human pending” in Immediate next actions with the
agent-rubric wording.

---

## DIFF B — `papers/RESEARCH_PROGRAM.md` measured foundation + kill-gate list

**From (table rows):**
```
| E3 normalize construct | Auto: exact **not** overstated (0 rescues); human **BLOCKED** | ...
| E2 LoRA universes | **BLOCKED** (GPU/adapters) | ...
```

**To:**
```
| E3 construct | Auto 0/486; agent-rubric audit EXACT_SURVIVES; clinician IAA open | `results_e3_normalize_construct.json`, `results_e3_human.json` |
| E2 LoRA universes | **GATED/STOP** (no RESULT; post-KILL) | `PREREG_E2_lora_universes.md` |
```

**From:**
```
2. **E3** — auto arm **EXACT_NOT_OVERSTATING_BY_NORMALIZE**; human arm **BLOCKED**
3. **E2** — prereg frozen, execution **BLOCKED**
```
**To:**
```
2. **E3** — auto + agent-rubric audit **EXACT_SURVIVES**; dual-clinician IAA / synonym ontology open
3. **E2** — prereg frozen, **GATED/STOP**, no RESULT artifact
```

Replace “E3 human pending” in caveats with agent-rubric / IAA-open wording.

---

## DIFF C — `trajectory/REGIME_P1_where_classical_fails.md` status table

**From:** `Paper α | FROZEN; exact-match limitation remains (Stage 1 skipped)`
**To:** `Paper α | FROZEN; Stage 1 agent-rubric audit executed (Gate 1 PASS); clinician IAA open`

**From:** Next Stage 3 / P2
**To:** Stage 3 / P2 **DONE** (`PREREG_E4_Rstar_killgate.md`); next owner decision = authorize Stage 4 or Idle

---

## DIFF D — Paper α limitation (`papers/paper1_draft.md` + `papers/latex/paper1.tex`)

**From (approx):** exact-match has not been validated against human-accepted equivalence …
pending a bounded human study

**To:**
```
A bounded agent-applied rubric audit of 100 sampled errors found zero acceptable
semantic equivalents (`trajectory/results_e3_human.json`). This does not substitute
for independent clinician annotation, inter-rater agreement, or validation of a
synonym ontology. Automated normalize-then-match rescues 0/486 M0 exact failures.
```

Also add E1 scoped sentence if missing:
```
Under the frozen old-task utility and synthetic extraction world, non-generative
baselines dominate the evaluated generative systems. This falsifies the
generative-substrate product thesis for that regime; it does not establish that
no generative-value regime exists.
```

---

## DIFF E — `papers/EVIDENCE_LEDGER.md` (strengthen rows)

Add/ensure columns conceptually: claim ID, scope, protocol, result, raw manifest,
recompute, limitations, withdrawal, last reviewed.

Key wording updates:
- E3 human row → agent-rubric; status Supported (limitation) with REPRODUCIBILITY note that pack is local
- Add REPRODUCIBILITY_LIMITATION on C-1b/C-3: raw JSONL gitignored; local archive under `artifacts/local_raw_archive/`
- Keep PROVEN only where reproducibility from preserved artifacts holds; else SUPPORTED + limitation
- NanoScribe/fabric product thesis remains Falsified for E1 world
- Generative value in R★ remains Plausible (untested)

---

## DIFF F — `papers/NANOSCRIBE_VNEXT.md` / `MASTER_PLAN.md` architecture truth banner

Add at top (after STOP note if present):
```
NanoScribe is an architectural research program. Beyond the fabric vertical slice,
its control plane, memory, routing, tools, permissions, distributed execution, and
user interface remain unimplemented unless separately evidenced.
Fabric vertical slice ≠ NanoScribe architecture.
```

Scrub any remaining “interference leading candidate” as active (C-1b REFUTED).

---

## DIFF G — sensitivity conclusion string (optional, non-verdict)

In `results_e1_utility_sensitivity.json` / utility JSON `weight_sensitivity.conclusion`,
replace phrase “hallucination weight” with “review-load (ρ) weight” when referring to β.
**Do not change numeric U fields.**

---

## Already correct (no owner action needed for ρ in DECISION_P1)

`trajectory/DECISION_P1_program_lock.md` already defines ρ as review load matching code.
`trajectory/PREREG_E2_lora_universes.md` already GATED/STOP after external check.


---

## DIFF H — Paper α / extension: pretraining-token and “50× flatness” correction (CRITICAL)

**Primary evidence**

| Source | Claim |
|--------|-------|
| `pretrain/AUDIT.md` (PUBLIC) | nano 3.15M: **4000 steps / 32.8M tokens** (~3.1 epochs); shard 10.96M unique |
| `scale/AUDIT.md` (PUBLIC) | Stage S 10M: **~200M tokens** (D≈20N) |
| `results_ownstack_v2_160m_fullft.json` | `target_tokens: 200000000` |
| `results_ownstack_v2_160m_chinchilla.json` | `target_tokens: 3200000000` |
| `papers/paper1_draft.md` / `paper1.tex` methods | “3.15M … and 10M … pretrained on **~200M** FineWeb tokens” — **FALSE for 3.15M** |

**Replace methods sentence**

From:
> … at 3.15M … and 10M …, pretrained on ~200M FineWeb tokens (D≈20N) …

To:
> … at 3.15M (pretrained on **32.8M** FineWeb tokens over ~3.1 epochs of a 10.96M-token shard; see `pretrain/AUDIT.md`) and 10M (pretrained on **~200M** tokens; see `scale/AUDIT.md`) …

**Replace “flat across ~50×” causal framing**

Do **not** state a parameter-only law that scale does not buy copying.

Use instead:
> Increasing parameters under the tested, **unequal pretraining schedules** did not
> monotonically eliminate the held-value gap: a 159M model trained on only 200M tokens
> remained high-gap (16.9±1.7 diluted), whereas substantially more pretraining data
> (3.2B tokens; ~20 tok/param) reduced it (7.0±1.0), and combining that with LoRA
> reached 4.2±0.9. Across 3.15M→10M both parameters and tokens changed; across
> 10M→159M/200M tokens-per-parameter collapsed from ~20 to ~1.26.

**Also update:** `papers/paper2_draft.md` §3.1 “flat across 50×”; `EMPIRICAL_FOUNDATION`
“within-stack scale flatness (~50×)”; `RESEARCH_PROGRAM` same row; `EVIDENCE_LEDGER`
“Within-stack scale (~50×)…” — downgrade from **Proven** pure-scale claim to
**Supported** unequal-schedule descriptive claim, or rewrite wording as above.

**Do not alter** the measured diluted-gap numbers themselves.

---

## DIFF I — E1 public wording: M1 dominance vs plural “baselines dominate”

From (overbroad):
> Non-generative baselines dominate official generative LM references

To:
> Under the frozen old-task utility and synthetic extraction world, the best
> non-generative baseline (**M1** hand-template/rules extractor) strictly outperforms
> the official generative references (U≈0.999 vs 0.925). **M2** (train-dict+span) is
> below official M0 on U (0.886) but within the pre-registered δ=0.05 non-necessity
> margin. This falsifies generative-substrate **necessity** for that regime; it does
> not establish that no generative-value regime exists. M1 is generator-aligned
> (rules-perfect for this synthetic world); rule-authoring/maintenance cost is not in U.

---

## DIFF J — E3 classification

Public and lockfile language must classify Stage 1 as:
> **agent-applied rubric audit** (not “human arm complete”).

Human/clinician equivalence remains **unvalidated**. Internal gate label
`EXACT_SURVIVES` may be reported as an agent-rubric threshold result only.


---

## DIFF C′ — `trajectory/REGIME_P1_where_classical_fails.md` (current working-tree wording)

**From:**
```
| Paper α | FROZEN; exact-match limitation remains (Stage 1 bounded human done; IAA open) |
```

**To:**
```
| Paper α | FROZEN; Stage 1 **agent-applied rubric audit** executed (Gate 1 PASS); dual-clinician IAA + synonym ontology open |
```


---

## DIFF H′ — exact LaTeX locus (`papers/latex/paper1.tex` methods)

**From (current file):**
```
... at 3.15M (d=192, L=6, H=6, KV=2, ff=512) and 10M (d=320, L=8, H=8, KV=2, ff=864), pretrained on $\sim$200M FineWeb tokens (D$\approx$20N; ...
```

**To:**
```
... at 3.15M (d=192, L=6, H=6, KV=2, ff=512; pretrained on 32.8M FineWeb tokens over $\sim$3.1 epochs of a 10.96M-token shard; \texttt{pretrain/AUDIT.md}) and 10M (d=320, L=8, H=8, KV=2, ff=864; pretrained on $\sim$200M tokens; \texttt{scale/AUDIT.md}), full-FT on the scribe task; ...
```

Also narrow any “flat across $\sim$50$\times$” claim per DIFF H body (unequal token budgets; not parameter-only law).
