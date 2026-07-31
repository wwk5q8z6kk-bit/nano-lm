# Executive Summary — Discussion-to-Implementation Reconciliation

## Final decision

# DOCUMENTATION_OVERSTATES_REALITY

(Empirical closed experiments are largely real; architecture/product packaging and status prose systematically overshoot the repository.)

## What we thought existed

- A full NanoScribe cognitive architecture (kernel, memory, routing, tools, UI).
- An append-only, transactional evidence fabric with Intent→Control.
- Integrated OSS stacks (Outlines/Guidance/LangExtract/etc.).
- ~~E2 in flight; E3 human pending~~ **SUPERSEDED:** E2=GATED_STOP (no RESULT); E3=agent-rubric EXACT_SURVIVES, clinician/IAA open.
- Post-α decision pipeline fully frozen in the archival record.

## What actually exists

- Strong empirical measurement program with many immutable-style JSON results (often only on disk / untracked).
- Thin fabric vertical slice (~schemas+slice+8 tests) with measured presented-error→0 under rules-strong v2.
- E1 KILL under frozen U with official M0 scored; sensitivity stable.
- E3 normalize 0/486; Stage 1 “human” = **agent rubric** n=100, not clinicians.
- E2: script + partial RunPod residue; **no RESULT**; **no active pods**.
- ~~E4/R★: protocol text only; no builder/data.~~ **SUPERSEDED:** post-α `trajectory/results_e4_utility.json` exists (Gate 4 KILL on tested R★ v1); outside Paper α; further E4 execute BLOCKED.
- NanoScribe beyond fabric: **documents**.
- OSS integrations named in this audit list: **absent**.

## What is proven (scoped)

- Held-out exact copying gaps; field localization; diversity causality (+66.7); within-stack flatness under full FT; LoRA×data behavioral interaction; C-1b REFUTED; C-3 T/B REFUTED; pointer P2 REFUTED for this impl; E1 non-LM dominates U; normalize does not erase gap; fabric v2 can drive presented error to 0 on this synthetic task under decidable R.

## What is partially built

- E3 construct “human” arm; fabric ledger/risk/telemetry; reproducibility packaging; Stage M kernel; CI (present, may be untracked); post-α lockfile corpus (working tree).

## What is missing

- Architecture control plane/memory/UI/distribution.
- E2 results; E4 world; dual-clinician IAA; synonym ontology; OSS adapters; many adversarial ledger tests; git archival of post-α artifacts.

## What is stale

- E2 RUNNING; E3 BLOCKED/pending; preregs “not run” despite JSON; REGIME “Stage 1 skipped”; Paper α “pending human study”; ρ=hallucination; Intent→Control as if real; interference-as-leading remnants.

## What should be removed (from active belief / plan language)

- NanoScribe product architecture as “implemented.”
- Fabric as append-only cognitive OS.
- OSS integrations as in-tree.
- E2 as current critical path for product.
- Any reopen of old-task generative substrate under OLD_TASK_U.

## Largest scientific risk

**Claim / status drift:** treating agent-rubric EXACT_SURVIVES and working-tree “locks” as if they were dual-clinician validation and git-tagged immutability — plus task-isomorphism under-read of E1 KILL.

## Largest architectural risk

**Believing NanoScribe exists** because fabric Phase-1 gates passed — leading to expansion contrary to E1 STOP and inventing modules without a surviving product kill-gate.

## Single highest-value next authorized work unit

**Documentation/archival reconciliation only (no experiments):** apply owner-approved patches from `PROPOSED_OWNER_LOCKFILE_PATCHES.md` items P1–P7 + decide commit-vs-declare for post-α artifacts (P9). Then choose explicitly: **Idle** or **authorize Stage 4 E4** against frozen P2 — nothing in between.


## Evidence classes (discovery discipline)

Labels from `REVISED_DISCOVERY_PASS_ACCEPTANCE.md` (retrievability, not confidence):

| Class | Definition | Qualifies | Example | Does **not** imply |
|-------|------------|-----------|---------|---------------------|
| PUBLIC-ANCHORED | Committed/tagged and publicly retrievable | Tagged commit contains the artifact | E1/E3 primaries at `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb` | Later local overlays are public; clinical readiness |
| LOCAL-DOCUMENTARY | Exists locally / in working tree or conversation archive but is not the public archival story for that claim | Untracked or dirty packaging notes | Session `READY_FOR_OWNER.md`, some freeze packaging churn | Public freeze completeness |
| RAW-UNINSPECTED | Bytes exist but have not been human-audited as a claim basis | Large raw dumps pending review | Some local raw archives before inventory | Scientific endorsement |
| ASPIRATIONAL | Protocol/design/product text without a measured RESULT | Roadmaps, ambition, NanoScribe visions | `TECHNOLOGY_ROADMAP.md` NanoScribe modules | Implementation or evidence |
| DECISION-GOVERNANCE | Program lock / kill / product-thesis scoping | Decision records, utility kill rules | E1 product thesis KILL row | New regime unlocks |
| STALE-CONTRADICTORY | Text that conflicts with measured/public state and must not be reused as current status | Old “E2 running”, “E4 untested” as present tense | Pre-remediation narrative residues | Current program authorization |
