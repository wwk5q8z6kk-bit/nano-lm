# Product thesis — what the breakthrough actually is

**Status:** Owner-directed scope amendment, 2026-08-05 ("we need to create a
cutting edge breakthrough useful product… take it to the core"). This
supersedes `STRATEGIC_RESET.md`'s "not an app/service/product" clause **as an
owner decision**, recorded as an amendment rather than drift. Everything below
is derived from this repository's own frozen evidence; no claim here exceeds
what that evidence supports, and §4 states plainly what it does not.

## 1. The finding this program already made (and buried)

Three results in this repo point the same direction, and none of them is
"train a bigger model":

**E1 — the generative substrate lost.** On a pre-registered utility, the
*deterministic, non-generative* solver M1 scored **0.999**; the best generative
reference (M0) scored **0.925**; M2 scored 0.886. Verdict recorded: **KILL**.
For structured extraction from constrained dialogue, an LLM was not necessary
and was not better. (`papers/EMPIRICAL_FOUNDATION.md`)

**Fabric — verification, not scale, produced trust.** Raw generator error of
18.4% (nano 3.15M) and 11.5% (scale 10M) fell to **0.0% presented error with
zero lost-correct answers** under `grounding.v2`, with 100% provenance. The
trust came from the verifier architecture. The same instrument shows a 3.2×
larger model buys only a modest raw-error improvement. (`fabric/README.md`)

> **DENOMINATOR CAVEAT (added 2026-08-05; SUBSTANTIALLY CORRECTED the same day
> after computing the actual numbers — read the correction, it matters).**
>
> **The gate is degenerate — this part stands.** `fabric/slice.py:247-248`
> computes `presented_err / max(1, presented)` and passes when
> `pres_rate <= raw_rate`, so a verifier abstaining on *everything* reports 0.0%
> and **PASSES**. It cannot distinguish a perfect verifier from a mute one. Real
> structural flaw; fixed by backlog B3 — port the coverage floor that
> `scribe/gate_grounded.py:129-133` already gets right. Unparseable generations
> at `slice.py:210-214` likewise enter neither numerator nor denominator.
>
> **But the inference I drew from it was wrong, and the data says so.** I
> claimed the 0.0% hid an uncounted withholding cost. Computed across all 24
> cells of `fabric/results_slice_v1.json`: **withheld = 2,642,
> caught_err = 2,642, lost_correct = 0.** Every withheld field was an error.
> Coverage is 81.5–91.0% *because the generator was wrong about 9–18% of
> fields*, not because the verifier was timid — and `caught_err`/`lost_correct`
> do count it, per cell, in the JSON. Comparing v1→v2 on the same instrument,
> v2 buys 0.00% risk for ~1–2 pp of coverage, all of it error.
>
> **So fabric is NOT an instance of over-abstention, and that claim is
> withdrawn.** What fabric has is a gate that *would not catch* over-abstention
> if it occurred — a latent hazard, not a realised one. The realised
> over-abstention is in the model (H6: absence 48.2%, conflict 57.2%,
> uncertainty 76.0%) and in wedge (3/10 useful on real documents). Two places,
> not three.

**H1–H5 — the model-training path is the slow path.** Five consecutive
preregistered attempts to make the small model do this natively were rejected
under their own gates. (H6 pending.)

**Conclusion:** the differentiated asset is not the model. It is the
**verification layer** — the machinery that makes an output *provably
grounded or explicitly absent*.

## 2. The product

**A scribe whose every asserted value is bound to evidence — or withheld.**

*(Claim status corrected 2026-08-05 after a code-level verification pass.
BUILT = verified in code; SEAM = interface only; NOT BUILT = does not exist.)*

- **BUILT — every field carries one or more source spans**; unsupported values
  are never presented. `wedge_v1/coe/bind.py:bind_ask_payload` resolves every
  evidence dict to an atom and re-checks `body[start:end] == text`;
  `runtime.py:_finalize_public_payload` is a single mandatory boundary and
  `_fail_closed_coe` blanks the payload and re-audits the abstention.
- **BUILT — absence is not inference**: the system structurally cannot assert
  a negative; every unevidenced path returns **ABSTAIN**. (Corrected: the
  earlier draft named `UNVERIFIABLE`/`QUALIFY` states — those exist in
  `fabric/schemas.py`, **not** in wedge. Wedge's vocabulary is ABSTAIN.)
- **PARTIAL — contradictions surface as DISPUTED with both spans**
  (`classical/merge.py:merge_field`) — but only for **three hardcoded fields**
  (`ttl_seconds`, `metformin_dose_mg`, `sample_n`). On an arbitrary corpus this
  marquee differentiator is effectively inert. Generalizing it is L2 work.
- **NOT BUILT — machine-valid structure** (tables/Mermaid compile-or-withhold).
  Zero occurrences of `mermaid` in `wedge_v1/`. This is an L2 **target**, not a
  current property. The earlier draft asserted it; that was wrong.
- **BUILT — local-first**: zero network imports in the package (no `requests`/
  `urllib`/`httpx`/`socket`/model-API clients); stdlib-only runtime.
- **SEAM ONLY — model-agnostic**: `lm/probe.py:LMBackend` is a Protocol whose
  sole implementation is a deterministic stub gated to three allowlisted task
  IDs. There is no local-model adapter, no API adapter, no config surface. The
  architecture permits it; nothing is pluggable today.

Why this is defensible: every vendor has an LLM scribe; none can tell a buyer
*which sentences are true*. Hallucination — not capability — is what stalls
adoption in clinical, legal, and financial documentation. A layer that makes
abstention and provenance first-class is precisely what a regulated buyer can
sign off on, and it is the thing this program has three weeks of frozen
evidence about.

Most of it is already built: `wedge_v1` is 45 modules / 18 test files (288
test functions) implementing Verified Ask — Chain-of-Evidence, abstention,
contradiction detection, retrieval-margin gating, ingest SLA, study lifecycle.

## 3. The honest strategic read

The race we would lose: training a from-scratch model to out-capability
frontier labs. The race nobody is running: making outputs *checkable*. The
second race is where this project's evidence, instruments, and code already
are. Model work (rungs L1–L5) remains valuable — it lowers cost and enables
the private path — but it is the *engine*, not the product.

## 4. What the evidence does NOT support (read before any external claim)

- All headline numbers are **closed-world, synthetic clinic dialogue**. There
  is no evidence of clinical validity, real-world generalization, or safety
  for patient care.
- `grounding.v2` is template-anchored to that synthetic world; it "could solve
  the task alone" in that setting. Real transcripts will break assumptions.
- E3's rubric was **agent-applied**; no independent clinician validation, no
  inter-rater agreement.
- Wedge's utility numbers are measured on its own fixtures. **The owner-corpus
  dogfood gate — a real private folder of ≥10 documents — has never been
  satisfied.**
- Therefore: no commercial claim, no deployment, and no external pilot until
  §5 produces real-corpus evidence.

## 5. CORRECTION — the dogfood already ran, and it failed

*(Added 2026-08-05 after review. The paragraph below this one proposed running
a real-document dogfood as "the fastest possible disproof." That experiment
had already been run and its result was sitting on disk, unread by me.)*

`.studies/p5-repo-papers-20260802-r3` and `-r4`: 10 real markdown documents
(this repo's own `papers/`, ~125 KB), 10 tasks, full identity freezing, zero
audit failures. Results:

| run | USEFUL | CORRECT_ABSTENTION | OVER_ABSTENTION |
|---|---|---|---|
| r3 (first contact) | **0** | 4 | 6 |
| r4 (after fix) | **3** | 4 | 3 |

Recorded decision: **`FIX_REPEATED_FAILURE`**, `first_repeated_failure_class:
OVER_ABSTENTION`, `representative_ready: false`, `manual_baseline:
time_saved_claim_supported: false`. The queries were keyword strings authored
by the same agent that then labeled its own output — so even 3/10 is generous.

**Therefore the measured failure mode of the product is over-abstention.** The
verification layer is real and fail-closed (that part is earned), but on the
only non-fixture corpus ever tested it withheld far more than it should have.
A scribe that abstains on 3–6 of 10 answerable questions is not yet useful,
however trustworthy its abstentions are.

**The real next action is diagnosis, not another dogfood run:** read
`.studies/p5-repo-papers-20260802-r4/`, classify each OVER_ABSTENTION by
cause (retrieval margin too strict? evidence-atom binding too narrow?
predicate decomposition refusing on partial support?), and fix the top cause.
That work is free, local, and it is the gate everything else sits behind.
A *second* corpus (open-licensed, per the rule below) is worth running only
after the known failure class is understood.

## 5b. The follow-on corpus (after diagnosis)

**Dogfood on real, OPEN-SOURCE documents** (see the standing data rule below).
Point `wedge_v1` at a corpus of genuinely third-party, openly licensed
documents — public meeting minutes/transcripts, open-licensed case reports,
public technical docs — measure U, and find where the verification layer fails
outside its own fixtures. Free, local, no GPU, no provider, and the fastest
possible disproof of the thesis in §2. "Real" here means *not authored by us
and not in-distribution*; it does not require private data. Every other
roadmap item — scale rungs, structured outputs, Mermaid, think-arm — is
downstream of knowing whether the core holds on documents we did not write.

Bonus: an openly licensed dogfood corpus is **reproducible and publishable**,
which private files never could be.

### STANDING DATA RULE (owner directive, 2026-08-05)

> **No private data is used for training, evaluation, or dogfooding on this
> machine. ALL material — text, code, transcripts, documents, structured
> examples, anything — must be open-source / openly licensed, acquired through
> `nano_ai/pretraining/` with pinned revisions, per-file hashes, and a recorded
> license.**

Applies to every modality and every stage: pretraining text (fineweb-edu,
ODC-By), code corpora (Stack-v2 slices — permissive-licensed subsets only),
dialogue/summarization sets (license-checked per source), structured/Mermaid
examples (permissive repos only), reasoning corpora (GSM8K MIT, LogiQA), and
dogfood documents. If a source's license cannot be verified and recorded, it
is not used — no exceptions, no "just for testing".

This supersedes the earlier owner-corpus dogfood gate in
`frontier/DEVELOPMENT_PLAN.md` ("a real private folder of ≥10 docs"), which is
hereby **retired and replaced** by an open-licensed real-document corpus.
It also reinforces the existing MIMIC-IV prohibition
(`papers/TRAINING_DATA_REGISTRY.md`) and `.gitignore`'s private-path
exclusions. Any future request to use private material requires an explicit,
separately recorded owner authorization — silence is refusal.

If the layer holds: that failure profile *is* the product roadmap, and the
model rungs become the cost/privacy engine beneath it.
If it breaks: we learn it for $0 instead of after a pilot.

## 6. PUSH BLOCKER — do not push to GitHub until scrubbed

The GitHub remote (`wwk5q8z6kk-bit/nano-lm`) is **PUBLIC**, and 39 local-only
commits contain operational/personal data that would become permanent on push:

- RunPod **account balance in USD** and spend rates — ~12 files under
  `artifacts/nano_h6/runops/**` (`PROVIDER_COST_*.json`, `cost-observation-*`)
- Account **email** `rill-mud-9i@icloud.com` (`PROVIDER_COST_20260805T001014Z.json`)
- Real name ↔ email binding (`papers/SUBMISSION_PACKET.md`)
- **SSH key path + pod IP/port** in 5 `PROVIDER_*` failure artifacts
- `/Users/mac/...` local paths across 12+ ledger files

Verified clean: **no credentials/secrets** in any of the 39 commit patches, and
**no file >50 MB** (largest 11.9 MB; ~42 MB total). Also: the repo has **no
LICENSE** despite being public, and **no CI** (so the 32 open bot PRs showing
"CLEAN" means *no checks configured*, not *tests pass*).

Safe to push right now with zero risk: the 4 local-only tags (all resolve to
commits already on the remote). Everything else waits on a scrub pass.
