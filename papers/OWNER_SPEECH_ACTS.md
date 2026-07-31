# Owner speech acts (typed illocution)

**Status:** Layer-3 operating design (NONCLAIM). Not Layer-1 evidence.  
**Adopted:** 2026-07-31  
**Companions:** `FIRST_PRINCIPLES_RISK_MITIGATION.md` (B1, B14, B23), `EXECUTION_QUEUE.md`, `DECISION_GATES.md`

> Natural language is cheap. Authority is typed.  
> Map owner utterances to **illocutionary force** before acting (Austin/Searle speech-act tradition; HCI confirmation design; capability-based least privilege).

```text
DOC_TYPE: SPEECH_ACT_TABLE
MAY_AUTHORIZE_EXECUTION: false
EVIDENCE_STANDARDS: UNCHANGED
```

---

## 1. Why this exists (first principles)

| Atom | Failure |
|------|---------|
| Vocabulary collision | Owner says `continue` (session resume); agent hears “do the gated thing.” |
| Mood ≠ capability | “proceed / go / keep going” lack scope bits (commit ≠ tag ≠ push ≠ execute). |
| Agent stall loop | Fail-closed gate + untyped owner chat → infinite “blocked” turns (wastes context, not science). |
| Auth forgery risk | Agent mints `OWNER_*` / `AUTHORIZE_*` from vibes → false science (P5, P7). |

**Invariant:** P7 typed authority. Mitigations must **clarify force**, not weaken gates.

---

## 2. Canonical speech acts

| Owner utterance (examples) | Force ID | Scope bits granted | Agent MUST |
|----------------------------|----------|--------------------|------------|
| `continue` / `keep going` / `/autonomous-skill` resume | `CONTINUE_SESSION` | none | Work only on **ungated** M0 (docs refresh, refuse logs, lint, decomposition). Never commit/tag/push/execute. |
| `idle` / `park` / `stop` | `IDLE` | none | Stop; write park note; no further hybrid bookkeeping. |
| `authorize commit` | `AUTHORIZE_COMMIT` | `commit` | Write/respect `OWNER_COMMIT_OK` with listed paths; additive commit only. |
| `proceed` (lab convention) | `AUTHORIZE_COMMIT` | `commit` | Same as authorize commit for a **path-restricted** ready commit; does **not** grant push/tag/execute. |
| `authorize push` | `AUTHORIZE_PUSH` | `push` | Push **only when local is ahead** of the listed ref(s); never force-push protected tags. If already synced → noop. |
| `authorize tag` + **one** tip policy | `AUTHORIZE_TAG` | `tag` (default **local only**; `tag_push` needs separate authorize) | Exactly **one** tip policy per act: `defer` \| `clean-lineage` \| `non-freeze-snapshot` \| `verdict-annotation`. Policies are **exclusive** (not a concurrent menu). |
| `AUTHORIZE_*` in `EXECUTION_QUEUE` / typed AUTH_RECORD | `AUTHORIZE_EXECUTE` | `execute` (+ listed recipe) | Run only that recipe; fail-closed if SHA drift. |
| `RATIFY_E4_EXECUTE` \| `VOID_E4_AUTH` \| `PARK_AS_EXPLORATORY` | `DISPOSE_E4` | none (disposition) | Sync status docs; still no freeze fold-in. |
| Ambiguous / novel prose | `UNTYPED` | none | Ask for one force ID from this table; do not invent markers. |
| `Begin ACTIVE FRONTIER` / research-envelope mandate block | `ACTIVE_MANDATE` | `research` `prototype` `local_eval` `frontier_commit` | Work inside envelope; never silently rewrite Evidence Core / protected tags. |

### Tip policies for `AUTHORIZE_TAG` (B17)

| Policy | Agent MUST | Agent MUST NOT |
|--------|------------|----------------|
| `defer` | Log that **freeze-brand** tip stays deferred (reason + protected SHAs). | Mint any tag; retarget freeze tags. |
| `clean-lineage` | Run **only** the clean-lineage recipe (`CLEAN_LINEAGE_FREEZE_RECIPE.md`). | Also mint HEAD non-freeze/verdict tags in the same act; move protected freeze tags. |
| `non-freeze-snapshot` | Create a **local** annotated tag on HEAD whose name does **not** claim freeze. | Push the tag (needs separate push/`tag_push`); use freeze-brand names. |
| `verdict-annotation` | Create **local** additive `verdict/<claim>@<sha>` tag(s) with ancestry + context-of-use. | Push tags; retarget freeze; claim freeze-era status for E4. |

**Multi-line owner messages:** If several tip policies appear **and** the message ends in `idle` / `park`, treat tip lines as **scope glossary / future single-act licenses** unless the owner says `execute now` for one policy. Immediate disposition = **defer** + **park**.

---

## 3. Research-backed design notes (methods, not results)

| Pattern | Application |
|---------|-------------|
| Speech-act / illocutionary force | Separate locution (“continue”) from force (`CONTINUE_SESSION`). |
| Capability-based security | Scope bits on receipts; missing bit → fail-closed. |
| Least privilege | Session-continue never implies publish/tag/execute. |
| Confirmation design (HCI) | High-consequence acts require explicit confirm phrase, not ambient “keep going.” |
| Policy-as-code | Lint + helper scripts enforce allowlists; prose alone is not a license. |
| Admissibility vs plausibility | Gate **admissibility** of an act (typed force) before judging whether the act is “probably what they meant.” |

---

## 4. Agent algorithm (normative)

```text
on owner_message M:
  force ← classify(M) using this table (exact phrases win; else UNTYPED)
  if force == CONTINUE_SESSION:
      do ungated M0 only; refresh refuse log; do not mint OWNER_* 
  if force == UNTYPED:
      reply with this menu (one line each); stop
  if force grants bits:
      write/update typed receipt with scope_bits[], expiry, path allowlist
      execute least privilege; verify postconditions
```

Helper: `scripts/classify_owner_speech_act.py`

---

## 5. What this does *not* do

- Does not let `continue` authorize commit/tag/push/execute.
- Does not move freeze tags.
- Does not reopen E2/E4/fabric/NanoScribe.
- Does not replace `EVIDENCE_LEDGER` with aspirations.

---

## 6. Authorization receipts (single-use)

Typed forces that grant bits SHOULD write a receipt (`wedge_v1/OWNER_COMMIT_OK`, `OWNER_PUSH_OK`, `OWNER_TAG_OK`, or an AUTH_RECORD).

Invariant:

```text
ACTIVE + unexpired + unused  →  may grant the named force once
CONSUMED / REUSABLE=false    →  MUST fail any later authorization check
```

After successful execution, mark the receipt:

```text
STATUS: CONSUMED
REUSABLE: false
AUTHORIZED_RESULT_COMMIT: <sha>   # for commit forces
CONSUMED_AT: <UTC>
GRANTS_PUSH: false
GRANTS_TAG: false
GRANTS_EXECUTION: false
scope_bits: []
```

Helper: `scripts/check_owner_receipt.py` (`--expect-consumed` to audit).

