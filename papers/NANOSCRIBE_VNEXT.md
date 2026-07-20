# NanoScribe vNext: Research Conclusion and Architecture Decision

*Owner-authored architecture decision, received 2026-07-20. Captured verbatim below the
reconciliation note. This is the program's forward design: a verification-first
cognitive fabric around the small model — not a generic multi-agent framework.*

## Reconciliation with measured evidence (maintainer note, 2026-07-20)

The document's priorities are not speculative — several are already answered or
directly supported by this repo's pre-registered results:

| vNext element | Status in evidence |
|---|---|
| **Stage C1 (copy-preserving adaptation: LoRA vs full FT)** | **Already measured** (P2 2×2): full FT on the under-trained 160M base reads 16.9±1.7; LoRA on the same checkpoint 7.1±1.2 (seed band ±1.3, pre-registered PASS) — full-parameter adaptation destroys the copy pathway that LoRA preserves. The doc's "highest-priority model-side experiment" is P2's landed result. |
| "Slots behave differently by value diversity" | **Directly supported** (slot-diversity sweep, H-slot SUPPORTED: diversity effect 66.7 pts, categorical per-type flips, position innocent; token-coverage secondary factor). |
| Stage V-line (verification-first, abstention, review routing) | Grounded in Stage G/A: 23/23 hallucinated fields caught, absence verifier +10 omissions, 100% presented precision at 19% review load. |
| Stage C2 (pointer/copy head) | Aligned with Stage M PREREG hypotheses (induction/retrieval circuitry); sequenced post-P2 as the doc's roadmap also implies. |
| Priority D (batched verification / inference engineering) | Batched scorer implemented and validated **byte-identical** on both anchors (0/40 mismatches each); adopted as CUDA fast path, native path remains reference. |
| "Every architectural claim must survive a pre-registered experiment" | The operating rule of this repo (PREREGs, decision rules fixed pre-run, honest-FAIL reporting — e.g. the 1B comparability failure reported as a finding). |

Open vNext items map onto the program roadmap: V1–V3 (evidence packets, semantic
grounding, conformal selective prediction) extend the companion verification track;
C1/C2 and M1 feed P3 (mechanism) and P4 (generality); R1/D1 are engineering stages
gated behind them.

---

## Executive verdict (owner document, verbatim)

NanoScribe should not become a generic multi-agent framework wrapped around a tiny
transformer. Its strongest path is: **a verification-first cognitive fabric** in which
a small model drafts structured hypotheses, deterministic and statistical modules test
each claim, uncertain outputs abstain or route to review, and memory accepts only
provenance-bearing validated state.

Trustworthiness ≠ model accuracy alone:
**trustworthiness = generation + verification + abstention + review routing.**

Core architecture: the **NanoScribe Verified Cognitive Fabric** — (1) interaction/intent
→ (2) deterministic cognitive kernel (policy, permissions, budgets, routing) → (3) task
DAG + (4) context compiler → (5) capability fabric (nano-lm, adapters, retrieval,
symbolic tools, external models) → (6) claim/evidence graph + (7) verification mesh
(presence, absence, type, relation, math) → (8) selective presentation (present /
qualify / abstain / review) → (9) memory consolidation (validated writes only).

Key design commitments:
- **Evidence Algebra**: outputs decompose into atomic claims (s,p,o,τ,σ) with evidence
  sets and verification states {verified, contradicted, unsupported, ambiguous,
  unverifiable}; hard invariant: ¬Found(x) ⇏ ¬x.
- **Memory**: working ⊕ episodic ⊕ validated-semantic ⊕ procedural ⊕ graph; generated
  statements never become persistent memory without classify→provenance→verify→
  contradiction-check→dedupe→scope→expiry→commit; truth lattice (Confirmed/Probable/
  Disputed/Unverified/Contradicted/Superseded); retrieval scored on semantic + lexical
  + graph + temporal + authority + verification − contamination.
- **Capabilities, not personalities**: typed capability contracts; rules-plus-classifier
  routing on expected utility (no learned MoE initially).
- **Model-side priorities**: (A) preserve copying pathways (LoRA/frozen-layers/replay/
  aux copy loss — see reconciliation: measured); (B) explicit pointer/copy head
  (p_gen·P_vocab + (1−p_gen)·P_copy); (C) schema-constrained decoding as a system
  guardrail, evaluated separately from model capability; (D) KV cache, dynamic lengths,
  batched verification (engineering, not intelligence claims); tensor factorization only
  as a gated research axis.
- **Selective prediction**: conformal abstention on a combined score (grounding, type,
  source-role, agreement, retrieval, −conflict); report presented-risk AND review load;
  gate: presented risk ≤ 2.5% without trivial coverage.
- **Bounded-degradation extensibility**: no all-to-all agent communication; typed
  packets; O(n) communication, O(n log n) coordination; independent verification;
  immutable event logs; union-bound reliability reasoning (minimize trust assumptions
  per claim).
- **Not yet**: multi-agent debate, global vector DB, autonomous memory writes, learned
  routing, consensus protocols, tensor networks in the 3M model, dozens of agents.

Experimental roadmap (each stage Question → prediction → measurement → decision):
- **V1 evidence packets** (schema validity 100%, span traceability 100%, no presented-
  error increase) → **V2 semantic grounding** (coverage up, review load < 19%, zero
  held-out regression) → **V3 conformal selective prediction** (presented risk ≤ 2.5%).
- **C1 copy-preserving adaptation** (measured — see reconciliation) → **C2 pointer
  head** (copying vs normalization vs unseen values vs wrong-role/wrong-field slices).
- **M1 validated memory** (false-memory rate, contradiction detection, provenance,
  deletion correctness; adversarial cases incl. superseded facts and injection).
- **R1 capability router** (deterministic baseline first) → **D1 distributed execution**
  (idempotency, versioned tasks, deterministic replay, bounded retries).

Priority order — Immediate: shared model/generation refactor; typed claim/evidence
packets; general verifier interface; risk-coverage evaluation; LoRA-vs-full-FT
copy-preservation (done); constrained source-span decoding; preserve all existing gates
as regression tests. Next: SQLite evidence/memory graph; semantic verifiers; calibrated
abstention; task DAG + deterministic router; execution logs/replay. Later: pointer-
generator; adapters/experts; distributed workers; continual learning; factorization;
neural MoE; formal proofs.

Final decision: **a small copy-aware language model embedded inside a typed,
evidence-bearing, selectively abstaining, verification-gated cognitive fabric** —
model proposes, evidence grounds, verifier tests, risk controller decides, human
reviews uncertainty, memory stores only validated state. Generalize:
field verification → claim verification → memory verification → tool/action
verification → controlled distributed cognition.
