# Active Now

**Updated:** 2026-08-22  
**Program:** Nano Core — capability ladder  
**Frontier:** P1 Master Scribing (medical DomainPack)

## Current gate

```text
REPOSITORY RECONCILIATION + CANONICAL DOC AUTHORITY
        ↓
SELECTIVE NANO MODEL INTEGRATION (as needed for P1)
        ↓
SCRIBE EVIDENCE / SPAN BOTTLENECK
        ↓
VERIFIED ENCOUNTER RECORD
        ↓
COHERENT NOTE REALIZATION
        ↓
EXTERNAL MEDICAL SCRIBE EVALUATION
        ↓
SCRIBE MASTERY DECISION (P1 exit)
```

We are at **step 1–2**: documentation authority reset (this branch) then P1 scribe engineering.

## Program status

| Field | Value |
|-------|-------|
| `program_execution_status` | `DOC_RESET_AND_P1_SCRIBING` |
| `capability_frontier` | `P1_SCRIBING` |
| `program_execution_status` | `DOC_RESET_AND_P1_SCRIBING` |
| `capability_frontier` | `P1_SCRIBING` |
| `evidence_core` | Frozen — Paper α, ledger, protected tags unmoved |
| `paid_compute` | NOT_AUTHORIZED |
| `training` | NOT_AUTHORIZED (unless owner types execute auth) |
| `clinical_claims` | FORBIDDEN without external + human validation |

## Bounded work in flight

1. **Docs migration** (`docs/canonical-program-reset` branch) — canonical `docs/` tree, README rewrite, superseded stubs
2. **Review before merge** — owner reviews file map + README before merging to `master`
3. **Next code frontier** — encounter representation schema + span/evidence path toward verified record (not full P2/P3 build)

## Owner gates (standing)

- Paid RunPod / cloud GPU runs
- PHI or private owner corpus in git
- Protected evidence tag moves
- Publication or clinical capability claims
- Merge doc reset to `master`

## Quick links

- Charter: [PROJECT_CHARTER.md](PROJECT_CHARTER.md)
- Architecture: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- Execution tasks: [EXECUTION_PLAN.md](EXECUTION_PLAN.md)
- Scribing target: [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
