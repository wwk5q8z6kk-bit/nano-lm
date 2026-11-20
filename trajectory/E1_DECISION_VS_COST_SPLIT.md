# E1 Decision vs Cost Split (DESIGN NOTE)

**NONCLAIM / DESIGN_DRAFT.** Does not amend `EVIDENCE_LEDGER` until owner commit.  
**Motivation:** FIRST_PRINCIPLES B18 + consult synthesis (admissibility vs plausibility).

## Problem atom

One packaging label (`PUBLIC_PARTIAL`) made readers unsure whether **KILL** itself was soft. It is not.

| Question | Answer today | Mechanism |
|----------|--------------|-----------|
| Does the frozen kill rule still fire on published utilities? | **Yes** | `aggregate_decision` + offline pytest |
| Are \(L\) and \(C\) device-reconstructed from a clean clone? | **Not yet** | needs `AUTHORIZE_E1_LC_REPLAY` |
| May we claim clinical / open-world readiness from E1? | **No** | FORBIDDEN rows |

## Innovation

Name the two questions differently so neither infects the other:

1. **Admissible decision** — symbolic gate over published numbers.  
2. **Plausible costs** — empirical timers/schedules under a device table.

## Activation path (no new science)

1. Keep shipping offline recompute tests (done).  
2. Keep L/C protocol design (done).  
3. Owner may later add ledger rows / `context_of_use` fields.  
4. Optional OWNER_TAG_OK: detached verdict annotation disclosing this split — never retarget premature freeze tag.
