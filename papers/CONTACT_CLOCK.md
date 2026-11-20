# Contact clock (LAB.B19 / B20 / B27) — design only

**DOC_TYPE:** CONDITIONAL_MITIGATION  
**MAY_AUTHORIZE_EXECUTION:** false  
**Adopted:** 2026-07-31

## Atom

Synthetic mini-corpus U and in-repo papers dogfood can become a **governance substitute for product contact**. Without a clock, the lab accumulates NONCLAIM docs while the useful-capability thesis stays untested on owner-private workflows.

## Mechanism

```text
CLOCK_START = first wedge Phase-2 RESULT timestamp
CONTACT_EVENT = AUTHORIZE_WEDGE_V1_OWNER_CORPUS RESULT
               OR explicit PRODUCT_STOP string from owner
WARN_AFTER = owner-chosen window (default proposal: 14 days of calendar idle on product auth)
ON_WARN = recommend SCIENCE_IDLE_NO_PRODUCT in EXECUTION_QUEUE (docs only)
ON_EXPIRE = do not invent auth; freeze further governance expansion until CONTACT_EVENT
```

## What counts as contact

- Classical+verify on **owner-local** corpus (≥10 docs), results stay local / redacted  
- Pre-written useful/not sentence  
- No PHI in git  

## What does **not** count

- More constitutions / scorecards / one-pagers  
- Synthetic re-scores  
- Papers-folder dogfood (in-repo)  
- LM probes  

## Exit

Either owner-corpus RESULT exists, or owner records `PRODUCT_STOP` / park. Synthetic U citations must keep `corpus_class: SYNTHETIC_MINI`.
