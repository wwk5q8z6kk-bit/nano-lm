# OWNER idle / park note

**UTC:** 2026-07-31T18:29:01.174623+00:00  
**Owner message:** multi-line speech-act bundle ending in `idle`

## Classification (per line)

| Locution | Force | Scope | Action taken |
|----------|-------|-------|--------------|
| `authorize push` | `AUTHORIZE_PUSH` | push | **REFUSED in-bundle** — conflicts with `idle` + ambiguous with multi-tip tag list |
| `authorize tag defer` | `AUTHORIZE_TAG` | tag / defer | **REFUSED** — mutually exclusive with other tip policies in same message |
| `authorize tag clean-lineage` | `AUTHORIZE_TAG` | tag / clean-lineage | **REFUSED** — conflict |
| `authorize tag non-freeze-snapshot` | `AUTHORIZE_TAG` | tag / non-freeze-snapshot | **REFUSED** — conflict |
| `authorize tag verdict-annotation` | `AUTHORIZE_TAG` | tag / verdict-annotation | **REFUSED** — conflict |
| `idle` | `IDLE` | none | **HONORED** — stop; no further hybrid bookkeeping this turn |

## Why fail-closed

1. Tip policies are **alternatives** (pick one), not a stack to run all.  
2. `idle` means stop; cannot simultaneously push/tag and idle.  
3. Protected tags (`paper-alpha-v1`, `post-alpha-evidence-freeze-2026-07-31`) must never be force-moved.  
4. Dirty freeze packaging remains uncommitted on purpose.

## To unlock one action, send exactly one line

- `authorize push` — push listed local commits to `origin` (no tags)  
- `authorize tag defer` — log deferral only (current B17-safe default)  
- `authorize tag clean-lineage` — design/execute clean-lineage recipe under tip policy  
- `authorize tag non-freeze-snapshot` — non-freeze-named snapshot tag only  
- `authorize tag verdict-annotation` — additive `verdict/*` annotation  
- `idle` / `park` — remain parked  

No push/tag performed from the conflicting bundle.
