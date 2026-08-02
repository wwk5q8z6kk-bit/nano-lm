# Decision record — Owner accept A1 design

**Date:** 2026-07-31  
**Auth string:** `OWNER_ACCEPT_A1_DESIGN`  
**Trigger:** Owner “proceed” after A1 `DESIGN_COMPLETE_PENDING_OWNER_ACCEPT`  
**UTC:** 2026-07-31T18:09Z

## Decision

Accept the Program A1 R★ revision **design package** in  
`trajectory/PROGRAM_A1_rstar_revision_design.md` (§§5–11).

## What this authorizes

- A1 design is **closed / accepted**.  
- Downstream work may *prepare* execute auth paperwork only.

## What this does **not** authorize

```text
AUTHORIZE_E4_RSTAR_V2_EXECUTE = NOT_GRANTED
world rebuild / data freeze = FORBIDDEN
G-ref train / E4′ score = FORBIDDEN
Program 1 = NOT_AUTHORIZED
training / paid compute = NOT_AUTHORIZED
Fabric / NanoScribe expansion = STOP
```

## Next gate (separate)

To instantiate I*′ and run E4′, owner must issue:

`AUTHORIZE_E4_RSTAR_V2_EXECUTE`

with compute ceiling, venue, and frozen U/C schedule acknowledged.
