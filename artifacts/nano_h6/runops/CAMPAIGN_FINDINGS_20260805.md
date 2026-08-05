# H6 replication campaign findings — 2026-08-05 (00:10–01:50 UTC)

**Terminal conclusion:** RunPod secure-cloud RTX 5090 pods are not booting
tonight, independent of datacenter, network-volume attachment, and image.
This is a provider-side fleet problem, not an H6 or science problem and not
a configuration problem on our side. Development remains sealed; the one-shot
gate is unconsumed; the frozen H6 science is untouched.

## Attempt table (all under guard ledgers unless noted)

| # | Time (UTC) | Pod / target | DC | Volume | Image | Outcome |
|---|---|---|---|---|---|---|
| 1 | Aug 4 23:40 | 5xvs95bqcynzn2 | EU-RO-1 | 04himzqxbm | frozen | zero uptime 5 min → terminated |
| 2 | Aug 5 00:10 | hu-r2 5m1vppg4ppdn38 | EU-RO-1 (Low) | 04himzqxbm | frozen | zero uptime 11 min → terminated |
| 3 | Aug 5 00:24 | r3 eo90bhxit60jbn | EU-RO-1 (**Medium**) | 04himzqxbm | frozen | zero uptime 11 min → terminated |
| 4 | Aug 5 00:37 | r4 blwrejm7cy6fp2 | EU-RO-1 (Medium) | 04himzqxbm | frozen | zero uptime **35 min** → terminated |
| 5 | Aug 5 01:19 | cz1 target | EU-CZ-1 | none | frozen | **rejected at allocation** (no resource, $0) |
| 6 | Aug 5 01:20 | no1 glam332vdpmy00 | EUR-NO-1 | **none** | frozen | zero uptime 15 min → terminated |
| 7 | Aug 5 01:37 | infra-probe d1i8kn9vnmdkuy (off-ledger diagnostic) | EU-RO-1 | none | **current 1.0.3/torch291** | zero uptime 8 min → terminated |

Reference: the ONLY successful boot of this configuration all week was the
ro1-retry pod (Aug 4 ~23:10 UTC), which reached SSH and full runtime
attestation before failing on the (since-understood) kernel pin.

## Hypotheses tested and refuted tonight

1. **Stock level** (launch on Medium vs Low) — refuted by attempt 3/4.
2. **Cold image pull needs longer than 11 min** — refuted by the 35-minute
   window (attempt 4).
3. **EU-RO-1 pool-specific** — refuted by attempt 6 (EUR-NO-1, same hang).
4. **Network-volume attachment pins bad hosts** — refuted by attempts 6–7
   (no volume, same hang).
5. **Frozen image stale/unpullable** — refuted by attempt 7 (current image,
   same hang, same DC that allocates readily).

## Spend

Guard ledger h6r-execution-20260804: $1.10 (of $5). Guard ledger
h6r2-cross-dc-20260805: $0.31 (of $3.50). Off-ledger diagnostic probe:
≈$0.13. Campaign total ≈ **$1.54**. Both 20 GB volumes retained
(~$0.0035/hr combined). Zero active pods at close.

## Ledger state at close

- `h6r-execution-20260804`: phase AUTHORIZED, verified, no active op —
  EU-RO-1 attempts closed by decisive evidence; authority remains unconsumed.
- `h6r2-cross-dc-20260805`: phase AUTHORIZED, verified, no active op — one
  operation reconciled_failed carrying the full documented record, including
  a sequencing deviation (the EUR-NO-1 create executed after its guard
  intent call was refused; documented in the operation's closing reason,
  outcome unaffected — that pod never booted).

## Resume conditions (next session picks up here)

1. **Boot-health precheck (no science, ≤$0.05):** create one minimal pod
   (any current image, RTX 5090, secure, EU-RO-1), poll 5 min. If it boots,
   the fleet has recovered → terminate it and run the authorized replication
   under `h6r-execution-20260804` (EU-RO-1 + volume, all pins intact).
   Recheck: `runpodctl pod create … && runpodctl pod get <id>`.
2. If boots resume but the **frozen image specifically** still hangs, that
   is the image-deprecation scenario → requires a disclosed authority
   amendment (image swap with byte-identical wheel versions) — **owner
   decision, not taken unilaterally**.
3. A RunPod support ticket about zero-uptime RTX 5090 secure-cloud pods is
   warranted if the state persists past ~12 h — owner-facing action.

## Provider status-page check (2026-08-05 01:55 UTC)

`uptime.runpod.io` (checked 01:55 UTC, page updated 01:49 UTC): **"all
systems operational"** — the provider has NOT acknowledged the condition.
Precedent for the symptom class exists in their own history: "May 25–26:
Elevated image pull error rate" (19 h degraded, upstream systems). Timeline
sharpening: the last successful boot of this account+config was ~23:10 UTC
Aug 4 (ro1-retry pod); the first hang began 23:40 UTC — the regression
window is ~23:10–23:40 UTC Aug 4, unannounced. This strengthens the
support-ticket recommendation: they likely do not know.

Nothing in tonight's failures spent the one-shot development gate, touched
sealed data, or modified any frozen identity.
