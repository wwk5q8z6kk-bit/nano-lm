# RunPod support ticket — pods allocate and bill but never start

**Send to:** support@runpod.io (or the in-console Help widget, which attaches
account context automatically).
**Account:** `user_3Gipqdtq5P6GQgrqGiAJ2FIWvPJ`
**Compiled:** 2026-08-05, after ~24 hours of the condition persisting.

---

## Subject

Secure Cloud pods reach `desiredStatus: RUNNING` and bill, but never start
(uptimeSeconds stays 0, no SSH, no logs) — 15+ pods across 4 GPU types and 6
datacenters. Community Cloud creation fails separately with a generic error.

## Summary

Since **2026-08-04 ~23:40 UTC**, every Secure Cloud pod created on this account
allocates, is charged, and reports `desiredStatus: RUNNING`, but
`runtime.uptimeInSeconds` stays at **0** indefinitely with an empty `ports`
array — no SSH endpoint, no logs, no container start. Observation windows from
5 to **35 minutes**.

One pod created at **2026-08-04 ~23:10 UTC** (~30 minutes before the first
failure) booted normally on the same account, same CLI, same image, and reached
SSH with full runtime attestation. That is the last successful start.

The condition is still reproducing **24 hours later** (2026-08-05 23:27 UTC,
pod `qsdbb867emhvl9`, CZ, 5 minutes at zero uptime).

## What we eliminated before contacting you

| Variable | Tested | Result |
|---|---|---|
| GPU type | RTX 5090, RTX 4090, B200, H100 SXM | all hang identically |
| Datacenter | EU-RO-1, EU-CZ-1, EUR-NO-1, EUR-IS, AP-IN, DC-IE | all hang |
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` and `1.0.3-cu1281-torch291-ubuntu2404` | both hang |
| Network volume | attached and not attached | both hang |
| CLI flags | with and without `--terminate-after`, `--ports` | both hang |
| API surface | `runpodctl` 2.8.1 (GraphQL) **and** `POST rest.runpod.io/v1/pods` | both hang |
| Observation window | 5, 8, 11, 15, 35 minutes | never starts |
| Account state | balance $100.09, spend $0.01/hr, no alerts | healthy |

## Two distinct symptoms

**1. Secure Cloud — allocates, bills, never starts.** Representative pod IDs,
all terminated after their observation windows:

- `5m1vppg4ppdn38` (RTX 5090, EU-RO-1)
- `eo90bhxit60jbn` (RTX 5090, EU-RO-1, Medium stock)
- `blwrejm7cy6fp2` (RTX 5090, EU-RO-1, 35-minute window)
- `5n12str0kcxb4i` (B200, EU-RO-1)
- `glam332vdpmy00` (RTX 4090, EUR-NO-1, no volume)
- `5860wh5ix6m8k0` (RTX 4090, created via REST API)
- `qsdbb867emhvl9` (RTX 4090, EU-CZ-1, 2026-08-05 — still current)

**2. Community Cloud — creation fails outright.** Attempted 2026-08-05 23:32 UTC:

```
{"error":"failed to create pod: graphql error: Something went wrong.
 Please try again later or contact support.","code":"graphql_error"}
```

Separately, several Secure Cloud attempts in EU-CZ-1, EU-SE-1 and for RTX A6000
were rejected at creation with `"This machine does not have the resources to
deploy your pod. Please try a different machine"` even when the datacenter
listing showed Medium stock.

## What we are asking

1. Is there an **account-level flag, hold, or provisioning gate** on
   `user_3Gipqdtq5P6GQgrqGiAJ2FIWvPJ` set on or around 2026-08-04 23:10–23:40
   UTC? Rapid create/terminate cycles during debugging may have tripped an
   automated heuristic.
2. Does the **spend limit of $80** (against a $100.09 balance) interact with
   pod placement or reservation in any way?
3. If neither: is there a known scheduler/worker-agent condition where a pod is
   claimed and billed but the container never starts and no failure status is
   propagated?

## Billing note

We were charged for pods that never ran a container. Total is small (~$4) and
we are not primarily seeking a refund — the priority is restoring the ability
to start pods. We would appreciate the charges for the non-starting pods listed
above being reviewed.

## Contact preference

Reply by email. We can reproduce on demand and can leave a failing pod alive
for inspection if that helps — say the word and we will create one and not
terminate it.
