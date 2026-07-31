# Tag audit — `post-alpha-evidence-freeze-2026-07-31`

**Audited:** 2026-07-31 (local + `git ls-remote origin`)
**Do not move or recreate this tag in the H/H′/I/J correction unit.**

| Property | Value |
|----------|-------|
| Local existence | YES (`git show-ref --tags`) |
| Remote existence | YES (`origin`) |
| Type | Annotated tag |
| Tag object | `e916db8cf5f9815b07988265a3d26066fb5a51d1` |
| Target commit | `a9d12cb1c456f6c465284e1d469c6326cb14d329` |
| Target subject | `docs: synchronize post-α claims and canonical status table` |
| Tagger date | 2026-07-31T10:13:44-04:00 |
| Ancestor of current HEAD? | YES (`6f3a823` is 5 commits ahead) |
| Contains E1/E3 primaries? | YES (`results_e1_utility.json`, `results_e3_*.json`, PREREG_E1/E2/E3) |
| Contains H/H′/I/J overlay? | NO (working-tree / later commits only) |
| Contains durable C3 replication JSONL? | NO (replication remains gitignored; local_raw_archive only) |
| Role | **IMMUTABLE_HISTORICAL_EVIDENCE_BOUNDARY** — public archival at sync commit; later hygiene/design commits are separate states; do **not** move this tag |
| Relation to `paper-alpha-v1` | Distinct; `paper-alpha-v1` → `0e01d73` unchanged |

## Archival rule

Preserve this tag as historical. After remaining remediation (H/H′/I/J correction commit, durable raw publish, clean-clone verify), create a **new differently named** annotated tag if needed. Do **not** force-move `paper-alpha-v1` or this freeze tag.


## Three repository states

1. **Evidence freeze (this tag)** — immutable at `a9d12cb1c456f6c465284e1d469c6326cb14d329`.
2. **Post-freeze hygiene** — later commits on master (tests/AAEA/polish/design auth).
3. **Design drafts** — E4 protocol text may advance under DESIGN_ONLY; never retarget this tag for that purpose.

Optional future non-evidence tag name: `e4-design-draft-YYYY-MM-DD` (not automatic).
