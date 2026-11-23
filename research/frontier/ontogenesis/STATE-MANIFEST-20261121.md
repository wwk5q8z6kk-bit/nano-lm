# State manifest, 2026-11-21 (gate G00, evidence repair)

**Scope.** A read of the owner's machine at about 01:15 UTC on 2026-11-22: every linked worktree of this repository, what each holds that `origin` does not, and what is running. It closes the "owner-local state" blocker listed in `architecture/06_VALIDATION_AND_UNRESOLVED_WORK.md` and the "not yet audited" line in `LEDGER.md`. It records where work is. It does not judge whether any result in that work is correct.

Machine-readable copy with a SHA-256 for every uncommitted file: [`results/state_manifest_20261121.json`](results/state_manifest_20261121.json).

## Finding

Most of the research record of the last five days was not on `origin`.

| Measure | Count |
|---|---|
| Worktrees inspected | 22 |
| Local branches holding commits absent from every `origin` branch | 76 of 112 |
| Unique commits absent from every `origin` branch | 386 |
| Of those, on branches checked out in a worktree | 13 worktrees (per-worktree counts below; they overlap, so they do not sum to a unique total) |
| Uncommitted or untracked files | 85 (1.3 MB) across 8 worktrees |
| Free space on the internal disk at the time | 17–28 GB of 926 GB |

"Absent from `origin`" is a statement about commit objects, so the content was tested separately:

| Content test | Result |
|---|---|
| Local-only commits with a diff | 375 |
| Of those, with a patch-identical commit on any `origin` branch | 21 |
| Branches whose every touched file already exists, byte for byte, somewhere on `origin` | 7 of 76 |
| File versions reachable only from local branches | 2,084 (1.33 GB) |

Squash merges did not carry this work to `origin`: for 69 of the 76 branches, files they changed exist nowhere on the remote. Per-branch figures are in [`results/local_only_content_20261121.json`](results/local_only_content_20261121.json).

## What was preserved

Nothing was pushed to this public repository. After the owner's go-ahead ("do all necessary"), the local branches were pushed to a separate **private** repository, `nano-lm-archive`, after a secrets scan of the local-only history found only hashes, names, run labels and one test-fixture dummy. That covers commits and code, not the git-ignored checkpoints below, which GitHub cannot hold. Final state of that mirror, checked ref by ref against the local repository on 2026-11-22: all 117 local branches match, plus all 20 tags and all 14 stash entries (under `preserved/stash-20261121/`). GitHub rejected the harness's own checkpoint and Codex snapshot refs (26 refs, an internal ref-rule error), so those stay only in the second bundle on the external volume. Local copies were also made on the external volume, in one dated directory:

| Copy | Contents | Check |
|---|---|---|
| `nano-lm-all-local-refs.bundle` | every local branch that existed at the time (113 at bundling; one was removed by another session before the later count of 112), tags and `HEAD` (1.68 GB) | `git bundle verify`: complete history; every worktree tip is reachable from a bundled branch |
| `nano-lm-all-refs-incl-stash-checkpoints.bundle` | the same plus the stash tip, harness checkpoint refs and Codex snapshot refs, which hold 493 further commits (1.68 GB) | `git bundle verify`: complete history |
| `stashes/` | all 14 stash entries as binary patches, with an index of their commit ids | exported with untracked files; not re-applied as a test |
| `uncommitted/` | the 85 uncommitted and untracked files, by worktree | each copy re-hashed against its source, 85 of 85 match |
| `ignored-local-data/` | git-ignored checkpoints and run output (about 13 GB) | complete; see "Ignored local data" below |

A bundle proves commits and refs only. The uncommitted files are safe as copies, not as history: their owners still need to commit them.

## Branches without a worktree

The largest local-only histories are not checked out anywhere: `rescue/p4-cuad-pilot-wip` (174 commits, last touched 2026-10-17), `cursor/span-port-route-b-182e` (172, 2026-08-13), the 43 `orx/*` branches (69 unique), local `master` (34, diverged from `origin/master`; never push it), and the `keep/*` and `frontier/*` copies of the agentic-benchmark work. All are in the bundle.

## Worktrees

`<repo>` is the main checkout, `<openresearch>` the OpenResearch worktree root. "Ahead / behind" is against `origin/master`.

| Worktree | Branch | Tip | Tip date | Commits not on origin | Uncommitted files | Ahead / behind |
|---|---|---|---|---|---|---|
| `<repo>` | `codex/p1-h7-scribing-path` | `aeb4ffed7` | 2026-11-19 | 5 | 0 | 5 / 136 |
| `<openresearch>/chat_01175017-6d0b-43ed-85d3-6710d1e5d62a` | `work/frontier-focus` | `becd28c01` | 2026-11-19 | 0 | 0 | 50 / 168 |
| `<openresearch>/chat_21a46e30-6ab7-4d30-a5be-6f432795fe7c` | `work/frontier-opus` | `a35cb4967` | 2026-11-19 | 38 | 0 | 54 / 168 |
| `<openresearch>/chat_2e45d4b3-7367-4db8-a43d-16df9a70914f` | `work/question-before-architecture` | `6cc1f74e1` | 2026-11-19 | 5 | 0 | 54 / 168 |
| `<openresearch>/chat_6976f6c7-bb29-48da-9a3b-f70b237eeb3b` | `frontier/v1_2-emitter` | `b9155bc57` | 2026-11-20 | 34 | 0 | 35 / 151 |
| `<openresearch>/chat_81a40cb0-f1e0-4342-a90f-6ce53a818345` | `HEAD` | `dc3b31027` | 2026-08-24 | 0 | 0 | 1 / 168 |
| `<openresearch>/chat_91365bcb-394b-43e1-97c8-d8c75676367d` | `HEAD` | `a43944afe` | 2026-11-19 | 32 | 0 | 33 / 151 |
| `<openresearch>/chat_a2d145c2-5ea6-46b9-8dba-b8e8044d28a2` | `HEAD` | `dc3b31027` | 2026-08-24 | 0 | 6 | 1 / 168 |
| `<repo>/.worktrees/foundation-mech` | `research/x03-x05-mechanism-lab` | `519430b64` | 2026-11-19 | 0 | 0 | 2 / 151 |
| `<repo>/.worktrees/memory-boundary-v2` | `research/memory-boundary-v2` | `e5c12fc9e` | 2026-11-18 | 3 | 0 | 3 / 151 |
| `<repo>/.worktrees/nanolm` | `nanolm/foundation-v1` | `5f98cff94` | 2026-11-17 | 2 | 0 | 2 / 151 |
| `<repo>/.worktrees/nanolm-discovery` | `nanolm/architecture-invention-v1` | `1e6848b92` | 2026-11-18 | 0 | 0 | 2 / 151 |
| `<repo>/.worktrees/nanolm-holonomy` | `nanolm/holonomic-discovery-v1` | `205df113c` | 2026-11-19 | 0 | 10 | 5 / 151 |
| `<repo>/.worktrees/nanolm-lab` | `nanolm/discovery-lab-v1` | `c8478d041` | 2026-11-18 | 12 | 44 | 12 / 151 |
| `<repo>/.worktrees/nanolm-master-cycle1` | `research/nanolm-master-cycle1` | `6d069d4ee` | 2026-11-19 | 2 | 12 | 2 / 151 |
| `<repo>/.worktrees/nanolm-paradigm` | `nanolm/autonomous-discovery-v1` | `3925748c9` | 2026-11-18 | 5 | 7 | 5 / 151 |
| `<repo>/.worktrees/nanolm-posttrain` | `nanolm/posttrain-v1` | `813cfabff` | 2026-11-17 | 0 | 2 | 2 / 151 |
| `<repo>/.worktrees/nanolm-unified-loop` | `research/unified-discovery-loop-v1` | `d17877112` | 2026-11-19 | 0 | 0 | 21 / 136 |
| `<repo>/.worktrees/nanolm-v3` | `nanolm/arch-evolution-v3` | `10e141081` | 2026-11-19 | 17 | 0 | 17 / 151 |
| `<repo>/.worktrees/nanolm-v4-fp` | `research/v4-first-principles` | `acc9b7e89` | 2026-10-26 | 0 | 3 | 0 / 151 |
| `<repo>/.worktrees/nanolm-v5-ds` | `research/v5-distributed-systems` | `5af199e48` | 2026-11-19 | 3 | 1 | 3 / 151 |
| `<repo>/.worktrees/v3-exact-recall` | `research/v3-exact-recall` | `679e9c675` | 2026-11-19 | 14 | 0 | 14 / 151 |

## Items the ledger listed as unaudited

| Item | Status found |
|---|---|
| Seven OpenResearch worktrees | Inspected. Two carry agentic-benchmark work (`frontier/v1_2-emitter` and one detached copy of its history); three carry the frontier-model pivot and the MTS-Dialog delimitation replication (`work/frontier-focus`, `work/frontier-opus`, `work/question-before-architecture`); two sit at an August P1 baseline commit, one of them with six modified documentation files. Four of them hold commits that are not on `origin` (38, 5, 34 and 32; the last two share history). |
| `orx/*` branches | 43 local branches: August–September leakage-grid, calibration, placebo, delimitation and external MTS-Dialog cells for P1. 39 of them hold commits that are not on `origin`, 69 unique commits in all. None is checked out in a worktree. |
| `origin/research/foundation-intelligence-program` | Fully contained in `origin/master` (zero commits ahead). |
| V4 first principles | Still three untracked files in `nanolm-v4-fp`. The ledger records it as never run, but one of the three is `research/v4_fp/results/v4_mathematical_smoke_results.json`, so a smoke run exists and is unrecorded. Now copied and hashed; still uncommitted. |
| Omega core, ACE-X9.2, IVE | Still five local commits on `codex/p1-h7-scribing-path` in the main checkout, which is also 28 commits behind its own upstream. Now in the bundle. |
| Holonomy publication draft | `NANOLM_PUBLICATION_DRAFT.md` is tracked at the root of the main checkout and of `nanolm-holonomy`; whether it is the draft the ledger means was not confirmed by reading it. `nanolm-holonomy` also has ten uncommitted files, including `nanolm_holonomy/architecture.py`; copied and hashed. |

## Running at the time of the read

- Address-by-optimiser factorial (G02), preregistered on `research/memory-saturation-factorial`: three cells in progress, results written to the external scratch volume.
- Correction-margin runs (five processes) and a formal-checks script, from the session working on this branch.
- No cloud training process was visible locally. Cloud state was not inspected.

The one load reading taken (01:00 UTC) was 235 on this machine, so wall-clock figures from runs in this window may not be comparable with earlier ones.

## Ignored local data

Git-ignored files that are not caches or virtual environments total about 13 GB across the main checkout and the worktrees. They are model checkpoints and run output, and no bundle can carry them. Caches, virtual environments and tool state directories were left out on purpose.

| Path | Size | Copy status |
|---|---|---|
| `nanolm` worktree, `artifacts/nanolm_v1/` | 5.0 GB | copied, `verify-copy` passed |
| `nanolm` worktree, `artifacts/nanolm_v2_cloud/` | 0.96 GB | copied, `verify-copy` passed |
| `nanolm-discovery` worktree, `artifacts/nanolm_omega_v1/checkpoints/` | 80 MB | copied, `verify-copy` passed |
| `nanolm-v3` worktree, `artifacts/nanolm_v3/runs/` (three `final.pt`, 125M and 350M) | 3.1 GB | copied, `verify-copy` passed |
| `v3-exact-recall` worktree, `artifacts/exact_recall/logs/` | small | copied, `verify-copy` passed |
| main checkout, `checkpoints/` | 3.0 GB | copied, `verify-copy` passed |
| main checkout, ignored paths under `artifacts/` (H4–H7, P4, campaign corpora, run-ops downloads) and smaller items elsewhere | 1.15 GB in 5,985 files | copied; each file re-hashed against its source, all match |

## What remains for G00

1. Done 2026-11-22, after the owner's approval: the git-ignored checkpoints and run output (6,130 files, 13,985,388,021 bytes) are in a private Cloud Storage bucket in the owner's credits project. Public access is blocked and the bucket is in a single region. Total size matches the external-volume copy byte for byte, and the CRC32C checksums of the two files re-sent separately match.
2. The eight worktrees with uncommitted files need a commit by whoever is working in them. Largest: `nanolm-lab` (44 files), `nanolm-master-cycle1` (12), `nanolm-holonomy` (10).
3. The retirement tool counted 21 worktrees against a cap of 12 and found none removable: each is unmerged, has uncommitted files, or holds git-ignored data.
4. Nobody has yet decided, branch by branch, which of the 69 branches with local-only content is live research, which is superseded, and which should be merged or archived.
