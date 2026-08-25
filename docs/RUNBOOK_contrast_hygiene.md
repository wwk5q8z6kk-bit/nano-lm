# Runbook — contrast hygiene and multi-session hygiene

Short, mechanical rules earned by defects that actually occurred in this
program. Each rule names the incident that produced it.

---

## R1 — A contrast must vary exactly one thing, and "distinct" is not "equivalent"

**Incident (C3 arm, voided).** The C3-off prompt removed the gold surface string
*and* changed the question from yes/no to wh-extraction. The measured "C3 effect"
was dominated by form. The existing guard checked that prompts stay **distinct**
— and they were. Distinctness is not sufficient.

**Rule.** Before contrasting cells A and B, prove the diff:

```sh
git diff --stat <shaA> <shaB>          # must touch ONLY the flag module
git diff <shaA> <shaB> -- nanoscribe/prompt.py    # must be EMPTY
```

For the leakage grid the passing signature is exactly one line in
`nanoscribe/leakage.py`. Anything else confounds the contrast.

## R2 — Record the run's code revision in every payload, and refuse cross-revision contrasts

**Incident (near-miss, mine).** I paired an L000 run at `7447df5` with an L100
run at `9523bf4` and nearly reported a cross-form contrast as a C1 effect. The
two runs carried the **same condition label** `C1off_C2off_Qon_QSoff` yet
different question templates, because one cell of a finished grid had been
re-run under a "unified wh question form". Grounded moved 16 → 2 on the label
alone.

This is invisible to every check we have: the condition string matched, the slot
set matched, the suite revision matched.

**Rule.** Emit the run's commit SHA in the payload, and make aggregation refuse
to mix. Note the naive form of this rule is **wrong**: sibling cells necessarily
sit on different SHAs (each carries its own flag commit). The correct invariant
is *same instrument*, not *same SHA*:

- record `git rev-parse HEAD` **and** a hash of the prompt-template module;
- a contrast is legal iff the **prompt-template hashes are equal**;
- an aggregate across cells is legal iff all prompt-template hashes are equal.

**When one cell of a finished grid is re-run, every other cell in that grid is
stale.** Re-run the grid, or label the old cells superseded.

## R3 — Never cite a metric whose value depends on an undocumented instrument choice

**Incident.** `asserted_grounded` for the identical condition is 16/192 under one
question form and 2/192 under another. The published headline used the earlier
form; the artifact predated the re-run by seven minutes.

**Rule.** If a headline number moves under a defensible alternative phrasing of
the instrument, publish both, name the canonical one, and say why.

## R4 — Bound any containment or overlap metric with a length statistic

**Incident.** "The model's quote contains the gold span" is satisfied trivially
by quoting the entire transcript. A bare "79.2% located" would have been read as
"finds the evidence 79% of the time" — exactly how `dc3b310`'s 83% propagated.

**Rule.** Report containment only alongside `len(gold)/len(quote)` and the
quote's length relative to its enclosing unit. State the bound in the claim
itself: *"located within a quote of median length L"*, never a bare percentage.

## R5 — A manipulation check must be built from the channel it is checking

**Incident.** `test_pure_echo_model_is_caught`'s parrot builds its answer from
`spec.raw_value`, not from the prompt it was shown. It is cell-invariant by
construction, so it cannot detect a prompt-channel leak — while appearing to
guard exactly that.

**Rule.** A parrot that tests a *prompt* channel must be constructed **from the
prompt text**. Otherwise "the manipulation check passes" is not evidence.

## R6 — `list_peers scope=repo` does not see sibling orx worktrees. Use `scope=machine`.

**Incident (mine, destructive).** I ran `git worktree remove --force` on a
**live** peer session's workspace after `list_peers scope="repo"` returned
"No other Claude Code instances found". Sibling orx worktrees report their
`Repo:` as the worktree path itself, so a repo-scoped query does not match them.
`scope="machine"` listed the session immediately. Their commits survived and I
disclosed within minutes, but the working directory was destroyed.

**Rule.**

- Before touching **any** shared branch or worktree: `list_peers` with
  **`scope="machine"`**, then match on the `CWD` / `Repo` path prefix yourself.
- A clean `git status` and a HEAD at branch tip are **not** evidence a session is
  dead.
- Prefer `git checkout --detach <sha>` in your own worktree over taking a branch
  another worktree holds. You almost never need the branch itself to read code.
- If you do disrupt a peer: say so immediately, state what survived, and name the
  root cause.

## R7 — Pilot a nuisance parameter before spending, on a disjoint set

**Incident (worked — record it as the positive pattern).** Before the leakage
grid ran, a pilot on throwaway encounters disjoint from every measurement
instance measured $\hat\pi_{C2} = 0/40$ (95% upper 0.072) and registered the
C1×C2 interaction as **unidentified rather than underpowered** — no replicate
count repairs a manipulation the model gives nothing to act on.

The landed data at n=192 confirmed it exactly: `asserted_grounded` identical
across all four C2 pairs in all twelve instances, Δ 0.000, sd 0.000.

**Rule.** When a design's effect size *is* a measurable nuisance parameter,
measure it first on a disjoint set, register the prediction, then run. A null
named in advance and confirmed at scale is stronger evidence than the same null
discovered afterwards, and it costs one minute of local compute.
