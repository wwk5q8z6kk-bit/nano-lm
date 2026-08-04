# Repository guidance

## Working style

- Work from the project goal, current evidence, and active plan. Do not add process for its own sake.
- Keep scientific results, engineering ideas, and AI capability claims separate.
- State only what the evidence supports. Preserve limitations and negative results.
- Prefer the smallest sufficient solver and verify consequential outputs.
- Run experiments only when their result can change the roadmap.
- Research exists to improve Nano. Each study must inform a concrete next step in data, training, architecture, grounding, verification, or abstention; it is not a separate destination.
- Never describe an agent-applied rubric as human or clinician evaluation.
- Preserve existing evidence tags and user-owned working-tree changes.

## Current project state

- The strategic center is `papers/STRATEGIC_RESET.md`.
- Paper alpha is the empirical foundation and remains frozen at `paper-alpha-v1`.
- E1 found that classical/rules methods beat the tested generative references on the closed scribe task under the frozen utility. This is scoped evidence, not a universal architecture claim.
- The active build target is Nano itself: a small, trainable, local-first, verification-gated AI scribe core that turns a supplied conversation transcript into an evidence-bound structured representation and abstains when support is insufficient.
- This repository builds and evaluates the AI, not an end-user product, app, service, UI, or commercial workflow. Audio capture, transcription, deployment, and distribution are outside the active scope.
- Wedge v1 is supporting document-evidence and validation infrastructure. Its purpose is to reveal mechanisms and measurements that can improve Nano, not to replace Nano or become a separate AI target.
- Classical baselines are mandatory for extraction-like tasks as performance floors, diagnostics, and possible verification scaffolds. They do not replace the goal of improving Nano's trained intelligence. Generation must earn its place under matched utility.
- Fabric is a closed-world verification regression harness, not the complete Nano AI or a general cognitive architecture.
- E3 Stage 1 was an agent-applied rubric audit; independent human and clinician validation remain open.
- Core project documents are `papers/STRATEGIC_RESET.md`, `papers/WEDGE_V1.md`, `papers/EVIDENCE_LEDGER.md`, `papers/EXECUTION_QUEUE.md`, and `papers/DECISION_GATES.md`.
