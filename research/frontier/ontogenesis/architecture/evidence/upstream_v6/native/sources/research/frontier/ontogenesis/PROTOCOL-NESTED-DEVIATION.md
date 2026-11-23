# Deviation record for `PROTOCOL-NESTED.md` (written before any nested result exists)

Date: 2026-11-21. The registered protocol text is unchanged. This file records how its execution differs from what was first started, and why.

**What happened.** The first execution ran four training processes on the laptop's GPU alongside eight other jobs. Each process grew to 6–7.5 GB and was still growing; with the other jobs the machine filled its 40 GB of swap and the internal disk fell to 17 GB free. I stopped the four processes. They were on their first seed; no result file had been written and no nested result had been observed.

**What changes.** The protocol is executed on a cloud machine using CPU, one thread per run, instead of the laptop GPU. Same code (the listed source files are unchanged since the protocol commit `a627e87`), same arguments, same seeds 21–25, same PyTorch (2.11.0) and NumPy (2.5.1) versions.

**What this can affect.** The task generator and the initial weights are identical, because they come from seeded CPU generators. Floating-point results of training differ between devices, and on this task the outcome of a run is a sudden jump whose timing is sensitive to such differences. So an individual seed may jump on one device and not on the other. The registered arms `dense` and `looped` were to be reused from `PROTOCOL-LOOP2.md`, which ran on the laptop GPU.

**Control added.** `dense` and `looped` are re-run on the cloud CPU for the same seeds. The registered decision rules will be evaluated twice and both reported:
1. as registered: new arms against the reused laptop-GPU `dense` and `looped`;
2. device-consistent: all seven arms from the cloud CPU.
If the two evaluations disagree, the result is reported as device-sensitive and no rule is declared met.

**What does not change.** Arms, seeds, step budget, optimiser settings, the definition of "learned", the decision rules R1–R4, and the statement that a one-seed difference is not a finding.

## Update, same day, still before any nested result exists

The cloud CPU plan above is withdrawn before use. Two measurements made it the wrong choice: the project's cloud CPU quota is 12 cores in total, and one CPU thread needs about 2.5 to 5 hours per run (71 to 138 seconds per 200 steps, measured locally), against roughly half an hour to an hour on the laptop GPU.

**Execution now.** The five new arms run on the laptop GPU, the registered device, with the registered code and arguments. The only difference from the first attempt is scheduling: three runs at a time from one queue, and no other job on the GPU. So no device deviation applies, the `dense` and `looped` arms are reused from `PROTOCOL-LOOP2.md` as registered, and the decision rules are evaluated once, as registered. The extra `dense` and `looped` CPU re-runs are not performed.

## Second update, same day, still before any nested result exists

The laptop-GPU execution was stopped again after ten minutes, with no result written. Three processes had reached 5.7 GB each and the laptop, which also runs the owner's own applications, was back to 41 GB of swap and 18 GB of free disk. The laptop cannot host this protocol without harming the machine.

**Execution now (final).** A rented cloud machine with 112 CPU cores became available (the host of a TPU node, whose CPU cores are not limited by the 12-core quota). The protocol runs there on CPU, one thread per run, all runs in parallel. This is the device deviation described in the first section of this file, and that section's control applies in full: `dense` and `looped` are re-run on the same machine for the same seeds, and the registered rules are evaluated twice (as registered against the reused laptop-GPU arms, and device-consistently with all seven arms from this machine). If the two evaluations disagree the result is reported as device-sensitive and no rule is declared met.
