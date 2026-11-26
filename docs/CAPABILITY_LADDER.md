# General-model capabilities

| Capability | Evidence |
|---|---|
| Language modeling | Held-out next-token loss or bits per scored byte across distinct text sources, with matched context and data accounting. |
| Reasoning and instruction following | Accuracy on held-out tasks, explicit scoring, simple controls and per-task errors. |
| Context and retrieval | Performance as context length or distractors change, with memory and throughput measurements. |
| Adaptation | Transfer to fresh tasks after an identified adaptation procedure, with an unadapted baseline. |

Evaluate the native model first. Attribute contributions from prompting, retrieval, tools and verifiers separately when measuring a larger system. Each study should name which capability it measures.

## Historical medical capability ladder

The former P1–P9 ladder described a product sequence beginning with NanoScribe. The project's general-model research has no clinical prerequisite. The historical terminology and protocol remain available. [Earlier scribing program](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/46c65f0044e81ac73251e18a5a7d69a7299a8079/docs/CAPABILITY_LADDER.md)
