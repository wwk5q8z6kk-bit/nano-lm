# P1 data options

Source check: 2026-10-22. This is a prospective answer to the Kaggle-data question,
not an experiment authorization. No dataset was downloaded or ingested, and no
training, evaluation, or H7 settings were changed for this note.

Kaggle can host versioned datasets and run Python/R notebooks with cloud compute,
including GPUs. It is useful both for bringing our own data and discovering data;
the platform does not establish a dataset's suitability for faithful scribing.
See [Kaggle Datasets](https://www.kaggle.com/docs/datasets) and
[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

A dataset's original provenance, license, version, and task labels matter more
than its hosting site. Prefer the authors' distribution over an unverified
Kaggle mirror; a mirror should be traceable to the original release and retain
its license and attribution. Public availability alone does not establish
compatibility with the project's current [no-PHI/private-data scope](../PROJECT_CHARTER.md).

| Candidate | Primary-source facts | Prospective value and limitation for P1 |
| --- | --- | --- |
| **ACI-Bench** | 207 full dialogue–note pairs: 67 train, 20 validation, and three 40-encounter test sets. Authors describe synthetic encounters and state that no patient data was used or disclosed. The data is CC BY 4.0. [Author repository](https://github.com/wyim/aci-bench), [author paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/10/s41597-023-02487-3.pdf), [original release](https://figshare.com/articles/dataset/aci-bench-corpus_zip/22494601). | Closest of these options to a future P1 dialogue experiment. Paired notes provide realistic output examples, but do not supply Nano's five-field state labels and exact character-span evidence. The paper's sentence-alignment analysis on 20 validation encounters is a different annotation unit. Some released transcripts have known swapped speaker tags; attribution needs explicit treatment. |
| **MTS-Dialog** | 1,701 short simulated dialogues with note sections, released under CC BY 4.0. Annotators constructed the dialogues from de-identified MTSamples clinical notes; the authors state no PHI was used. [Author repository and license](https://github.com/abachaa/MTS-Dialog), [paper, data creation and ethics sections](https://aclanthology.org/2023.eacl-main.168.pdf). | Relevant to short dialogue and section-level documentation, but its de-identified-source lineage differs from wholly fictional cases. Under the current no-PHI rule, that provenance merits explicit review before use; “simulated dialogue” alone is insufficient grounds for ingestion. This is not a finding that the released data contains PHI. |
| **Synthea** | Generates fictional longitudinal patient records, with structured exports including FHIR, C-CDA, and CSV. The generator is Apache 2.0 licensed. [Original project](https://github.com/synthetichealth/synthea). | Useful for controlled synthetic facts and eventual chart-context tests. Its standard exports are EHR records, not spoken dialogue transcripts. A dialogue layer would need separate construction and validation; generator truth alone cannot establish what a patient actually said. |

Our recommendation is to consider **a small ACI-Bench adaptation pilot after H7**,
as a separate experiment. First pin the author release and transcript variant,
check the selected examples' provenance and speaker labels, and annotate the
P1 field states and exact evidence spans. Keep encounter identities and original
splits intact, including alternate transcripts of the same encounter. The paired
note is a reference, not evidence for a claim absent from the input transcript.

The pilot's decision value would be whether the scribing path transfers from
our controlled fixtures to longer, more varied synthetic dialogue. Measure
state errors, unsupported assertions, evidence correctness, and omissions on
the annotated slice; note similarity alone cannot answer that question. A
future result would have its own denominator and could not be pooled with H7
or used as evidence of clinical readiness. This note approves no medical
dataset wholesale and launches no work.
