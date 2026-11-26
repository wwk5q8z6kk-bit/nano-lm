# Prospective synthetic NanoScribe teacher comparison

This capsule tests how many source-supported fields reach a verified five-field encounter note. The 160 invented encounters have 800 preassigned labels and cited character ranges on explicit facts. The input, construction program and study plan were sealed before the first request. The inputs cover clinical speakers and bystanders, corrections, qualifiers, prior episodes, future plans, uncertainty and unrelated timing. [Corpus](fixture.json) · [Original seal](seal.json) · [Prospective plan](plan.md)

## Offline replay

Use Python 3.11 or later and a local Nano-LM checkout at `d109d068a8388a564a769a6d93d8d88c790ecce6`. From the capsule directory, pass the checkout as a directory argument. Replace `path/to/nano-lm` with the local path.

```sh
python3 -B score.py --repo path/to/nano-lm
```

The runner verifies the sealed inputs and implementation fingerprints, the system prompt and the request hash for each saved response. It replays all five fields through the public grounded NanoScribe solver and final note verifier, computes the two combination selectors and the reference-source diagnostic, and compares all aggregate, field, scenario and paired counts to `expected.json`. It uses Python's standard library plus the specified Nano-LM source. No network, credentials, downloaded model weights or GPU are needed. Add `--output replayed.json` to save computed results. To measure an intentionally different source version, add `--candidate --output candidate.json`; in this mode the reference-checkout and score-equivalence checks are skipped. The fixed request prompt must still match. [Replay script](score.py) · [Expected values](expected.json)

`build.py` reproduces the corpus with its original fixed seed. To check construction separately, copy just `build.py` into an empty directory, run it there with Python and compare the generated `fixture.json` SHA-256 to the original seal. The build script refuses to replace an existing corpus. All stored offsets are Python character offsets into the original transcript, with the end excluded.

## What was measured

There are 487 positive reports, 69 explicit denials, 147 uncertain fields and 97 missing fields. A strict exact field requires the right polarity, source interval and scalar, or an appropriate abstention. The latter metric allows both missing and uncertain. The secondary exact-state metric distinguishes them. A licensing error is an asserted scalar for a withheld field, the wrong polarity, or overlap with an annotated distractor for that field. Other wrong intervals and boundary errors are counted separately. An unusable final note earns no correct fields.

The deterministic arm selects the sole admissible, unambiguous scalar from the product menu. Each teacher arm sends the same five-slot system prompt and a verbatim input transcript. The checked teacher feeds its raw output to NanoScribe. Agreement retains a scalar only if the checked teacher and control have the same state and source. Recovery fills an otherwise missing teacher field from the checked control. Both selectors re-run their resulting proposals through NanoScribe. The post-hoc reference-source arm supplies the annotated explicit facts and withholds the others to isolate verifier recall. It is a diagnostic that requires the reference labels. [Sanitized completion records](predictions.jsonl) · [Model settings, prompt, token usage and prices](generation.json) · [Source diagnostics](source-diagnosis.json)

## Interpretation and limits

The 16 construction scenarios contribute ten variations each. Several fields share a transcript and several transcripts reuse vocabulary or templates. This is a mechanism-oriented synthetic test. Neither population-level clinical performance nor safe autonomous use has been established. The gold construction sometimes demands one precisely designated mention even when another says the same thing. The examples and all generated teacher outputs in this capsule are for evaluation only. Training rights were outside the scope of this study.

## Results

| Arm | Exact fields / 800 | Exact citations / 556 available | Exact citations / emitted | Correct abstentions / 244 | Licensing errors |
|---|---:|---:|---:|---:|---:|
| Unique menu choice | 489 | 245 | 245/245 | 244 | 0 |
| Flash before verification | 735 | 515 | 515/558 | 220 | 18 |
| Flash verified | 637 | 393 | 393/400 | 244 | 1 |
| Flash plus control agreement | 476 | 232 | 232/232 | 244 | 0 |
| Flash with missing-slot recovery | 637 | 393 | 393/400 | 244 | 1 |
| Pro before verification | 744 | 514 | 514/549 | 230 | 5 |
| Pro verified | 638 | 395 | 395/396 | 243 | 0 |
| Pro plus control agreement | 476 | 232 | 232/232 | 244 | 0 |
| Pro with missing-slot recovery | 642 | 398 | 398/399 | 244 | 0 |
| Exact reference spans through verifier | 662 | 418 | 418/418 | 244 | 0 |

On the matched field pairs, only Flash was exact on 15 and only Pro was exact on 16. Pro's difference was +1 exact fields and +2 exact explicit facts. Raw predictions, preselected source spans and public verified notes can be rescored without a new model call. [Paired and subgroup counts](expected.json)

| Vertex service | Successful calls | Empty STOP responses | HTTP 429 | Gross USD estimate |
|---|---:|---:|---:|---:|
| Flash | 160 | 0 | 13 | $0.153069 |
| Pro | 160 | 1 | 9 | $1.337576 |

The prospective total at the recorded standard global rates is $1.490645, based on provider-reported prompt, completion and reasoning tokens. There was one request per successful model/encounter pair. Rate-limited attempts used delayed retries. One failed Pro authorization expired during the run. It was renewed and only the unfinished request was replayed. Google returned 1 successful responses without candidate text. These were retained as received. Charges on the final invoice and any credit offsets were unverified. [Generation and rate receipt](generation.json) · [Transport amendment](transport-amendment.json) · [Authorization refresh](credential-refresh.json) · [Google pricing](https://cloud.google.com/model-garden/pricing)

With reference spans provided locally, the verifier still withheld 138 of 556 annotated facts. The number of exact raw proposals lost during verification was Flash 122 (reproduced using annotated sources in 122), Pro 119 (reproduced using annotated sources in 119). The most common reference-proposal rejection reasons were 37 complaints rejected for lacking a clinical span, 32 ratings rejected as unlinked to the current episode, 19 allergens rejected as incomplete names, and 14 durations rejected by a field-type guard. For example, the verifier withheld “throbbing headache” from “The main problem is a throbbing headache” and “five mornings” from “For five mornings.” Six apparent incomplete numeric tokens arose on article-sensitive score boundaries, so exact annotation convention matters. [Counts and witnesses](source-diagnosis.json)

Flash licensed 1 designated-source error(s). Pro licensed 0 designated-source error(s). One duration case deliberately retains distinct annotated occurrences for equivalent assertions under the original scoring rule. A previous “one week” at 258–266 and a corrected “a week” at 319–325 have the same duration. The earlier occurrence fails the source metric even though both imply the same duration. Additional flagged intervals are listed individually in the source diagnosis.

The verified sources support prioritizing independent validation of these checker rejection reasons before increasing teacher spending. Flash cost $0.153069, and Pro cost $1.337576. The same 16 scenario templates and model settings were used throughout. The recorded comparison was stopped after the predeclared paired extension completed.
