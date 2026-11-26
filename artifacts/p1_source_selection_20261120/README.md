# P1 source selection and faithful scribing

These experiments measure whether a model and the NanoScribe verifier can retain an assertion from an eligible source while withholding unsupported fields. The verifier baseline is master commit `d109d068a8388a564a769a6d93d8d88c790ecce6`. Each study has its own inputs, denominator and scoring rules.

| Study | Comparison | Finding |
|---|---|---|
| [Two frontier teachers](teachers/README.md) | 160 invented encounters, 556 expected assertions and 244 withheld fields | The verified Flash and Pro routes retained 393 and 395 exact assertions. Supplying the prepared reference quotations retained 418. The verifier rejected valid evidence even when selection was correct. |
| [Student training](students/README.md) | 44 separately authored synthetic development encounters, 89 expected assertions | The first eligible candidate recovered 65. A tuned 7B adapter recovered 58, and a trained direct reader recovered 58 with zero assertions on withheld fields. |
| [Public transcript transfer](transfer/README.md) | Six longer public ACI-Bench role plays, 14 expected assertions | Raw Flash and Pro recovered 7 and 5 exact spans. The verifier retained 1 and 3. Six ambiguous fields were excluded before replay. |
| [Native Nano training](native/README.md) | Three objectives and three seeds, 183 present and 201 absent fields per seed | Training a native model to copy from the source increased average exact recovery on unseen whole values from 10.4% to 54.1%. It also increased false fills on absent fields for all three seeds. |
| [Native follow-up design](native-followup/design.md) | New disjoint splits and matched counterfactuals | The predeclared comparison tests selection of whole source values and development-calibrated abstention on unfamiliar phrasing. |

The first three studies share the same five-field production verification path. The native experiment uses a separate four-field generated task. All results here concern synthetic data or public role plays. The reference judgments have no clinician adjudication.
