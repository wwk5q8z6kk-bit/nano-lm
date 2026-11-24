# Independent review of the forward reference

**Scope:** causal forward computation, incremental state, checkpoint identity,
reset behavior, loss input types and resource inventory. This review did not
train a model or evaluate language quality. It inspected the source and
reproduced three concrete failure cases before making the repairs below.

## Findings and repairs

| Finding | Reproduced behavior before repair | Repair |
|---|---|---|
| Parameter replacement left the checkpoint digest stale | With default tiny configuration and seed 7, consume tokens `[1,2,3]`, replace block 0's K matrix with zeros, then consume token 4 using the old cache. The call accepted the old version and disagreed with fresh forward logits by **0.11297044445137433** at worst. Array read-only flags did not protect the containing dictionary. | Fingerprint the configuration, parameter names, shapes, dtypes and bytes. Check this fingerprint at `forward`, `new_state` and `step`; reject an altered parameter definition. There is no supported weight-update API in this reference. |
| Unsigned integer logits silently changed cross-entropy | For `uint8` logits `[0,255]` and target 0, subtraction wrapped around and produced about **0.3135 nats**, while float64 gives **255 nats**. | Require floating-point logits. A regression rejects the unsigned input and checks the float64 known answer. |
| Fractional cache dimensions produced a fractional-type inventory | `cache_bytes(default_config, 1, 1.5, 8)` returned `1536.0`, despite a token count having integer semantics. | Require integer, non-boolean dimensions; batch and element size must be positive and prefix length nonnegative. Zero-length caches remain valid. |

The new regression group checks rejection of all three altered-model entry
points, four invalid cache-dimension cases, integer logits and the independent
float64 loss value. The revised suite passes **13/13 recorded behavioral
cases**. The earlier 12-case result remains a historical receipt; it did not
exercise these failure cases and was not overwritten.

## Confirmed implementation boundaries

- The full-prefix causal mask and incremental read/append order agree in both
  tiny dense and shared-block arms. Observed maximum logit discrepancies remain
  below `1.5e-15` at the specified float64 tolerance.
- Reused blocks retain separate KV caches for each logical depth. Weight sharing
  reduces distinct parameter storage without reducing logical-depth cache size.
- Fresh states reset position and caches. Batch rows are independent in the
  tested inputs. The caller is still responsible for starting a new state for a
  new request.
- The parameter formula and actual tiny tensor enumeration agree. The nominal
  15,208,704 and 11,798,784 counts are arithmetic only; no nominal model was
  allocated or trained for this review.

## Practical limit of the repair

Recomputing the parameter digest reads all parameter bytes on each model call.
That is a deliberate correctness check for a tiny, unoptimized reference. It
is **not** a requirement to hash every production checkpoint on every token,
and its execution time cannot support a model-efficiency comparison. A
production implementation needs its own immutable checkpoint lifecycle and
cache-version protocol. The digest and NumPy flags are not an authentication
or adversarial-security boundary.

This review found no additional concrete causal-topology, cache-allocation or
parameter-count defect within the stated tiny forward-only contract. It does
not certify untested shapes, precision regimes, gradients, training, sliding
windows, packed documents or adaptive tokenwise depth.

## Receipts and ledger review

- Revised source: [native_reference.py](../prototype/native_reference.py).
- Behavioral suite: [test_native_reference.py](../prototype/test_native_reference.py).
- Final 13-case receipt: [native_reference_v3_final.json](../results/native_reference_v3_final.json).
- Contract: [implementation contract](../07_IMPLEMENTATION_CONTRACT.md).

At the review point, the 24-event ledger passed full schema validation, eight
local evidence-artifact checksums and exact-prefix comparison with the complete
prior Git ledger. Revisions of D-026 and D-032 correctly point to their original
events. Status labels and literature publication categories still require human
semantic review; these are not inferred from schema validity.

