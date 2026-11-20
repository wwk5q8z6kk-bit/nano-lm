# Token Budget Reconciliation

*Generated 2026-07-31T14:00:43.203189+00:00*

## Authoritative table

| Configuration | Parameters | Unique corpus tokens | Total training tokens | Epochs/passes | Tokens/param | Source | Confidence |
|---------------|-----------:|---------------------:|----------------------:|--------------:|-------------:|--------|------------|
| nano_3.15M_pretrain | 3.15M | 10.96M | 32.80M | 3.1 | 10.41 | `pretrain/AUDIT.md` | HIGH |
| scale_10M_pretrain | 10.00M | — | 200.00M | ~1 epoch-ish (AUDIT) | 20.00 | `scale/AUDIT.md` | HIGH_APPROX |
| ownstack_160M_fullft_weak_base | 159.28M | — | 200.00M | — | 1.26 | `trajectory/results_ownstack_v2_160m_fullft.json` | HIGH |
| ownstack_160M_chinchilla_fullft | 159.28M | — | 3200.00M | — | 20.09 | `trajectory/results_ownstack_v2_160m_chinchilla.json` | HIGH |
| ownstack_160M_lora_on_weak_or_chin_base | 159.28M | — | base-dependent (200M or 3.2B pretrain) + | — | N/A (adaptation on p | `trajectory/results_ownstack_v2_160m_lora.json; results_corner_3p2b_lora_seed0.json` | MEDIUM |

## Distinctions

- **Unique corpus size** (nano shard): 10.96M tokens.
- **Total training tokens / optimizer-consumed**: nano 32.8M (repeated passes); scale ~200M; 160M weak 200M; 160M Chinchilla 3.2B.
- **Fine-tuning tokens** are separate and not the scale-law confounder here; the methods error concerns **pretraining**.
- Values marked HIGH_APPROX use audit “~200M” / D≈20N language rather than a single exact counter field in a JSON.

## Methods error

Paper α currently groups 3.15M and 10M as both pretrained on ~200M. **False for 3.15M** (32.8M).

## Hits scanned: 27 lines mentioning 200M/50×/pretrained/flat (see JSON).
