# wedge_v1 — Local research document intelligence

```bash
python3 -m wedge_v1 smoke
python3 -m wedge_v1 ask --corpus DIR "…"
python3 -m wedge_v1 compare --corpus DIR metformin
python3 -m wedge_v1 dogfood
python3 -m wedge_v1 owner-smoke                  # example/owner_corpus contact (5 tasks)
python3 -m wedge_v1 owner-dogfood --demo         # public fixture rehearsal
python3 -m wedge_v1 habit                        # weekly local ask/find/compare counts
# export WEDGE_OWNER_CORPUS=~/notes && python3 -m wedge_v1 owner-smoke
```

See `data/OWNER_CORPUS.md`. Results are gitignored. Classical-first; no LM.
Not a Layer-1 Evidence Ledger claim until owner promotion.

## P3 usefulness / habit (local, gitignored)

```bash
python -m wedge_v1 owner-ready --demo
python -m wedge_v1 review --demo --next
python -m wedge_v1 review --demo --label O01:USEFUL --label O03:CORRECT_ABSTENTION
python -m wedge_v1 review --demo --interactive
python -m wedge_v1 habit --json
python -m wedge_v1 gallery --from wedge_v1/results_owner_dogfood.json
# private when ready:
# export OWNER_CORPUS=/path/to/private/docs
# python -m wedge_v1 owner-dogfood --corpus "$OWNER_CORPUS" && python -m wedge_v1 review --corpus "$OWNER_CORPUS" --interactive
```

`LM_PROBE = NOT_INDICATED` until repeated gallery evidence meets the admission rule.

## Architecture lab

```bash
python -m wedge_v1 arch-registry
python -m wedge_v1 adversarial
python -m wedge_v1 evolve
```
