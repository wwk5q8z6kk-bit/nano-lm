# Owner corpus (private)

Private documents stay **out of git**. Results are gitignored.

```bash
# Prove path on public synthetic fixture
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood --demo

# Real private folder (outside repo)
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood --corpus ~/Documents/my-research-notes

# Or env
export OWNER_CORPUS=~/Documents/my-research-notes
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood
```

Optional: edit a local copy of `data/owner_dogfood_tasks.json` and pass `--tasks`.  
Outputs: `results_owner_dogfood.json`, `results_owner_failure_gallery.{json,md}`.

Fixture corpus (public, non-PHI): `fixtures/owner_corpus/`.
