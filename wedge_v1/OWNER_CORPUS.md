# Owner corpus (private)

Private documents stay **out of git**.

```bash
# Option A — env
export OWNER_CORPUS=~/Documents/my-research-notes
python3 -m wedge_v1 owner-dogfood --smoke

# Option B — gitignored folder
mkdir -p wedge_v1/data/owner_corpus
# copy md/txt/pdf here (never commit)
python3 -m wedge_v1 owner-dogfood --corpus wedge_v1/data/owner_corpus --smoke

# Option C — synthetic/example demo (CI / no private docs)
python3 -m wedge_v1 owner-dogfood --demo --smoke
```

Edit `wedge_v1/data/owner_dogfood_tasks.json` for your questions (or pass `--tasks`).
Results: `results_owner_*.json` / `results_owner_*.md` (gitignored). Not Layer-1 evidence.

## Study lite (10–20 scoped tasks)

```bash
python3 -m wedge_v1 study check --corpus "$OWNER_CORPUS" --tasks wedge_v1/data/owner_tasks/questions-v1.json --dir wedge_v1/.studies/first-use
python3 -m wedge_v1 study run   --corpus "$OWNER_CORPUS" --tasks wedge_v1/data/owner_tasks/questions-v1.json --dir wedge_v1/.studies/first-use
```

Demo (fixtures): same commands with `--corpus wedge_v1/fixtures/owner_corpus`.
