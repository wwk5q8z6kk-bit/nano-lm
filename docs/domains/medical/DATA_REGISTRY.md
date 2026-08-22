# Medical Data Registry

## Rules

| Rule | Enforcement |
|------|-------------|
| **No PHI in git** | Owner private corpora stay local; `.gitignore` for runtime outputs |
| **Synthetic / public eval only in repo** | Scribe template data, public benchmarks |
| **Provenance required** | Every dataset documents source, license, generation method |
| **No clinical claims from mock data** | Synthetic success ≠ clinical validation |

## In-repo data (examples)

| Path | Content |
|------|---------|
| `scribe/` | Synthetic dialogue / scribe training & eval fixtures |
| `data/` | External lexicons, clinical termsets (check license per file) |
| `wedge_v1/data/` | Demo/fixture corpora for document intelligence harness |

## Owner private data

```text
WEDGE_OWNER_CORPUS / OWNER_CORPUS — local only
.local-data/ — gitignored stand-ins
```

Never commit private notes, transcripts, or patient information.

## Future external medical eval

Hold-out medical dialogue sets for P1 must be:

- licensed or owner-authorized
- stored outside public repo if sensitive
- referenced by manifest only in git
