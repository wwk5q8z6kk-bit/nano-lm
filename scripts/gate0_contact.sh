#!/usr/bin/env bash
# Gate 0 — owner corpus contact loop (product eval; not Layer-1 evidence).
# Usage:
#   export WEDGE_OWNER_CORPUS=/path/to/your/private/notes   # >=10 docs recommended
#   ./scripts/gate0_contact.sh
# Optional:
#   USEFUL="Saved 10 min finding TTL conflicts" ./scripts/gate0_contact.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
CORPUS="${WEDGE_OWNER_CORPUS:-${OWNER_CORPUS:-}}"

if [[ -z "${CORPUS}" ]]; then
  echo "ERROR: set WEDGE_OWNER_CORPUS or OWNER_CORPUS to your private document folder." >&2
  echo "Example: export WEDGE_OWNER_CORPUS=~/Research/notes" >&2
  exit 2
fi

if [[ ! -d "${CORPUS}" ]]; then
  echo "ERROR: corpus path not found: ${CORPUS}" >&2
  exit 2
fi

echo "== Gate 0 contact loop =="
echo "corpus: ${CORPUS}"
echo

"${PY}" -m wedge_v1 owner-ready --corpus "${CORPUS}"
echo

"${PY}" -m wedge_v1 smoke
echo

"${PY}" -m wedge_v1 owner-dogfood --corpus "${CORPUS}"
echo

"${PY}" -m wedge_v1 gallery --from wedge_v1/results_owner_dogfood.json \
  -o wedge_v1/results_owner_failure_gallery.md
echo

"${PY}" -m wedge_v1 evolve
echo

"${PY}" -m wedge_v1 measure-u wedge_v1/results_owner_dogfood.json --class OWNER_PRIVATE
echo

CONTACT_ARGS=(contact --corpus "${CORPUS}" --class OWNER_PRIVATE)
if [[ -n "${USEFUL:-}" ]]; then
  CONTACT_ARGS+=(--useful "${USEFUL}")
fi
if [[ -n "${NOT_USEFUL:-}" ]]; then
  CONTACT_ARGS+=(--not-useful "${NOT_USEFUL}")
fi
"${PY}" -m wedge_v1 "${CONTACT_ARGS[@]}"
echo

"${PY}" -m wedge_v1 adversarial
echo

"${PY}" -m wedge_v1 habit --corpus "${CORPUS}" --json
echo

echo "== Next (owner) =="
echo "  ${PY} -m wedge_v1 review --corpus \"${CORPUS}\" --interactive"
echo "  ${PY} -m wedge_v1 habit --rerun --corpus \"${CORPUS}\""
echo
echo "Record usefulness when ready:"
echo "  USEFUL=\"one sentence\" ${PY} -m wedge_v1 contact --corpus \"${CORPUS}\" --class OWNER_PRIVATE --useful \"...\""
