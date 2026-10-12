#!/usr/bin/env bash
# Canonical nanoscribe CI entrypoint — measurement/CI Wave 0 lane.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 -m pytest nanoscribe -q -c pytest.nanoscribe.ini -p no:cacheprovider
python3 -m pytest nanoscribe/test_collection_floor.py -q -c pytest.nanoscribe.ini -p no:cacheprovider
