#!/bin/sh
# Runs every check that must pass before shipping the Italian pack/site:
#   1. packbuilder check         - pack-level schema/coverage/article checks
#      (engine/tools/packbuilder/qa/check.py with the Italian spec)
#   2. engine/tools/validate_pack.py - engine's schema, referential-integrity,
#      and generated-.js-in-sync checks
#   3. stale-build guard - rebuilds index.html to a scratch file and
#      byte-compares it against the committed one, so a forgotten
#      `./build.sh` after editing the pack or engine is caught here
#      rather than shipping a stale page.
# Usage: ./check.sh   (PACKBUILDER_PATH=<vocab-engine>/tools overrides engine/tools)
set -e
cd "$(dirname "$0")"

echo "== packbuilder check =="
PYTHONPATH="${PACKBUILDER_PATH:-engine/tools}" python3 -m packbuilder check --lang it --repo .

echo
echo "== engine/tools/validate_pack.py =="
python3 engine/tools/validate_pack.py pack

echo
echo "== stale-build guard =="
TMP="$(mktemp /tmp/italian_index_check.XXXXXX.html)"
trap 'rm -f "$TMP"' EXIT
./build.sh "$TMP" > /dev/null
if ! cmp -s "$TMP" index.html; then
  echo "FAIL index.html is stale: rebuild differs from the committed file." >&2
  echo "      Run ./build.sh and commit the result." >&2
  exit 1
fi
echo "OK index.html matches a fresh build ($(wc -c < index.html | tr -d ' ') bytes)"

echo
echo "All checks passed."
