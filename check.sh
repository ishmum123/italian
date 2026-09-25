#!/bin/sh
# Runs every check that must pass before shipping the Italian pack/site:
#   1. packbuilder check         - pack-level schema/coverage/article checks
#      (engine/tools/packbuilder/qa/check.py with the Italian spec)
#   2. engine/tools/validate_pack.py - engine's schema, referential-integrity,
#      and generated-.js-in-sync checks
#   3. stale-build guard (engine/tools/check_site.sh) - rebuilds index.html and sw.js
#      to a scratch dir and byte-compares them against the committed files, and checks
#      both are tracked by git and committed, so a forgotten `./build.sh` or a page
#      published without its sw.js is caught here rather than shipping stale.
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
sh engine/tools/check_site.sh pack        # [page], default index.html

echo
echo "All checks passed."
