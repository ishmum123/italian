#!/usr/bin/env python3
"""Build the Italian pack with the shared builder in vocab-engine.

Equivalent to:
    PYTHONPATH=engine/tools python3 -m packbuilder build --lang it --repo . [--stage ...]

The pipeline lives in engine/tools/packbuilder (core stages) and
engine/tools/packbuilder/langs/it.py (Italian rules). This repo keeps only the
data it owns: tools/gloss_overrides.json, tools/forced_a1.txt,
tools/id_map_v1.json, and the generated pack/ + tools/REPORT.md.

Set PACKBUILDER_PATH to use a vocab-engine checkout other than the engine/
submodule (e.g. PACKBUILDER_PATH=../vocab-engine/tools while developing it).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("PACKBUILDER_PATH") or str(ROOT / "engine" / "tools"))
try:
    from packbuilder.cli import main
except ModuleNotFoundError:
    sys.exit("packbuilder not found in engine/tools: init/update the engine submodule to a vocab-engine "
             "commit that has tools/packbuilder (git submodule update --init --remote engine), "
             "or set PACKBUILDER_PATH to a vocab-engine/tools directory")

if __name__ == "__main__":
    sys.exit(main(["build", "--lang", "it", "--repo", str(ROOT)] + sys.argv[1:]))
