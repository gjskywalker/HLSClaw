#!/usr/bin/env python3
"""Build the shared Vitis guide RAG library used by the flat-RAG ablation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _find_kg_build_script() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "kg-rag" / "scripts" / "build_vitis_rag.py",
        here.parents[3] / "kg-rag" / "scripts" / "build_vitis_rag.py",
    ]
    for ancestor in here.parents:
        candidates.append(ancestor / "HLSClaw" / "skills" / "kg-rag" / "scripts" / "build_vitis_rag.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing KG-RAG build script. Searched:\n  {searched}")


def main() -> None:
    kg_build = _find_kg_build_script()
    sys.argv = [str(kg_build), *sys.argv[1:]]
    runpy.run_path(str(kg_build), run_name="__main__")


if __name__ == "__main__":
    main()
