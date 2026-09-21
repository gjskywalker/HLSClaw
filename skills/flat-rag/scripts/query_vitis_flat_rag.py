#!/usr/bin/env python3
"""Query the shared Vitis guide RAG library in flat/vector retrieval mode."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _force_naive_mode(argv: list[str]) -> list[str]:
    rendered: list[str] = []
    skip_next = False
    saw_mode = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--mode":
            saw_mode = True
            rendered.extend(["--mode", "naive"])
            skip_next = index + 1 < len(argv)
            continue
        if arg.startswith("--mode="):
            saw_mode = True
            rendered.append("--mode=naive")
            continue
        rendered.append(arg)
    if not saw_mode:
        rendered.extend(["--mode", "naive"])
    return rendered


def _find_kg_query_script() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "kg-rag" / "scripts" / "query_vitis_rag.py",
        here.parents[3] / "kg-rag" / "scripts" / "query_vitis_rag.py",
    ]
    for ancestor in here.parents:
        candidates.append(ancestor / "HLSClaw" / "skills" / "kg-rag" / "scripts" / "query_vitis_rag.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing KG-RAG query script. Searched:\n  {searched}")


def main() -> None:
    kg_query = _find_kg_query_script()
    sys.argv = [str(kg_query), *_force_naive_mode(sys.argv[1:])]
    runpy.run_path(str(kg_query), run_name="__main__")


if __name__ == "__main__":
    main()
