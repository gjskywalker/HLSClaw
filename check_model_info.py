#!/usr/bin/env python3
"""Quick check for current LLM selection in agentskl."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_local_env(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val.startswith(("'", '"')) and val.endswith(("'", '"')) and len(val) >= 2:
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    _load_local_env(base_dir / ".llm_env")

    cfg_path = base_dir / "config.py"
    spec = importlib.util.spec_from_file_location("agentskl_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from {cfg_path}")

    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    llm = cfg.LLM_CONFIG
    print(f"provider: {llm.get('provider', '<unknown>')}")
    print(f"api_base: {llm.get('api_base', '<unknown>')}")
    print(f"model: {llm.get('model', '<unknown>')}")
    print(f"api_key_set: {'yes' if bool(llm.get('api_key')) else 'no'}")


if __name__ == "__main__":
    main()
