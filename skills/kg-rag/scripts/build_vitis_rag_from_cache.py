#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from pathlib import Path

def _read_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val.startswith(("'", '"')) and val.endswith(("'", '"')) and len(val) >= 2:
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def _resolve_endpoint(local_env: dict[str, str]) -> tuple[str, str, str]:
    if os.getenv("RAG_API_KEY") or local_env.get("RAG_API_KEY"):
        return (
            (os.getenv("RAG_API_KEY") or local_env.get("RAG_API_KEY", "")).strip(),
            (os.getenv("RAG_API_BASE") or local_env.get("RAG_API_BASE", "https://api.openai.com/v1")).strip(),
            "openai",
        )
    for key_name, base_name, default_base, provider in (
        ("OPENAI_API_KEY", "OPENAI_API_BASE", "https://api.openai.com/v1", "openai"),
        ("OPENROUTER_API_KEY", "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1", "openrouter"),
    ):
        key = (os.getenv(key_name) or local_env.get(key_name, "")).strip()
        if key:
            base_url = (os.getenv(base_name) or local_env.get(base_name, default_base)).strip()
            return key, base_url, provider
    return "", "", ""


def _normalize_model_for_endpoint(model: str, provider: str) -> str:
    if provider == "openai" and model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _find_repo_root(start: Path) -> Path:
    start_dir = start if start.is_dir() else start.parent
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "HLSClaw").is_dir() and (candidate / "RAG-Anything").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


def _load_content_list(path: Path, max_blocks: int) -> list[dict]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid content list JSON: {path}")
    text_only = [item for item in data if isinstance(item, dict) and item.get("type") == "text"]
    if max_blocks > 0:
        text_only = text_only[:max_blocks]
    return text_only


async def _run(args: argparse.Namespace) -> int:
    repo_root = _find_repo_root(Path(__file__).resolve())
    hlsclaw_root = repo_root / "HLSClaw"
    rag_anything_root = repo_root / "RAG-Anything"
    env_store = hlsclaw_root / ".llm_env"
    cache_path = Path(args.content_list).resolve()
    working_dir = Path(args.working_dir).resolve()

    if not cache_path.is_file():
        print(f"[ERROR] Cached content list not found: {cache_path}")
        return 2

    local_env = _read_env_file(env_store)
    api_key, base_url, rag_provider = _resolve_endpoint(local_env)
    if not api_key:
        print("[ERROR] Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        return 2
    llm_model = _normalize_model_for_endpoint(args.llm_model, rag_provider)
    embed_model = _normalize_model_for_endpoint(args.embed_model, rag_provider)

    content_list = _load_content_list(cache_path, args.max_blocks)
    if not content_list:
        print(f"[ERROR] No text content found in: {cache_path}")
        return 2

    if str(rag_anything_root) not in sys.path:
        sys.path.insert(0, str(rag_anything_root))
    from rag_common import build_rag_instance

    rag = build_rag_instance(
        working_dir=str(working_dir),
        parser="mineru",
        parse_method="auto",
        llm_model=llm_model,
        embed_model=embed_model,
        embedding_dim=args.embed_dim,
        api_key=api_key,
        base_url=base_url or args.base_url,
    )
    await rag.insert_content_list(
        content_list=content_list,
        file_path=args.source_pdf_name,
        display_stats=True,
    )
    print(f"[OK] Cache-based RAG build complete. blocks={len(content_list)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Vitis RAG from cached content_list.json.")
    parser.add_argument("--content-list", required=True, help="Path to *_content_list.json.")
    parser.add_argument("--working-dir", required=True, help="RAG working dir.")
    parser.add_argument("--source-pdf-name", default="vitis-user-guide.pdf", help="Source pdf display name.")
    parser.add_argument("--llm-model", default="qwen/qwen3-235b-a22b", help="Generation model.")
    parser.add_argument("--embed-model", default="openai/text-embedding-3-small", help="Embedding model.")
    parser.add_argument("--embed-dim", type=int, default=1536, help="Embedding dimension.")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--max-blocks", type=int, default=0, help="Limit text blocks for faster smoke build; 0 means all.")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
