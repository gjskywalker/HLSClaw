#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path


DEFAULT_PDF_NAME = "vitis-user-guide.pdf"


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


def _resolve_key(local_env: dict[str, str]) -> str:
    for key_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if os.getenv(key_name):
            return os.getenv(key_name, "").strip()
        if local_env.get(key_name):
            return local_env[key_name].strip()
    return ""


def _safe_library_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip())
    normalized = normalized.strip(".-")
    return normalized


def _same_path(lhs: Path, rhs: Path) -> bool:
    try:
        return lhs.resolve() == rhs.resolve()
    except OSError:
        return False


def _default_pdf_path(hlsclaw_root: Path) -> Path:
    return (hlsclaw_root / "skills" / "kg-rag" / "references" / DEFAULT_PDF_NAME).resolve()


def _resolve_library_root(
    hlsclaw_root: Path,
    pdf_arg: str,
    library_name_arg: str,
) -> tuple[Path, Path]:
    base_root = hlsclaw_root / "kg-lib"
    if library_name_arg:
        library_name = _safe_library_name(library_name_arg)
        if not library_name:
            raise ValueError("Resolved library name is empty; provide a valid --library-name")
        library_root = base_root / library_name
        return library_root, library_root / "library_manifest.json"

    pdf_path = Path(pdf_arg).expanduser().resolve() if pdf_arg else _default_pdf_path(hlsclaw_root)
    if _same_path(pdf_path, _default_pdf_path(hlsclaw_root)):
        return base_root, base_root / "library_manifest.json"

    library_name = _safe_library_name(pdf_path.stem)
    if not library_name:
        raise ValueError("Failed to derive library name from the PDF stem")
    library_root = base_root / library_name
    return library_root, library_root / "library_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Query a PDF RAG library via RAG-Anything.")
    parser.add_argument("--question", required=True, help="Question for retrieval.")
    parser.add_argument("--mode", default="hybrid", help="Query mode: hybrid/local/global/naive.")
    parser.add_argument(
        "--pdf",
        default="",
        help=(
            "Optional source PDF path used to derive the library location. "
            "Defaults to the legacy Vitis guide library."
        ),
    )
    parser.add_argument(
        "--library-name",
        default="",
        help="Optional explicit library name under HLSClaw/kg-lib.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional direct path to library_manifest.json. Overrides --pdf/--library-name.",
    )
    parser.add_argument("--llm-model", default="openai/gpt-4o-mini", help="Generation model.")
    parser.add_argument("--embed-model", default="openai/text-embedding-3-small", help="Embedding model.")
    parser.add_argument("--embed-dim", type=int, default=1536, help="Embedding dimension.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    hlsclaw_root = Path(__file__).resolve().parents[3]
    rag_anything_root = repo_root / "RAG-Anything"
    env_store = hlsclaw_root / ".llm_env"

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        lib_root = manifest_path.parent
    else:
        lib_root, manifest_path = _resolve_library_root(
            hlsclaw_root=hlsclaw_root,
            pdf_arg=args.pdf,
            library_name_arg=args.library_name,
        )
    working_dir = lib_root / "rag_storage"

    if not rag_anything_root.is_dir():
        print(f"[ERROR] Missing RAG-Anything: {rag_anything_root}")
        return 2
    if not working_dir.is_dir():
        print(f"[ERROR] Missing RAG library: {working_dir}")
        print("[HINT] Run build_vitis_rag.py first.")
        return 2
    if not manifest_path.is_file():
        print(f"[WARN] Manifest not found: {manifest_path}")

    local_env = _read_env_file(env_store)
    api_key = _resolve_key(local_env)
    if not api_key:
        print("[ERROR] Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        return 2

    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"[WARN] Failed to parse manifest {manifest_path}: {exc}")
    else:
        print(f"[WARN] Manifest not found: {manifest_path}")

    if str(rag_anything_root) not in sys.path:
        sys.path.insert(0, str(rag_anything_root))
    from rag_common import build_rag_instance

    base_url = (os.getenv("OPENROUTER_API_BASE") or local_env.get("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1").strip()
    parser_name = str(manifest.get("parser", "mineru") or "mineru")
    parse_method = str(manifest.get("parse_method", "auto") or "auto")

    import asyncio

    async def _query() -> str:
        rag = build_rag_instance(
            working_dir=str(working_dir),
            parser=parser_name,
            parse_method=parse_method,
            llm_model=args.llm_model,
            embed_model=args.embed_model,
            embedding_dim=args.embed_dim,
            api_key=api_key,
            base_url=base_url,
        )
        # aquery() requires rag.lightrag to be initialized; ensure it by using aquery_with_multimodal()
        return await rag.aquery_with_multimodal(
            args.question,
            multimodal_content=None,
            mode=args.mode,
            vlm_enhanced=False,
        )

    try:
        answer = (asyncio.run(_query()) or "").strip()
    except Exception as exc:
        print("[ERROR] Query failed.")
        print(exc)
        return 1

    if not answer:
        print("[WARN] Empty query result.")
        return 1

    retrieval_dir = Path(str(manifest.get("retrieval_dir", lib_root / "retrieval")))
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    out_path = retrieval_dir / "latest_query_result.json"
    out_obj = {
        "question": args.question,
        "mode": args.mode,
        "answer": answer,
        "library_root": str(lib_root),
        "manifest_path": str(manifest_path),
        "pdf_path": str(manifest.get("pdf_path", "")),
    }
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[RAG_RESULT_BEGIN]")
    print(answer)
    print("[RAG_RESULT_END]")
    print(f"[INFO] saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
