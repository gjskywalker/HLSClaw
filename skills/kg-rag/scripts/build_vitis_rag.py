#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PDF_NAME = "vitis-user-guide.pdf"


def _required_rag_artifacts(working_dir: Path) -> list[Path]:
    return [
        working_dir / "graph_chunk_entity_relation.graphml",
        working_dir / "kv_store_full_docs.json",
        working_dir / "kv_store_text_chunks.json",
        working_dir / "vdb_chunks.json",
    ]


def _rag_library_ready(working_dir: Path) -> bool:
    return all(path.is_file() for path in _required_rag_artifacts(working_dir))


def _safe_library_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip())
    normalized = normalized.strip(".-")
    return normalized


def _same_path(lhs: Path, rhs: Path) -> bool:
    try:
        return lhs.resolve() == rhs.resolve()
    except OSError:
        return False


def _default_pdf_path(skill_root: Path) -> Path:
    return (skill_root / "references" / DEFAULT_PDF_NAME).resolve()


def _resolve_pdf_path(skill_root: Path, pdf_arg: str) -> Path:
    if pdf_arg:
        return Path(pdf_arg).expanduser().resolve()
    return _default_pdf_path(skill_root)


def _resolve_library_root(
    hlsclaw_root: Path,
    pdf_path: Path,
    default_pdf_path: Path,
    library_name_arg: str,
) -> tuple[str, Path]:
    base_root = hlsclaw_root / "kg-lib"
    if library_name_arg:
        library_name = _safe_library_name(library_name_arg)
        if not library_name:
            raise ValueError("Resolved library name is empty; provide a valid --library-name")
        return library_name, base_root / library_name
    if _same_path(pdf_path, default_pdf_path):
        return "", base_root
    library_name = _safe_library_name(pdf_path.stem)
    if not library_name:
        raise ValueError("Failed to derive library name from the PDF stem")
    return library_name, base_root / library_name


def _manifest_matches_pdf(manifest: dict[str, object], pdf_path: Path) -> bool:
    manifest_pdf = str(manifest.get("pdf_path", "") or "").strip()
    if not manifest_pdf:
        return False
    return _same_path(Path(manifest_pdf), pdf_path)


def _find_default_cache_path(
    pdf_path: Path,
    output_dir: Path,
    rag_anything_root: Path,
    explicit_cache: str,
) -> Path:
    if explicit_cache:
        return Path(explicit_cache).expanduser().resolve()

    file_name = f"{pdf_path.stem}_content_list.json"
    roots = [
        output_dir / pdf_path.stem,
        rag_anything_root / "output" / pdf_path.stem,
    ]
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.glob(f"**/{file_name}"))
        if matches:
            return matches[0].resolve()

    return (rag_anything_root / "output" / pdf_path.stem / "hybrid_auto" / file_name).resolve()


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PDF RAG library via RAG-Anything.")
    parser.add_argument(
        "--pdf",
        default="",
        help=(
            "Path to the source PDF. Defaults to skills/kg-rag/references/"
            + DEFAULT_PDF_NAME
        ),
    )
    parser.add_argument(
        "--library-name",
        default="",
        help=(
            "Optional library name under HLSClaw/kg-lib. "
            "If omitted, the default Vitis guide keeps the legacy layout at HLSClaw/kg-lib, "
            "while custom PDFs use their sanitized file stem."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild even if manifest exists.")
    parser.add_argument("--parser", default="mineru", help="RAGAnything parser: mineru or docling.")
    parser.add_argument("--parse-method", default="auto", help="Parse method: auto/ocr/txt.")
    parser.add_argument("--llm-model", default="qwen/qwen3-235b-a22b", help="Generation model.")
    parser.add_argument("--embed-model", default="openai/text-embedding-3-small", help="Embedding model.")
    parser.add_argument("--embed-dim", type=int, default=1536, help="Embedding dimension.")
    parser.add_argument("--max-blocks", type=int, default=0, help="For cache fallback: limit text blocks; 0 means all.")
    parser.add_argument(
        "--prefer-cache",
        action="store_true",
        help="Skip parser stage and build directly from cached content_list when available.",
    )
    parser.add_argument(
        "--cached-content-list",
        default="",
        help="Optional path to existing *_content_list.json for parser-failure fallback.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    hlsclaw_root = Path(__file__).resolve().parents[3]
    rag_anything_root = repo_root / "RAG-Anything"
    skill_root = hlsclaw_root / "skills" / "kg-rag"
    default_pdf_path = _default_pdf_path(skill_root)
    pdf_path = _resolve_pdf_path(skill_root, args.pdf)
    library_name, lib_root = _resolve_library_root(
        hlsclaw_root=hlsclaw_root,
        pdf_path=pdf_path,
        default_pdf_path=default_pdf_path,
        library_name_arg=args.library_name,
    )
    working_dir = lib_root / "rag_storage"
    output_dir = lib_root / "output"
    retrieval_dir = lib_root / "retrieval"
    manifest_path = lib_root / "library_manifest.json"
    env_store = hlsclaw_root / ".llm_env"
    cache_script = skill_root / "scripts" / "build_vitis_rag_from_cache.py"
    cache_path = _find_default_cache_path(
        pdf_path=pdf_path,
        output_dir=output_dir,
        rag_anything_root=rag_anything_root,
        explicit_cache=args.cached_content_list,
    )

    if not rag_anything_root.is_dir():
        print(f"[ERROR] Missing RAG-Anything: {rag_anything_root}")
        return 2
    if not pdf_path.is_file():
        print(f"[ERROR] Missing source PDF: {pdf_path}")
        return 2

    lib_root.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in (working_dir, output_dir, retrieval_dir):
            if path.exists() and str(path).startswith(str(lib_root)):
                shutil.rmtree(path, ignore_errors=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_manifest: dict[str, object] = {}
    if manifest_path.is_file() and not args.force:
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            old_pdf_mtime = float(old_manifest.get("pdf_mtime", 0.0))
            if (
                _manifest_matches_pdf(old_manifest, pdf_path)
                and old_pdf_mtime >= pdf_path.stat().st_mtime
                and old_manifest.get("status") == "ready"
                and _rag_library_ready(working_dir)
            ):
                print("[INFO] Existing RAG library is up-to-date. Use --force to rebuild.")
                print(f"[INFO] manifest={manifest_path}")
                return 0
        except (ValueError, OSError, TypeError):
            old_manifest = {}
    elif not args.force and _rag_library_ready(working_dir) and _same_path(pdf_path, default_pdf_path) and not library_name:
        manifest = {
            "status": "ready",
            "source_mode": "existing_library",
            "built_at": int(time.time()),
            "library_name": library_name,
            "library_root": str(lib_root),
            "pdf_path": str(pdf_path),
            "pdf_mtime": pdf_path.stat().st_mtime,
            "working_dir": str(working_dir),
            "output_dir": str(output_dir),
            "retrieval_dir": str(retrieval_dir),
            "parser": args.parser,
            "parse_method": args.parse_method,
            "llm_model": args.llm_model,
            "embed_model": args.embed_model,
            "embed_dim": args.embed_dim,
            "max_blocks": args.max_blocks,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("[INFO] Existing RAG library detected without a fresh manifest. Reusing it.")
        print(f"[INFO] manifest={manifest_path}")
        return 0

    if not args.force and _rag_library_ready(working_dir):
        if old_manifest.get("pdf_path"):
            print("[WARN] Existing RAG artifacts belong to a different PDF. Rebuilding this library root cleanly.")
        else:
            print("[WARN] Existing RAG artifacts found without a trusted manifest. Rebuilding this library root cleanly.")
        for path in (working_dir, output_dir, retrieval_dir):
            if path.exists() and str(path).startswith(str(lib_root)):
                shutil.rmtree(path, ignore_errors=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    local_env = _read_env_file(env_store)
    api_key = _resolve_key(local_env)
    if not api_key:
        print("[ERROR] Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        return 2

    build_cmd = [
        sys.executable,
        str(rag_anything_root / "rag_build.py"),
        "--pdf",
        str(pdf_path),
        "--working-dir",
        str(working_dir),
        "--output-dir",
        str(output_dir),
        "--parser",
        args.parser,
        "--parse-method",
        args.parse_method,
        "--llm-model",
        args.llm_model,
        "--embed-model",
        args.embed_model,
        "--embed-dim",
        str(args.embed_dim),
    ]

    env = os.environ.copy()
    if not env.get("OPENROUTER_API_KEY"):
        env["OPENROUTER_API_KEY"] = api_key
    if local_env.get("OPENROUTER_API_BASE") and not env.get("OPENROUTER_API_BASE"):
        env["OPENROUTER_API_BASE"] = local_env["OPENROUTER_API_BASE"]

    print("[INFO] Building PDF RAG library...")
    print(f"[INFO] pdf_path={pdf_path}")
    print(f"[INFO] library_root={lib_root}")
    print(f"[INFO] working_dir={working_dir}")
    print(f"[INFO] output_dir={output_dir}")
    source_mode = "parse"

    def _run_cache_build() -> int:
        fallback_cmd = [
            sys.executable,
            str(cache_script),
            "--content-list",
            str(cache_path),
            "--working-dir",
            str(working_dir),
            "--source-pdf-name",
            pdf_path.name,
            "--llm-model",
            args.llm_model,
            "--embed-model",
            args.embed_model,
            "--embed-dim",
            str(args.embed_dim),
            "--base-url",
            env.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
            "--max-blocks",
            str(args.max_blocks),
        ]
        fallback = subprocess.run(
            fallback_cmd,
            cwd=str(repo_root),
            env=env,
            text=True,
        )
        return fallback.returncode

    if args.prefer_cache and cache_script.is_file() and cache_path.is_file():
        print("[INFO] --prefer-cache enabled. Building from cached content_list.")
        cache_rc = _run_cache_build()
        if cache_rc != 0:
            print(f"[ERROR] Cache build failed with return code {cache_rc}")
            return cache_rc
        source_mode = "cached_content_list"
    else:
        result = subprocess.run(
            build_cmd,
            cwd=str(rag_anything_root),
            env=env,
            text=True,
        )
        if result.returncode != 0:
            if cache_script.is_file() and cache_path.is_file():
                print("[WARN] Direct parsing failed. Falling back to cached content_list build.")
                cache_rc = _run_cache_build()
                if cache_rc != 0:
                    print(f"[ERROR] Fallback build failed with return code {cache_rc}")
                    return cache_rc
                source_mode = "cached_content_list"
            else:
                print(f"[ERROR] Build failed with return code {result.returncode}")
                return result.returncode

    manifest = {
        "status": "ready",
        "source_mode": source_mode,
        "built_at": int(time.time()),
        "library_name": library_name,
        "library_root": str(lib_root),
        "pdf_path": str(pdf_path),
        "pdf_mtime": pdf_path.stat().st_mtime,
        "working_dir": str(working_dir),
        "output_dir": str(output_dir),
        "retrieval_dir": str(retrieval_dir),
        "parser": args.parser,
        "parse_method": args.parse_method,
        "llm_model": args.llm_model,
        "embed_model": args.embed_model,
        "embed_dim": args.embed_dim,
        "max_blocks": args.max_blocks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] PDF RAG library ready. manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
