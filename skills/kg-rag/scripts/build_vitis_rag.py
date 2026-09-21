#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PDF_NAME = "vitis-user-guide.pdf"
DOMAIN_ENTITIES = {
    "Array Partition": ("array partition", "partitioning", "array_partition"),
    "Dataflow Pragma": ("dataflow", "#pragma hls dataflow"),
    "HLS Optimization Directives": ("optimization directives", "pragma hls"),
    "Initiation Interval": ("initiation interval", " ii ", " ii="),
    "Interface Pragma": ("interface pragma", "#pragma hls interface", "m_axi"),
    "Latency": ("latency",),
    "Loop Pipelining": ("loop pipelining", "pipelining loops", "#pragma hls pipeline", "pipeline ii"),
    "Loop Unrolling": ("loop unrolling", "#pragma hls unroll", "unroll"),
    "Memory Architecture": ("memory architecture", "memory bandwidth", "memory access", "m_axi"),
    "Resource Utilization": ("resource", "utilization", "dsp", "bram", "lut", "ff"),
    "Task-Level Parallelism": ("task-level parallelism", "producer-consumer", "parallelism"),
    "Throughput": ("throughput",),
    "Vitis HLS": ("vitis hls", "vitis high-level synthesis"),
}
DOMAIN_RELATIONS = (
    ("Loop Pipelining", "Initiation Interval"),
    ("Loop Unrolling", "Resource Utilization"),
    ("Array Partition", "Memory Architecture"),
    ("Dataflow Pragma", "Task-Level Parallelism"),
    ("Interface Pragma", "Memory Architecture"),
    ("HLS Optimization Directives", "Vitis HLS"),
    ("Throughput", "Latency"),
)


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
    repo_root_pdf = skill_root.parents[2] / DEFAULT_PDF_NAME
    if repo_root_pdf.is_file():
        return repo_root_pdf
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


def _find_repo_root(start: Path) -> Path:
    start_dir = start if start.is_dir() else start.parent
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "HLSClaw").is_dir() and (candidate / "RAG-Anything").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


def _load_cached_text_blocks(cache_path: Path, max_blocks: int) -> list[str]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid content list JSON: {cache_path}")
    blocks: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if text:
            blocks.append(text)
        if max_blocks > 0 and len(blocks) >= max_blocks:
            break
    return blocks


def _chunk_words(blocks: list[str], max_words: int = 900) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for block in blocks:
        words = block.split()
        if current and current_words + len(words) > max_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(block)
        current_words += len(words)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _contains_any(content_lower: str, needles: tuple[str, ...]) -> bool:
    padded = f" {content_lower} "
    return any(needle in padded for needle in needles)


def _write_graphml(path: Path, entities: dict[str, list[str]], relations: dict[str, list[str]]) -> None:
    def esc(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <graph edgedefault="undirected">',
    ]
    for name in entities:
        lines.append(f'    <node id="{esc(name)}" />')
    for index, relation_name in enumerate(relations):
        src, tgt = relation_name.split("<SEP>", 1)
        lines.append(f'    <edge id="e{index}" source="{esc(src)}" target="{esc(tgt)}" />')
    lines.extend(["  </graph>", "</graphml>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_local_cached_library(
    cache_path: Path,
    working_dir: Path,
    pdf_path: Path,
    embed_dim: int,
    max_blocks: int,
) -> dict[str, object]:
    blocks = _load_cached_text_blocks(cache_path, max_blocks=max_blocks)
    if not blocks:
        raise ValueError(f"No text blocks found in cached content list: {cache_path}")

    if working_dir.exists():
        shutil.rmtree(working_dir, ignore_errors=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    doc_id = "doc-" + hashlib.md5(str(pdf_path).encode("utf-8")).hexdigest()
    text = "\n\n".join(blocks)
    text_chunks: dict[str, dict[str, object]] = {}
    chunk_texts = _chunk_words(blocks)
    for index, chunk in enumerate(chunk_texts):
        chunk_id = "chunk-" + hashlib.md5(f"{index}:{chunk}".encode("utf-8")).hexdigest()
        text_chunks[chunk_id] = {
            "tokens": len(chunk.split()),
            "content": chunk,
            "chunk_order_index": index,
            "full_doc_id": doc_id,
            "file_path": pdf_path.name,
            "llm_cache_list": [],
            "create_time": int(time.time()),
            "update_time": int(time.time()),
            "_id": chunk_id,
        }

    entity_chunks: dict[str, dict[str, object]] = {}
    entity_chunk_ids: dict[str, list[str]] = {}
    for name, needles in DOMAIN_ENTITIES.items():
        matches = [
            chunk_id
            for chunk_id, raw in text_chunks.items()
            if _contains_any(str(raw["content"]).lower(), needles)
        ]
        if matches:
            entity_chunk_ids[name] = matches
            entity_chunks[name] = {
                "chunk_ids": matches,
                "count": len(matches),
                "create_time": int(time.time()),
                "update_time": int(time.time()),
                "_id": name,
            }

    relation_chunks: dict[str, dict[str, object]] = {}
    for src, tgt in DOMAIN_RELATIONS:
        src_chunks = set(entity_chunk_ids.get(src, []))
        tgt_chunks = set(entity_chunk_ids.get(tgt, []))
        matches = sorted(src_chunks & tgt_chunks) or sorted((src_chunks | tgt_chunks))
        if matches:
            key = f"{src}<SEP>{tgt}"
            relation_chunks[key] = {
                "chunk_ids": matches,
                "count": len(matches),
                "create_time": int(time.time()),
                "update_time": int(time.time()),
                "_id": key,
            }

    stores: dict[str, object] = {
        "kv_store_full_docs.json": {
            doc_id: {
                "content": text,
                "file_path": pdf_path.name,
                "_id": doc_id,
            }
        },
        "kv_store_doc_status.json": {
            doc_id: {
                "status": "processed",
                "content_length": len(text),
                "chunks_count": len(text_chunks),
                "file_path": pdf_path.name,
                "_id": doc_id,
            }
        },
        "kv_store_text_chunks.json": text_chunks,
        "kv_store_entity_chunks.json": entity_chunks,
        "kv_store_relation_chunks.json": relation_chunks,
        "kv_store_full_entities.json": {doc_id: {"entity_names": sorted(entity_chunks), "_id": doc_id}},
        "kv_store_full_relations.json": {
            doc_id: {
                "relation_pairs": [key.split("<SEP>") for key in sorted(relation_chunks)],
                "_id": doc_id,
            }
        },
        "kv_store_llm_response_cache.json": {},
        "vdb_chunks.json": {"embedding_dim": embed_dim, "data": []},
        "vdb_entities.json": {"embedding_dim": embed_dim, "data": []},
        "vdb_relationships.json": {"embedding_dim": embed_dim, "data": []},
    }
    for filename, payload in stores.items():
        (working_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_graphml(working_dir / "graph_chunk_entity_relation.graphml", entity_chunk_ids, relation_chunks)
    return {
        "text_blocks": len(blocks),
        "chunks": len(text_chunks),
        "entities": len(entity_chunks),
        "relations": len(relation_chunks),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PDF RAG library via RAG-Anything.")
    parser.add_argument(
        "--pdf",
        default="",
        help=(
            "Path to the source PDF. Defaults to the repository-level "
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

    repo_root = _find_repo_root(Path(__file__).resolve())
    hlsclaw_root = repo_root / "HLSClaw"
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

    if os.getenv("RAG_FORCE_LOCAL", "").lower() in {"1", "true", "yes", "on"}:
        if not cache_path.is_file():
            print(f"[ERROR] RAG_FORCE_LOCAL requires cached content list: {cache_path}")
            return 2
        try:
            stats = _build_local_cached_library(
                cache_path=cache_path,
                working_dir=working_dir,
                pdf_path=pdf_path,
                embed_dim=args.embed_dim,
                max_blocks=0,
            )
        except Exception as exc:
            print(f"[ERROR] Local cached RAG build failed: {exc}")
            return 1
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "ready",
            "source_mode": "local_cached_content_list",
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
            "rag_provider": "local",
            "llm_model": "local",
            "embed_model": "local",
            "embed_dim": args.embed_dim,
            "requested_max_blocks": args.max_blocks,
            "max_blocks": 0,
            "local_cache_stats": stats,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Local cached RAG library ready. stats={stats} manifest={manifest_path}")
        return 0

    local_env = _read_env_file(env_store)
    api_key, base_url, rag_provider = _resolve_endpoint(local_env)
    if not api_key:
        print("[ERROR] Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        return 2
    llm_model = _normalize_model_for_endpoint(args.llm_model, rag_provider)
    embed_model = _normalize_model_for_endpoint(args.embed_model, rag_provider)

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
        llm_model,
        "--embed-model",
        embed_model,
        "--embed-dim",
        str(args.embed_dim),
        "--api-key",
        api_key,
        "--base-url",
        base_url,
    ]

    env = os.environ.copy()
    if rag_provider == "openai":
        env["OPENAI_API_KEY"] = api_key
        env["OPENROUTER_API_KEY"] = ""
    elif rag_provider == "openrouter":
        env["OPENROUTER_API_KEY"] = api_key
    if base_url:
        if rag_provider == "openai":
            env["OPENAI_API_BASE"] = base_url
        elif rag_provider == "openrouter":
            env["OPENROUTER_API_BASE"] = base_url

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
            llm_model,
            "--embed-model",
            embed_model,
            "--embed-dim",
            str(args.embed_dim),
            "--base-url",
            base_url,
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
        "rag_provider": rag_provider,
        "llm_model": llm_model,
        "embed_model": embed_model,
        "embed_dim": args.embed_dim,
        "max_blocks": args.max_blocks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] PDF RAG library ready. manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
