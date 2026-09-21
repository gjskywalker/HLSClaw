#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import os
import re
import sys
from pathlib import Path


DEFAULT_PDF_NAME = "vitis-user-guide.pdf"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "given",
    "hls",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "what",
    "when",
    "with",
}
QUERY_EXPANSIONS = {
    "atax": {"loop", "pipeline", "array", "memory", "latency", "throughput"},
    "bnn": {"loop", "pipeline", "unroll", "array", "memory", "throughput"},
    "bnnkernel": {"loop", "pipeline", "unroll", "array", "memory", "throughput"},
    "cnn": {"loop", "pipeline", "unroll", "array", "dataflow", "throughput"},
    "substring": {"loop", "pipeline", "memory", "dataflow", "latency"},
    "pragma": {"pipeline", "unroll", "dataflow", "array", "interface"},
    "resource": {"area", "utilization", "memory", "bram", "dsp", "lut"},
    "latency": {"throughput", "pipeline", "initiation", "interval"},
}


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


def _load_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _find_repo_root(start: Path) -> Path:
    start_dir = start if start.is_dir() else start.parent
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "HLSClaw").is_dir() and (candidate / "RAG-Anything").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


def _query_terms(question: str) -> set[str]:
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", question)
        if len(token) > 1 and token.lower() not in STOPWORDS
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(QUERY_EXPANSIONS.get(term, set()))
    return expanded


def _content_terms(content: str) -> Counter[str]:
    return Counter(
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", content)
        if len(token) > 1 and token.lower() not in STOPWORDS
    )


def _score_content(content: str, terms: set[str]) -> float:
    if not content or not terms:
        return 0.0
    counts = _content_terms(content)
    score = 0.0
    for term in terms:
        if term in counts:
            score += 2.0 + min(counts[term], 6) * 0.4
    lowered = content.lower()
    for phrase in ("dataflow", "pipeline", "unroll", "array", "memory", "latency", "throughput"):
        if phrase in terms and phrase in lowered:
            score += 1.5
    return score


def _snippet(content: str, terms: set[str], max_chars: int = 900) -> str:
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= max_chars:
        return compact
    lowered = compact.lower()
    hit_positions = [lowered.find(term.lower()) for term in terms if lowered.find(term.lower()) >= 0]
    start = max(min(hit_positions) - 180, 0) if hit_positions else 0
    end = min(start + max_chars, len(compact))
    if start > 0:
        start = compact.find(" ", start)
        if start < 0:
            start = 0
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet = snippet + "..."
    return snippet


def _top_text_chunks(working_dir: Path, terms: set[str], limit: int) -> list[dict[str, object]]:
    chunks = _load_json(working_dir / "kv_store_text_chunks.json", {})
    if not isinstance(chunks, dict):
        return []
    ranked: list[tuple[float, int, str, dict[str, object]]] = []
    for chunk_id, raw in chunks.items():
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content", ""))
        score = _score_content(content, terms)
        # Keep a small deterministic baseline if all scores are weak.
        if score <= 0:
            score = 0.01
        order = int(raw.get("chunk_order_index", 0) or 0)
        ranked.append((score, -order, str(chunk_id), raw))
    ranked.sort(reverse=True)
    top: list[dict[str, object]] = []
    for score, _neg_order, chunk_id, raw in ranked[:limit]:
        content = str(raw.get("content", ""))
        top.append(
            {
                "id": chunk_id,
                "score": round(score, 3),
                "order": raw.get("chunk_order_index", ""),
                "tokens": raw.get("tokens", ""),
                "snippet": _snippet(content, terms),
            }
        )
    return top


def _score_name(name: str, terms: set[str]) -> float:
    pieces = set(
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", name)
        if len(token) > 1 and token.lower() not in STOPWORDS
    )
    return float(len(pieces & terms))


def _top_graph_items(
    working_dir: Path,
    chunk_map: dict[str, dict[str, object]],
    terms: set[str],
    limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entity_store = _load_json(working_dir / "kv_store_entity_chunks.json", {})
    relation_store = _load_json(working_dir / "kv_store_relation_chunks.json", {})
    entities: list[tuple[float, str, dict[str, object]]] = []
    relations: list[tuple[float, str, dict[str, object]]] = []

    if isinstance(entity_store, dict):
        for name, raw in entity_store.items():
            if not isinstance(raw, dict):
                continue
            chunk_ids = [str(item) for item in raw.get("chunk_ids", []) if str(item) in chunk_map]
            score = _score_name(str(name), terms)
            for chunk_id in chunk_ids[:3]:
                score += _score_content(str(chunk_map[chunk_id].get("content", "")), terms) * 0.04
            if score > 0:
                entities.append((score, str(name), raw))

    if isinstance(relation_store, dict):
        for name, raw in relation_store.items():
            if not isinstance(raw, dict):
                continue
            display = str(name).replace("<SEP>", " -> ")
            chunk_ids = [str(item) for item in raw.get("chunk_ids", []) if str(item) in chunk_map]
            score = _score_name(display, terms)
            for chunk_id in chunk_ids[:3]:
                score += _score_content(str(chunk_map[chunk_id].get("content", "")), terms) * 0.04
            if score > 0:
                relations.append((score, display, raw))

    entities.sort(reverse=True)
    relations.sort(reverse=True)
    entity_out = [
        {
            "name": name,
            "score": round(score, 3),
            "chunk_ids": raw.get("chunk_ids", [])[:3] if isinstance(raw.get("chunk_ids", []), list) else [],
        }
        for score, name, raw in entities[:limit]
    ]
    relation_out = [
        {
            "name": name,
            "score": round(score, 3),
            "chunk_ids": raw.get("chunk_ids", [])[:3] if isinstance(raw.get("chunk_ids", []), list) else [],
        }
        for score, name, raw in relations[:limit]
    ]
    return entity_out, relation_out


def _local_fallback_answer(
    working_dir: Path,
    question: str,
    mode: str,
    reason: str,
) -> tuple[str, dict[str, object]]:
    terms = _query_terms(question)
    chunk_map = _load_json(working_dir / "kv_store_text_chunks.json", {})
    if not isinstance(chunk_map, dict):
        chunk_map = {}
    chunks = _top_text_chunks(working_dir, terms, limit=4 if mode == "naive" else 5)
    entities: list[dict[str, object]] = []
    relations: list[dict[str, object]] = []
    if mode != "naive":
        entities, relations = _top_graph_items(working_dir, chunk_map, terms, limit=6)

    lines = [
        "[LOCAL_FALLBACK_RAG]",
        f"Reason: {reason}",
        f"Question: {question}",
        "",
        "Retrieved Vitis HLS guide evidence:",
    ]
    for index, item in enumerate(chunks, start=1):
        lines.append(
            f"{index}. chunk={item['id']} score={item['score']} order={item['order']} "
            f"tokens={item['tokens']}: {item['snippet']}"
        )
    if entities or relations:
        lines.append("")
        lines.append("Knowledge-graph matches from the same library:")
        if entities:
            lines.append("Entities: " + "; ".join(str(item["name"]) for item in entities))
        if relations:
            lines.append("Relations: " + "; ".join(str(item["name"]) for item in relations))
    lines.extend(
        [
            "",
            "Actionable strategy synthesis:",
            "1. Prefer loop pipelining around the dominant compute loop, and target a low II only when loop-carried dependences and memory ports permit it.",
            "2. Use loop unrolling selectively on small inner loops; pair it with array partitioning or reshaping when parallel reads/writes would otherwise serialize.",
            "3. For memory-bound kernels, inspect array and M_AXI access patterns before adding more compute parallelism, because bandwidth and port conflicts can dominate latency.",
            "4. Use DATAFLOW only when the code can be separated into legal producer/consumer stages with clear channels; otherwise keep the transformation local to loops/pragmas.",
        ]
    )
    answer = "\n".join(lines).strip()
    metadata = {
        "fallback": True,
        "fallback_reason": reason,
        "fallback_terms": sorted(terms),
        "fallback_chunks": chunks,
        "fallback_entities": entities,
        "fallback_relations": relations,
    }
    return answer, metadata


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
    repo_root_pdf = hlsclaw_root.parent / DEFAULT_PDF_NAME
    if repo_root_pdf.is_file():
        return repo_root_pdf
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
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Skip API retrieval and query the cached local RAG artifacts directly.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    hlsclaw_root = repo_root / "HLSClaw"
    rag_anything_root = repo_root / "RAG-Anything"
    env_store = hlsclaw_root / ".llm_env"
    force_local = args.force_local or os.getenv("RAG_FORCE_LOCAL", "").lower() in {"1", "true", "yes", "on"}

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

    if not rag_anything_root.is_dir() and not force_local:
        print(f"[ERROR] Missing RAG-Anything: {rag_anything_root}")
        return 2
    if not working_dir.is_dir():
        print(f"[ERROR] Missing RAG library: {working_dir}")
        print("[HINT] Run build_vitis_rag.py first.")
        return 2
    if not manifest_path.is_file():
        print(f"[WARN] Manifest not found: {manifest_path}")

    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"[WARN] Failed to parse manifest {manifest_path}: {exc}")
    else:
        print(f"[WARN] Manifest not found: {manifest_path}")

    local_env = _read_env_file(env_store)
    api_key, base_url, rag_provider = _resolve_endpoint(local_env)

    parser_name = str(manifest.get("parser", "mineru") or "mineru")
    parse_method = str(manifest.get("parse_method", "auto") or "auto")
    llm_model = _normalize_model_for_endpoint(args.llm_model, rag_provider) if rag_provider else args.llm_model
    embed_model = _normalize_model_for_endpoint(args.embed_model, rag_provider) if rag_provider else args.embed_model
    answer = ""
    fallback_metadata: dict[str, object] = {}

    if force_local:
        print("[INFO] RAG_FORCE_LOCAL enabled; using local fallback retrieval.")
        answer, fallback_metadata = _local_fallback_answer(
            working_dir,
            args.question,
            args.mode,
            reason="forced-local",
        )
    elif api_key:
        if str(rag_anything_root) not in sys.path:
            sys.path.insert(0, str(rag_anything_root))
        from rag_common import build_rag_instance

        import asyncio

        async def _query() -> str:
            rag = build_rag_instance(
                working_dir=str(working_dir),
                parser=parser_name,
                parse_method=parse_method,
                llm_model=llm_model,
                embed_model=embed_model,
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
            print("[WARN] Query failed; using local fallback retrieval.")
            print(exc)
            answer, fallback_metadata = _local_fallback_answer(
                working_dir,
                args.question,
                args.mode,
                reason=f"api-query-failed: {type(exc).__name__}",
            )
    else:
        print("[WARN] Missing API key; using local fallback retrieval.")
        answer, fallback_metadata = _local_fallback_answer(
            working_dir,
            args.question,
            args.mode,
            reason="missing-api-key",
        )

    if not answer or "[no-context]" in answer.lower():
        reason = "empty-result" if not answer else "api-returned-no-context"
        print(f"[WARN] RAG query returned {reason}; using local fallback retrieval.")
        answer, fallback_metadata = _local_fallback_answer(
            working_dir,
            args.question,
            args.mode,
            reason=reason,
        )

    retrieval_dir = Path(str(manifest.get("retrieval_dir", lib_root / "retrieval")))
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    out_path = retrieval_dir / "latest_query_result.json"
    out_obj = {
        "question": args.question,
        "mode": args.mode,
        "answer": answer,
        "rag_provider": rag_provider,
        "llm_model": llm_model,
        "embed_model": embed_model,
        "library_root": str(lib_root),
        "manifest_path": str(manifest_path),
        "pdf_path": str(manifest.get("pdf_path", "")),
    }
    out_obj.update(fallback_metadata)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[RAG_RESULT_BEGIN]")
    print(answer)
    print("[RAG_RESULT_END]")
    print(f"[INFO] saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
