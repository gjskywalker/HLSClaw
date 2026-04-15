#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


HLS_PRAGMA_RE = re.compile(r"#\s*pragma\s+HLS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
II_RE = re.compile(r"\bii\s*=\s*(\d+)", re.IGNORECASE)
FACTOR_RE = re.compile(r"\bfactor\s*=\s*(\d+)", re.IGNORECASE)
DEPTH_RE = re.compile(r"\bdepth\s*=\s*(\d+)", re.IGNORECASE)
DEFINE_RE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", re.MULTILINE)
FOR_HEADER_RE = re.compile(r"for\s*\(([^;]*);([^;]*);([^\)]*)\)")
LABEL_RE = re.compile(r"^[A-Za-z_]\w*\s*:\s*$")
M_AXI_PORT_RE = re.compile(
    r"#\s*pragma\s+HLS\s+interface\s+(?:mode\s*=\s*)?m_axi\b[^\n]*\bport\s*=\s*([A-Za-z_]\w*)\b(?:[^\n]*\bbundle\s*=\s*([A-Za-z_]\w*))?",
    re.IGNORECASE,
)
ARRAY_DECL_RE = re.compile(
    r"^\s*(?:static\s+)?(?:const\s+)?(?:[\w:<>]+\s+)*([A-Za-z_]\w*)\s*((?:\[[^\]]+\])+) *;",
)
SUPPORTED_PLAN_PARAMETERS = {"ii", "factor", "depth"}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
INTERFACE_PORT_OPTIONS = {
    "latency": [32],
    "num_read_outstanding": [16],
    "num_write_outstanding": [16],
    "max_read_burst_length": [64],
    "max_write_burst_length": [64],
}
INTERFACE_GLOBAL_OPTIONS = {
    "m_axi_auto_max_ports": ["true"],
    "m_axi_alignment_byte_size": [64],
    "m_axi_max_widen_bitwidth": [512],
}
DATAFLOW_OPTION_DEFAULTS: Dict[str, List[Any]] = {
    "default_channel": ["fifo"],
    "fifo_depth": [4, 8],
    "start_fifo_depth": [8],
    "scalar_fifo_depth": [8],
    "task_level_fifo_depth": [8],
}
OP_IMPL_LATENCY_DEFAULTS: Dict[str, List[tuple[str, int]]] = {
    "mul": [("dsp", 2), ("fabric", 2)],
    "add": [("fabric", 1)],
    "sub": [("fabric", 1)],
    "fmul": [("fulldsp", 4), ("fabric", 4)],
    "fadd": [("fulldsp", 4), ("fabric", 4)],
    "fsub": [("fulldsp", 4), ("fabric", 4)],
    "dmul": [("fulldsp", 8), ("fabric", 8)],
    "dadd": [("fulldsp", 8), ("fabric", 8)],
    "dsub": [("fulldsp", 8), ("fabric", 8)],
}
SUPPORTED_OPS = set(OP_IMPL_LATENCY_DEFAULTS)
SUPPORTED_STORAGE_TYPES = {"ram_1p", "ram_2p", "ram_s2p", "ram_t2p"}
SUPPORTED_STORAGE_IMPLS = {"bram", "lutram", "uram"}


def _priority_key(priority: str) -> int:
    return PRIORITY_ORDER.get(str(priority or "").strip().lower(), 3)


def _clean_priority(priority: str) -> str:
    text = str(priority or "").strip().lower()
    return text if text in PRIORITY_ORDER else ""


def _pragma_kind(line: str) -> Optional[str]:
    if line.lstrip().startswith("//"):
        return None
    match = HLS_PRAGMA_RE.search(line)
    if not match:
        return None
    return match.group(1).lower()


def _split_comment(line: str) -> tuple[str, str]:
    if "//" not in line:
        return line.rstrip(), ""
    body, comment = line.split("//", 1)
    return body.rstrip(), "//" + comment.rstrip()


def _join_comment(body: str, comment: str) -> str:
    return f"{body} {comment}".rstrip() if comment else body.rstrip()


def _extract_int(pattern: re.Pattern[str], text: str) -> Optional[int]:
    match = pattern.search(text)
    if not match:
        return None
    return int(match.group(1))


def _set_keyword_value(line: str, keyword: str, value: int, anchor: str) -> str:
    body, comment = _split_comment(line)
    pattern = re.compile(rf"\b{re.escape(keyword)}\s*=\s*\d+", re.IGNORECASE)
    replacement = f"{keyword}={value}"
    if pattern.search(body):
        body = pattern.sub(replacement, body, count=1)
    else:
        anchor_pattern = re.compile(rf"\b{re.escape(anchor)}\b", re.IGNORECASE)
        match = anchor_pattern.search(body)
        if match:
            insert_at = match.end()
            body = body[:insert_at] + f" {replacement}" + body[insert_at:]
        else:
            body = body.rstrip() + f" {replacement}"
    return _join_comment(body, comment)


def _set_complete_partition_factor(line: str, value: int) -> str:
    body, comment = _split_comment(line)
    if re.search(r"\bcomplete\b", body, re.IGNORECASE):
        body = re.sub(r"\bcomplete\b", f"cyclic factor={value}", body, count=1, flags=re.IGNORECASE)
    elif re.search(r"\b(block|cyclic)\b", body, re.IGNORECASE):
        partition_type = re.search(r"\b(block|cyclic)\b", body, re.IGNORECASE)
        if FACTOR_RE.search(body):
            body = re.sub(r"\bfactor\s*=\s*\d+", f"factor={value}", body, count=1, flags=re.IGNORECASE)
        elif partition_type:
            insert_at = partition_type.end()
            body = body[:insert_at] + f" factor={value}" + body[insert_at:]
    else:
        body = body.rstrip() + f" cyclic factor={value}"
    return _join_comment(body, comment)


def _candidate_values(current: Optional[int], defaults: List[int], minimum: int, maximum: int = 64) -> List[int]:
    values: List[int] = []
    if current is None:
        values.extend(defaults)
    else:
        values.extend([max(minimum, current // 2), current * 2])
        values.extend(defaults)
    unique: List[int] = []
    seen = set()
    for value in values:
        value = max(minimum, min(maximum, int(value)))
        if current is not None and value == current:
            continue
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _load_define_values(code: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for match in DEFINE_RE.finditer(code):
        name = match.group(1)
        expr = match.group(2).strip()
        try:
            values[name] = int(eval(expr, {"__builtins__": {}}, values))
        except Exception:
            continue
    return values


def _infer_tripcount_from_header(header: str, defines: Dict[str, int]) -> Optional[int]:
    match = FOR_HEADER_RE.search(" ".join(header.split()))
    if not match:
        return None
    init = match.group(1).strip()
    cond = match.group(2).strip()
    cond_match = re.match(r"(?:.*\b)?([A-Za-z_]\w*)\s*([<>]=?)\s*(.+)", cond)
    if not cond_match:
        return None
    loop_var = cond_match.group(1)
    op = cond_match.group(2)
    expr = cond_match.group(3).strip()
    try:
        limit = int(eval(expr, {"__builtins__": {}}, defines))
    except Exception:
        return None
    start = 0
    init_match = re.search(rf"\b{re.escape(loop_var)}\s*=\s*(-?\d+)\b", init)
    if init_match:
        start = int(init_match.group(1))
    if op == "<":
        return max(0, limit - start)
    if op == "<=":
        return max(0, (limit - start) + 1)
    return None


def _site_tripcount(code_lines: List[str], site_line: int, defines: Dict[str, int]) -> Optional[int]:
    site_idx = max(0, site_line - 1)
    search_indices = list(range(site_idx, max(-1, site_idx - 4), -1)) + list(range(site_idx + 1, min(len(code_lines), site_idx + 13)))
    visited = set()
    for idx in search_indices:
        if idx in visited or idx < 0 or idx >= len(code_lines):
            continue
        visited.add(idx)
        stripped = code_lines[idx].strip()
        if (
            not stripped
            or stripped.startswith("//")
            or stripped in {"{", "}"}
            or stripped.endswith(";")
            or LABEL_RE.match(stripped)
            or _pragma_kind(stripped) is not None
        ):
            continue
        if "for" in stripped:
            return _infer_tripcount_from_header(stripped, defines)
        if idx < site_idx:
            break
    return None


def _preferred_unroll_defaults(tripcount: Optional[int]) -> List[int]:
    if not tripcount or tripcount <= 1:
        return [2, 4]
    values: List[int] = []
    for candidate in (2, 3, 4, 8, 16):
        if candidate <= tripcount and candidate not in values:
            values.append(candidate)
    if tripcount <= 32 and tripcount not in values:
        values.append(tripcount)
    return values or [2]


def _scan_pragma_sites(code: str, max_pragmas: int) -> List[Dict[str, Any]]:
    sites: List[Dict[str, Any]] = []
    for idx, raw_line in enumerate(code.splitlines(), start=1):
        kind = _pragma_kind(raw_line)
        if not kind:
            continue
        sites.append({"line": idx, "kind": kind, "text": raw_line.rstrip()})
        if len(sites) >= max_pragmas:
            break
    return sites


def _site_tuning_metadata(site: Dict[str, Any]) -> tuple[Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]:
    line = str(site["text"])
    kind = str(site["kind"]).lower()

    if kind == "pipeline":
        return "ii", _extract_int(II_RE, line), "II", "pipeline", None

    if kind == "unroll":
        if re.search(r"\bcomplete\b", line, re.IGNORECASE):
            return None, None, None, None, "UNROLL pragma is complete unroll and has no tunable factor"
        return "factor", _extract_int(FACTOR_RE, line), "factor", "unroll", None

    if kind in {"array_partition", "array_reshape"}:
        return "factor", _extract_int(FACTOR_RE, line), "factor", kind, None

    if kind == "stream":
        return "depth", _extract_int(DEPTH_RE, line), "depth", "stream", None

    if kind == "dataflow":
        return None, None, None, None, "DATAFLOW pragma itself is not parameterized; tune associated STREAM depths instead"

    return None, None, None, None, "Pragma kind is not handled by pragma-tuning"


def _intent_and_risk(kind: str) -> tuple[str, str]:
    kind = kind.lower()
    if kind == "pipeline":
        return "throughput", "medium"
    if kind == "unroll":
        return "latency", "medium"
    if kind in {"array_partition", "array_reshape"}:
        return "memory_bandwidth", "medium"
    if kind == "stream":
        return "dataflow_balance", "low"
    return "generic", "medium"


def _render_variants_for_values(site: Dict[str, Any], parameter_kind: str, values: List[int]) -> List[Dict[str, Any]]:
    line = str(site["text"])
    kind = str(site["kind"]).lower()
    current = None
    keyword = ""
    anchor = ""
    if kind == "pipeline":
        current = _extract_int(II_RE, line)
        keyword = "II"
        anchor = "pipeline"
    elif kind == "unroll":
        current = _extract_int(FACTOR_RE, line)
        keyword = "factor"
        anchor = "unroll"
    elif kind in {"array_partition", "array_reshape"}:
        current = _extract_int(FACTOR_RE, line)
        keyword = "factor"
        anchor = kind
    elif kind == "stream":
        current = _extract_int(DEPTH_RE, line)
        keyword = "depth"
        anchor = "stream"
    else:
        return []

    intent, risk = _intent_and_risk(kind)
    rendered: List[Dict[str, Any]] = []
    seen = set()
    for value in values:
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value <= 0 or int_value == current or int_value in seen:
            continue
        seen.add(int_value)
        if kind in {"array_partition", "array_reshape"} and current is None and re.search(r"\bcomplete\b", line, re.IGNORECASE):
            new_text = _set_complete_partition_factor(line, int_value)
        else:
            new_text = _set_keyword_value(line, keyword, int_value, anchor)
        rendered.append(
            {
                "new_text": new_text,
                "parameter_kind": parameter_kind,
                "parameter_value": int_value,
                "intent": intent,
                "risk": risk,
            }
        )
    return rendered


def _heuristic_variants_for_site(site: Dict[str, Any], tripcount: Optional[int]) -> tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    parameter_kind, current_value, _keyword, _anchor, skip_reason = _site_tuning_metadata(site)
    info: Dict[str, Any] = {"parameter_kind": parameter_kind, "current_value": current_value, "loop_tripcount": tripcount}
    if not parameter_kind:
        return [], skip_reason or "No tunable parameters detected", info

    defaults_map = {
        "pipeline": [1, 2],
        "unroll": _preferred_unroll_defaults(tripcount),
        "array_partition": [2, 4],
        "array_reshape": [2, 4],
        "stream": [2, 4, 8],
    }
    minimum_map = {
        "ii": 1,
        "factor": 2,
        "depth": 2,
    }
    kind = str(site["kind"]).lower()
    values = _candidate_values(
        current=current_value,
        defaults=defaults_map.get(kind, [2, 4]),
        minimum=minimum_map.get(parameter_kind, 1),
        maximum=256 if parameter_kind == "depth" else (tripcount if kind == "unroll" and tripcount else 64),
    )
    variants = _render_variants_for_values(site, parameter_kind, values)
    if kind == "unroll" and current_value is None and tripcount:
        variants = [variant for variant in variants if int(variant["parameter_value"]) < tripcount]
        if not variants:
            skip_reason = "UNROLL pragma is already fully unrolled for this loop; no smaller useful factor remained"
    return variants, None if variants else skip_reason or "No alternative parameter values to try", info


def _load_plan(plan_path: Optional[Path]) -> Dict[str, Any]:
    if not plan_path or not plan_path.is_file():
        return {}
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _normalize_pragma_plan_targets(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_targets = plan.get("pragma_targets", plan.get("tuning_targets", []))
    if not isinstance(raw_targets, list):
        return []
    merged: Dict[tuple[int, str], Dict[str, Any]] = {}
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if not kind:
            continue
        try:
            line = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        key = (line, kind)
        target = merged.get(key, {"line": line, "kind": kind, "try_values": []})
        parameter = str(item.get("parameter") or item.get("parameter_kind") or "").strip().lower()
        chosen_parameter = str(target.get("parameter", "")).strip().lower()
        accepted_parameter = False
        if parameter and not chosen_parameter:
            target["parameter"] = parameter
            chosen_parameter = parameter
            accepted_parameter = True
        elif parameter and chosen_parameter == parameter:
            accepted_parameter = True
        elif parameter and chosen_parameter not in SUPPORTED_PLAN_PARAMETERS and parameter in SUPPORTED_PLAN_PARAMETERS:
            target["parameter"] = parameter
            chosen_parameter = parameter
            accepted_parameter = True
        elif parameter and parameter != chosen_parameter:
            ignored = list(target.get("ignored_parameters", []))
            if parameter not in ignored:
                ignored.append(parameter)
            target["ignored_parameters"] = ignored
        priority = _clean_priority(str(item.get("priority", "")))
        if priority:
            current_priority = _clean_priority(str(target.get("priority", "")))
            if not current_priority or _priority_key(priority) < _priority_key(current_priority):
                target["priority"] = priority
        reason = str(item.get("reason", "")).strip()
        if reason and (accepted_parameter or not str(target.get("reason", "")).strip()):
            target["reason"] = reason
        existing_values = list(target.get("try_values", []))
        for value in item.get("try_values", []) or []:
            if parameter and not accepted_parameter:
                continue
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                continue
            if int_value <= 0 or int_value in existing_values:
                continue
            existing_values.append(int_value)
        if existing_values:
            target["try_values"] = existing_values
        merged[key] = target
    prioritized = list(merged.values())
    prioritized.sort(key=lambda item: (_priority_key(str(item.get("priority", ""))), item["line"], item["kind"]))
    return prioritized


def _normalize_interface_plan_targets(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_targets = plan.get("interface_targets", [])
    if not isinstance(raw_targets, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        option = str(item.get("option", "")).strip().lower().lstrip("-")
        port = str(item.get("port", "")).strip()
        if not option:
            continue
        target: Dict[str, Any] = {"option": option}
        if port:
            target["port"] = port
        try_values: List[Any] = []
        for value in item.get("try_values", []) or []:
            if option in INTERFACE_GLOBAL_OPTIONS and isinstance(value, str):
                text = value.strip().lower()
                if text and text not in try_values:
                    try_values.append(text)
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed <= 0 or parsed in try_values:
                continue
            try_values.append(parsed)
        if try_values:
            target["try_values"] = try_values
        priority = _clean_priority(str(item.get("priority", "")))
        if priority:
            target["priority"] = priority
        reason = str(item.get("reason", "")).strip()
        if reason:
            target["reason"] = reason
        normalized.append(target)
    normalized.sort(key=lambda item: (_priority_key(str(item.get("priority", ""))), str(item.get("port", "")), item["option"]))
    return normalized


def _normalize_storage_plan_targets(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_targets = plan.get("storage_targets", [])
    if not isinstance(raw_targets, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        variable = str(item.get("variable", "")).strip()
        if not variable:
            continue
        target: Dict[str, Any] = {"variable": variable}
        bindings: List[Dict[str, Any]] = []
        for binding in item.get("bindings", []) or []:
            if not isinstance(binding, dict):
                continue
            storage_type = str(binding.get("type", "")).strip().lower()
            impl = str(binding.get("impl", "")).strip().lower()
            entry: Dict[str, Any] = {}
            if storage_type:
                entry["type"] = storage_type
            if impl:
                entry["impl"] = impl
            try:
                latency = int(binding.get("latency"))
            except (TypeError, ValueError):
                latency = 0
            if latency > 0:
                entry["latency"] = latency
            if entry:
                bindings.append(entry)
        if bindings:
            target["bindings"] = bindings
        priority = _clean_priority(str(item.get("priority", "")))
        if priority:
            target["priority"] = priority
        reason = str(item.get("reason", "")).strip()
        if reason:
            target["reason"] = reason
        normalized.append(target)
    normalized.sort(key=lambda item: (_priority_key(str(item.get("priority", ""))), item["variable"]))
    return normalized


def _normalize_op_plan_targets(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_targets = plan.get("op_targets", [])
    if not isinstance(raw_targets, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "")).strip().lower()
        if not op:
            continue
        target: Dict[str, Any] = {"op": op}
        impls: List[str] = []
        for value in item.get("try_impls", []) or []:
            text = str(value).strip().lower()
            if text and text not in impls:
                impls.append(text)
        if impls:
            target["try_impls"] = impls
        latencies: List[int] = []
        for value in item.get("latency_values", []) or []:
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                continue
            if int_value < 0 or int_value in latencies:
                continue
            latencies.append(int_value)
        if latencies:
            target["latency_values"] = latencies
        priority = _clean_priority(str(item.get("priority", "")))
        if priority:
            target["priority"] = priority
        reason = str(item.get("reason", "")).strip()
        if reason:
            target["reason"] = reason
        normalized.append(target)
    normalized.sort(key=lambda item: (_priority_key(str(item.get("priority", ""))), item["op"]))
    return normalized


def _normalize_dataflow_plan_targets(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_targets = plan.get("dataflow_targets", [])
    if not isinstance(raw_targets, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        option = str(item.get("option", "")).strip().lower().lstrip("-")
        if not option:
            continue
        target: Dict[str, Any] = {"option": option}
        values: List[Any] = []
        for value in item.get("try_values", []) or []:
            if option == "default_channel":
                text = str(value).strip().lower()
                if text and text not in values:
                    values.append(text)
                continue
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                continue
            if int_value <= 0 or int_value in values:
                continue
            values.append(int_value)
        if values:
            target["try_values"] = values
        priority = _clean_priority(str(item.get("priority", "")))
        if priority:
            target["priority"] = priority
        reason = str(item.get("reason", "")).strip()
        if reason:
            target["reason"] = reason
        normalized.append(target)
    normalized.sort(key=lambda item: (_priority_key(str(item.get("priority", ""))), item["option"]))
    return normalized


def _target_parameter(target: Dict[str, Any]) -> str:
    return str(target.get("parameter", "")).strip().lower()


def _site_supports_target(site: Dict[str, Any], target: Dict[str, Any]) -> bool:
    parameter_kind, _current_value, _keyword, _anchor, _skip_reason = _site_tuning_metadata(site)
    target_parameter = _target_parameter(target)
    if not parameter_kind:
        return False
    if target_parameter and target_parameter in SUPPORTED_PLAN_PARAMETERS and target_parameter != parameter_kind:
        return False
    return True


def _match_pragma_plan_target(
    target: Dict[str, Any],
    sites: List[Dict[str, Any]],
    used_sites: set[tuple[int, str]],
) -> tuple[Optional[Dict[str, Any]], str]:
    kind = str(target["kind"]).lower()
    target_parameter = _target_parameter(target)
    exact_key = (int(target["line"]), kind)

    exact_site = next(
        (
            site
            for site in sites
            if (int(site["line"]), str(site["kind"]).lower()) == exact_key
            and exact_key not in used_sites
            and _site_supports_target(site, target)
        ),
        None,
    )
    if exact_site is not None:
        return exact_site, ""

    same_kind = [
        site
        for site in sites
        if str(site["kind"]).lower() == kind and (int(site["line"]), str(site["kind"]).lower()) not in used_sites
    ]
    if not same_kind:
        return None, "No pragma site of the requested kind was found in the hardware rewrite output"

    supported_sites = [site for site in same_kind if _site_supports_target(site, target)]
    if not supported_sites:
        if target_parameter and target_parameter not in SUPPORTED_PLAN_PARAMETERS:
            return None, f"Parameter '{target_parameter}' is not supported by pragma-tuning"
        return None, "No tunable pragma site of the requested kind supports the requested parameter"

    best_site = min(
        supported_sites,
        key=lambda site: (abs(int(site["line"]) - int(target["line"])), 0 if int(site["line"]) >= int(target["line"]) else 1, int(site["line"])),
    )
    return best_site, ""


def _scan_m_axi_ports(code: str) -> List[Dict[str, Any]]:
    ports: List[Dict[str, Any]] = []
    seen = set()
    for idx, line in enumerate(code.splitlines(), start=1):
        match = M_AXI_PORT_RE.search(line)
        if not match:
            continue
        port = match.group(1)
        bundle = match.group(2) or ""
        key = (port, bundle)
        if key in seen:
            continue
        seen.add(key)
        ports.append({"line": idx, "port": port, "bundle": bundle, "text": line.rstrip()})
    return ports


def _array_dim_product(dim_text: str) -> int:
    size = 1
    for raw_part in re.findall(r"\[([^\]]+)\]", dim_text):
        part = raw_part.strip()
        try:
            value = int(part)
        except (TypeError, ValueError):
            return 0
        if value <= 0:
            return 0
        size *= value
    return size


def _scan_local_arrays(code: str, max_arrays: int = 8) -> List[Dict[str, Any]]:
    arrays: List[Dict[str, Any]] = []
    for idx, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#pragma"):
            continue
        match = ARRAY_DECL_RE.match(line)
        if not match:
            continue
        variable = match.group(1)
        dims_text = match.group(2)
        total_elems = _array_dim_product(dims_text)
        arrays.append(
            {
                "line": idx,
                "variable": variable,
                "total_elems": total_elems,
                "text": line.rstrip(),
            }
        )
        if len(arrays) >= max_arrays:
            break
    return arrays


def _source_uses_dataflow(code: str) -> bool:
    return re.search(r"#\s*pragma\s+HLS\s+dataflow\b", code, re.IGNORECASE) is not None


def _code_without_comments(code: str) -> str:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#pragma"):
            continue
        lines.append(line.split("//", 1)[0])
    return "\n".join(lines)


def _detect_op_targets_from_code(code: str) -> List[str]:
    cleaned = _code_without_comments(code)
    has_double = re.search(r"\bdouble\b", cleaned) is not None
    has_float = has_double or re.search(r"\bfloat\b", cleaned) is not None
    has_mul = "*" in cleaned
    has_add = re.search(r"(?<!\+)\+(?!\+)|\+=", cleaned) is not None
    has_sub = re.search(r"(?<!-)\-(?!-)|\-=", cleaned) is not None
    ops: List[str] = []
    if has_double:
        if has_mul:
            ops.append("dmul")
        if has_add:
            ops.append("dadd")
        if has_sub:
            ops.append("dsub")
        return ops
    if has_float:
        if has_mul:
            ops.append("fmul")
        if has_add:
            ops.append("fadd")
        if has_sub:
            ops.append("fsub")
        return ops
    if has_mul:
        ops.append("mul")
    if has_add:
        ops.append("add")
    if has_sub:
        ops.append("sub")
    return ops


def _empty_tcl_overrides() -> Dict[str, Any]:
    return {
        "config_compile": {},
        "config_interface": {},
        "config_dataflow": {},
        "config_rtl": {},
        "config_schedule": {},
        "config_op": [],
    }


def _make_candidate(
    *,
    family: str,
    kind: str,
    target_key: str,
    target_line: int,
    parameter_kind: str = "",
    parameter_value: Any = None,
    source_pragma: str = "",
    tuned_pragma: str = "",
    directives: Optional[List[str]] = None,
    edits: Optional[List[Dict[str, Any]]] = None,
    tcl_directives: Optional[List[str]] = None,
    tcl_overrides: Optional[Dict[str, Any]] = None,
    intent: str = "generic",
    risk: str = "medium",
    selection_source: str = "heuristic_fallback",
    priority: str = "",
    reason: str = "",
    conflict_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    candidate = {
        "family": family,
        "kind": kind,
        "target_key": target_key,
        "target_line": int(target_line or 0),
        "parameter_kind": parameter_kind,
        "parameter_value": parameter_value,
        "source_pragma": source_pragma,
        "tuned_pragma": tuned_pragma,
        "directives": list(directives or []),
        "edits": list(edits or []),
        "tcl_directives": list(tcl_directives or []),
        "tcl_overrides": _empty_tcl_overrides(),
        "intent": intent,
        "risk": risk,
        "selection_source": selection_source,
        "site_keys": [[family, target_key]],
        "conflict_keys": list(conflict_keys or [f"{family}:{target_key}"]),
    }
    if tcl_overrides:
        for key, value in tcl_overrides.items():
            if key == "config_op":
                candidate["tcl_overrides"]["config_op"] = [str(item) for item in list(value or [])]
            elif isinstance(value, dict):
                candidate["tcl_overrides"][key] = {str(k): v for k, v in value.items()}
    priority = _clean_priority(priority)
    if priority:
        candidate["priority"] = priority
    if reason:
        candidate["reason"] = reason
    return candidate


def _build_pragma_candidates_for_selected_sites(
    selected_sites: List[Dict[str, Any]],
    site_overrides: Dict[tuple[int, str], Dict[str, Any]],
    source: str,
    code_lines: List[str],
    define_values: Dict[str, int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    site_summaries: List[Dict[str, Any]] = []
    skipped_pragmas: List[Dict[str, Any]] = []

    for site in selected_sites:
        site_key = (int(site["line"]), str(site["kind"]).lower())
        override = site_overrides.get(site_key, {})
        tripcount = _site_tripcount(code_lines, int(site["line"]), define_values)
        parameter_kind, current_value, _keyword, _anchor, skip_reason = _site_tuning_metadata(site)
        info: Dict[str, Any] = {"parameter_kind": parameter_kind, "current_value": current_value, "loop_tripcount": tripcount}
        if not parameter_kind:
            variants: List[Dict[str, Any]] = []
        elif override.get("try_values"):
            override_values = []
            for value in list(override.get("try_values", [])):
                try:
                    int_value = int(value)
                except (TypeError, ValueError):
                    continue
                if tripcount and str(site["kind"]).lower() == "unroll" and int_value > tripcount:
                    continue
                override_values.append(int_value)
            heuristic_variants, _, heuristic_info = _heuristic_variants_for_site(site, tripcount)
            info.update(heuristic_info)
            heuristic_values = [variant["parameter_value"] for variant in heuristic_variants]
            merged_values = []
            for value in override_values + heuristic_values:
                if value not in merged_values:
                    merged_values.append(value)
            variants = _render_variants_for_values(site, parameter_kind, merged_values)
            if not variants:
                variants, skip_reason, info = _heuristic_variants_for_site(site, tripcount)
        else:
            variants, skip_reason, info = _heuristic_variants_for_site(site, tripcount)

        summary = {
            "line": site["line"],
            "kind": site["kind"],
            "text": site["text"],
            "parameter_kind": info.get("parameter_kind"),
            "current_value": info.get("current_value"),
            "loop_tripcount": info.get("loop_tripcount"),
            "generated_variants": [variant["new_text"] for variant in variants],
            "selection_source": source,
        }
        if override.get("priority"):
            summary["priority"] = override["priority"]
        if override.get("reason"):
            summary["reason"] = override["reason"]
        site_summaries.append(summary)

        if not variants:
            skipped = {
                "line": site["line"],
                "kind": site["kind"],
                "text": site["text"],
                "reason": skip_reason or "No tunable parameters detected",
            }
            if override.get("reason"):
                skipped["plan_reason"] = override["reason"]
            skipped_pragmas.append(skipped)
            continue

        for variant in variants:
            candidate = _make_candidate(
                family="pragma",
                kind=str(site["kind"]),
                target_key=f"{site['line']}:{site['kind']}",
                target_line=int(site["line"]),
                parameter_kind=str(variant["parameter_kind"]),
                parameter_value=variant["parameter_value"],
                source_pragma=str(site["text"]),
                tuned_pragma=str(variant["new_text"]),
                directives=[str(variant["new_text"])],
                edits=[
                    {
                        "type": "replace_line",
                        "line": site["line"],
                        "old_text": site["text"],
                        "new_text": variant["new_text"],
                    }
                ],
                intent=str(variant["intent"]),
                risk=str(variant["risk"]),
                selection_source=source,
                priority=str(override.get("priority", "")),
                reason=str(override.get("reason", "")),
            )
            candidates.append(candidate)

    return candidates, site_summaries, skipped_pragmas


def _build_pragma_candidates(
    *,
    plan: Dict[str, Any],
    sites: List[Dict[str, Any]],
    code_lines: List[str],
    define_values: Dict[str, int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    normalized_targets = _normalize_pragma_plan_targets(plan)
    selected_sites: List[Dict[str, Any]] = []
    site_overrides: Dict[tuple[int, str], Dict[str, Any]] = {}
    unmatched_plan_targets: List[Dict[str, Any]] = []
    selection_mode = "heuristic_fallback"

    if normalized_targets:
        used_sites: set[tuple[int, str]] = set()
        for target in normalized_targets:
            site, reason = _match_pragma_plan_target(target, sites, used_sites)
            if not site:
                unmatched_plan_targets.append(
                    {
                        "line": target["line"],
                        "kind": target["kind"],
                        "parameter": _target_parameter(target),
                        "reason": reason or "No matching pragma site was found in the hardware rewrite output",
                    }
                )
                continue
            key = (int(site["line"]), str(site["kind"]).lower())
            selected_sites.append(site)
            site_overrides[key] = target
            used_sites.add(key)
        if selected_sites:
            selection_mode = "plan"

    if not selected_sites:
        selected_sites = sites
        site_overrides = {}

    candidates, site_summaries, skipped_pragmas = _build_pragma_candidates_for_selected_sites(
        selected_sites=selected_sites,
        site_overrides=site_overrides,
        source=selection_mode,
        code_lines=code_lines,
        define_values=define_values,
    )
    return candidates, site_summaries, skipped_pragmas, unmatched_plan_targets, selection_mode


def _build_interface_candidates(
    *,
    code: str,
    plan: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    ports = _scan_m_axi_ports(code)
    candidates: List[Dict[str, Any]] = []
    site_summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    selection_mode = "none"
    if not ports:
        return candidates, site_summaries, skipped, selection_mode

    plan_targets = _normalize_interface_plan_targets(plan)
    if plan_targets:
        selection_mode = "plan"
        for target in plan_targets:
            option = target["option"]
            port_name = str(target.get("port", "")).strip()
            values = list(target.get("try_values", [])) or list(INTERFACE_PORT_OPTIONS.get(option, INTERFACE_GLOBAL_OPTIONS.get(option, [])))
            if port_name:
                match = next((item for item in ports if item["port"] == port_name), None)
                if not match:
                    skipped.append({"port": port_name, "kind": "interface", "reason": "No matching explicit m_axi port exists in the hardware rewrite output"})
                    continue
                rendered = []
                for value in values:
                    rendered.append(
                        _make_candidate(
                            family="interface",
                            kind="set_directive_interface",
                            target_key=f"{port_name}:{option}",
                            target_line=int(match["line"]),
                            parameter_kind=option,
                            parameter_value=value,
                            tcl_directives=[f"set_directive_interface -mode m_axi -{option} {value} \"{{top_func}}\" {port_name}"],
                            intent="memory_interface",
                            risk="medium",
                            selection_source="plan",
                            priority=str(target.get("priority", "")),
                            reason=str(target.get("reason", "")),
                            conflict_keys=[f"interface:{port_name}:{option}"],
                        )
                    )
                site_summaries.append(
                    {
                        "line": match["line"],
                        "port": port_name,
                        "kind": "m_axi",
                        "option": option,
                        "generated_variants": [candidate["tcl_directives"][0] for candidate in rendered],
                        "selection_source": "plan",
                        "reason": target.get("reason", ""),
                    }
                )
                candidates.extend(rendered)
                continue

            if option not in INTERFACE_GLOBAL_OPTIONS:
                skipped.append({"kind": "interface", "option": option, "reason": "Option requires a concrete m_axi port in the current DSE layer"})
                continue
            rendered = []
            for value in values:
                rendered.append(
                    _make_candidate(
                        family="interface",
                        kind="config_interface",
                        target_key=f"global:{option}",
                        target_line=0,
                        parameter_kind=option,
                        parameter_value=value,
                        tcl_overrides={"config_interface": {f"-{option}": value}},
                        intent="memory_interface",
                        risk="medium",
                        selection_source="plan",
                        priority=str(target.get("priority", "")),
                        reason=str(target.get("reason", "")),
                        conflict_keys=[f"interface:global:{option}"],
                    )
                )
            site_summaries.append(
                {
                    "line": 0,
                    "port": "",
                    "kind": "m_axi_global",
                    "option": option,
                    "generated_variants": [candidate["tcl_overrides"]["config_interface"] for candidate in rendered],
                    "selection_source": "plan",
                    "reason": target.get("reason", ""),
                }
            )
            candidates.extend(rendered)
        return candidates, site_summaries, skipped, selection_mode

    selection_mode = "heuristic_fallback"
    for port in ports[:2]:
        generated: List[str] = []
        for option, values in INTERFACE_PORT_OPTIONS.items():
            for value in values:
                candidate = _make_candidate(
                    family="interface",
                    kind="set_directive_interface",
                    target_key=f"{port['port']}:{option}",
                    target_line=int(port["line"]),
                    parameter_kind=option,
                    parameter_value=value,
                    tcl_directives=[f"set_directive_interface -mode m_axi -{option} {value} \"{{top_func}}\" {port['port']}"],
                    intent="memory_interface",
                    risk="medium",
                    selection_source=selection_mode,
                    conflict_keys=[f"interface:{port['port']}:{option}"],
                )
                candidates.append(candidate)
                generated.append(candidate["tcl_directives"][0])
        site_summaries.append(
            {
                "line": port["line"],
                "port": port["port"],
                "kind": "m_axi",
                "generated_variants": generated,
                "selection_source": selection_mode,
            }
        )
    return candidates, site_summaries, skipped, selection_mode


def _default_storage_bindings(array: Dict[str, Any]) -> List[Dict[str, Any]]:
    bindings = [{"type": "ram_2p", "impl": "bram"}, {"type": "ram_t2p", "impl": "bram"}]
    total_elems = int(array.get("total_elems") or 0)
    if total_elems and total_elems <= 256:
        bindings.append({"type": "ram_2p", "impl": "lutram"})
    if total_elems and total_elems >= 2048:
        bindings.append({"type": "ram_2p", "impl": "uram"})
    unique: List[Dict[str, Any]] = []
    seen = set()
    for binding in bindings:
        key = (binding.get("type", ""), binding.get("impl", ""), binding.get("latency", 0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(binding)
    return unique


def _build_storage_candidates(
    *,
    code: str,
    plan: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    arrays = _scan_local_arrays(code)
    candidates: List[Dict[str, Any]] = []
    site_summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    selection_mode = "none"
    if not arrays:
        return candidates, site_summaries, skipped, selection_mode

    plan_targets = _normalize_storage_plan_targets(plan)
    if plan_targets:
        selection_mode = "plan"
        for target in plan_targets:
            variable = target["variable"]
            array = next((item for item in arrays if item["variable"] == variable), None)
            if not array:
                skipped.append({"variable": variable, "kind": "bind_storage", "reason": "No matching local array declaration exists in the hardware rewrite output"})
                continue
            bindings = list(target.get("bindings", [])) or _default_storage_bindings(array)
            rendered: List[str] = []
            for binding in bindings:
                storage_type = str(binding.get("type", "")).strip().lower()
                impl = str(binding.get("impl", "")).strip().lower()
                if storage_type not in SUPPORTED_STORAGE_TYPES or impl not in SUPPORTED_STORAGE_IMPLS:
                    continue
                latency = binding.get("latency")
                latency_suffix = f" -latency {latency}" if latency else ""
                command = f"set_directive_bind_storage -type {storage_type} -impl {impl}{latency_suffix} \"{{top_func}}\" {variable}"
                candidates.append(
                    _make_candidate(
                        family="storage",
                        kind="bind_storage",
                        target_key=variable,
                        target_line=int(array["line"]),
                        parameter_kind=storage_type,
                        parameter_value=impl,
                        tcl_directives=[command],
                        intent="ii_relief",
                        risk="medium",
                        selection_source="plan",
                        priority=str(target.get("priority", "")),
                        reason=str(target.get("reason", "")),
                        conflict_keys=[f"storage:{variable}"],
                    )
                )
                rendered.append(command)
            site_summaries.append(
                {
                    "line": array["line"],
                    "variable": variable,
                    "kind": "bind_storage",
                    "generated_variants": rendered,
                    "selection_source": "plan",
                    "reason": target.get("reason", ""),
                }
            )
        return candidates, site_summaries, skipped, selection_mode

    selection_mode = "heuristic_fallback"
    for array in arrays[:4]:
        rendered: List[str] = []
        for binding in _default_storage_bindings(array):
            command = f"set_directive_bind_storage -type {binding['type']} -impl {binding['impl']} \"{{top_func}}\" {array['variable']}"
            candidates.append(
                _make_candidate(
                    family="storage",
                    kind="bind_storage",
                    target_key=array["variable"],
                    target_line=int(array["line"]),
                    parameter_kind=str(binding["type"]),
                    parameter_value=str(binding["impl"]),
                    tcl_directives=[command],
                    intent="ii_relief",
                    risk="medium",
                    selection_source=selection_mode,
                    conflict_keys=[f"storage:{array['variable']}"],
                )
            )
            rendered.append(command)
        site_summaries.append(
            {
                "line": array["line"],
                "variable": array["variable"],
                "kind": "bind_storage",
                "generated_variants": rendered,
                "selection_source": selection_mode,
            }
        )
    return candidates, site_summaries, skipped, selection_mode


def _build_op_candidates(
    *,
    code: str,
    plan: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    candidates: List[Dict[str, Any]] = []
    site_summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    selection_mode = "none"
    detected_ops = _detect_op_targets_from_code(code)
    if not detected_ops:
        return candidates, site_summaries, skipped, selection_mode

    plan_targets = _normalize_op_plan_targets(plan)
    targets = plan_targets if plan_targets else [{"op": op} for op in detected_ops]
    selection_mode = "plan" if plan_targets else "heuristic_fallback"
    for target in targets:
        op = str(target["op"]).lower()
        if op not in SUPPORTED_OPS:
            skipped.append({"kind": "config_op", "op": op, "reason": "Operation is not supported by the current DSE layer"})
            continue
        default_pairs = OP_IMPL_LATENCY_DEFAULTS.get(op, [])
        try_impls = list(target.get("try_impls", [])) or [impl for impl, _lat in default_pairs]
        latency_values = list(target.get("latency_values", []))
        rendered: List[str] = []
        for impl, default_latency in default_pairs:
            if try_impls and impl not in try_impls:
                continue
            latencies = latency_values or [default_latency]
            for latency in latencies:
                command = f"config_op {op} -impl {impl} -latency {latency}"
                candidates.append(
                    _make_candidate(
                        family="op_binding",
                        kind="config_op",
                        target_key=op,
                        target_line=0,
                        parameter_kind=op,
                        parameter_value=impl,
                        tcl_overrides={"config_op": [command]},
                        intent="operator_binding",
                        risk="medium",
                        selection_source=selection_mode,
                        priority=str(target.get("priority", "")),
                        reason=str(target.get("reason", "")),
                        conflict_keys=[f"op:{op}"],
                    )
                )
                rendered.append(command)
        if rendered:
            site_summaries.append(
                {
                    "line": 0,
                    "kind": "config_op",
                    "op": op,
                    "generated_variants": rendered,
                    "selection_source": selection_mode,
                    "reason": target.get("reason", ""),
                }
            )
    return candidates, site_summaries, skipped, selection_mode


def _build_dataflow_candidates(
    *,
    code: str,
    plan: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    candidates: List[Dict[str, Any]] = []
    site_summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    selection_mode = "none"
    if not _source_uses_dataflow(code):
        return candidates, site_summaries, skipped, selection_mode

    plan_targets = _normalize_dataflow_plan_targets(plan)
    targets = plan_targets if plan_targets else [{"option": option} for option in DATAFLOW_OPTION_DEFAULTS]
    selection_mode = "plan" if plan_targets else "heuristic_fallback"
    for target in targets:
        option = str(target["option"]).lower()
        values = list(target.get("try_values", [])) or list(DATAFLOW_OPTION_DEFAULTS.get(option, []))
        if not values:
            skipped.append({"kind": "config_dataflow", "option": option, "reason": "No candidate values were available"})
            continue
        rendered: List[Any] = []
        for value in values:
            candidate = _make_candidate(
                family="dataflow_config",
                kind="config_dataflow",
                target_key=option,
                target_line=0,
                parameter_kind=option,
                parameter_value=value,
                tcl_overrides={"config_dataflow": {f"-{option}": value}},
                intent="dataflow_balance",
                risk="medium",
                selection_source=selection_mode,
                priority=str(target.get("priority", "")),
                reason=str(target.get("reason", "")),
                conflict_keys=[f"dataflow:{option}"],
            )
            candidates.append(candidate)
            rendered.append(candidate["tcl_overrides"]["config_dataflow"])
        site_summaries.append(
            {
                "line": 0,
                "kind": "config_dataflow",
                "option": option,
                "generated_variants": rendered,
                "selection_source": selection_mode,
                "reason": target.get("reason", ""),
            }
        )
    return candidates, site_summaries, skipped, selection_mode


def _family_mode_summary(modes: Dict[str, str]) -> str:
    used = {value for value in modes.values() if value and value != "none"}
    if not used:
        return "none"
    if len(used) == 1:
        return next(iter(used))
    return "mixed"


def build_candidate_payload(
    code_path: Path,
    variant_kind: str,
    source_turn: Optional[int],
    max_pragmas: int,
    plan_path: Optional[Path] = None,
) -> Dict[str, Any]:
    code = code_path.read_text(encoding="utf-8")
    code_lines = code.splitlines()
    define_values = _load_define_values(code)
    sites = _scan_pragma_sites(code, max_pragmas=max_pragmas)
    plan = _load_plan(plan_path)
    plan_source_turn = plan.get("source_turn")
    try:
        inherited_source_turn = int(plan_source_turn) if plan_source_turn is not None else None
    except (TypeError, ValueError):
        inherited_source_turn = None
    based_on_hardware_rewrite = bool(plan.get("based_on_hardware_rewrite")) or variant_kind.lower() == "hardware"
    resolved_variant_kind = variant_kind
    if based_on_hardware_rewrite and variant_kind.lower() in {"", "optimized", "auto"}:
        resolved_variant_kind = "hardware"
    resolved_source_turn = source_turn if source_turn is not None else inherited_source_turn

    pragma_candidates, site_summaries, skipped_pragmas, unmatched_plan_targets, pragma_mode = _build_pragma_candidates(
        plan=plan,
        sites=sites,
        code_lines=code_lines,
        define_values=define_values,
    )
    interface_candidates, interface_sites, skipped_interfaces, interface_mode = _build_interface_candidates(code=code, plan=plan)
    storage_candidates, storage_sites, skipped_storage, storage_mode = _build_storage_candidates(code=code, plan=plan)
    op_candidates, op_sites, skipped_ops, op_mode = _build_op_candidates(code=code, plan=plan)
    dataflow_candidates, dataflow_sites, skipped_dataflow, dataflow_mode = _build_dataflow_candidates(code=code, plan=plan)

    candidates = pragma_candidates + interface_candidates + storage_candidates + op_candidates + dataflow_candidates
    for idx, candidate in enumerate(candidates, start=1):
        candidate["id"] = idx

    selection_mode_by_family = {
        "pragma": pragma_mode,
        "interface": interface_mode,
        "storage": storage_mode,
        "op_binding": op_mode,
        "dataflow_config": dataflow_mode,
    }
    family_candidate_counts = {
        "pragma": len(pragma_candidates),
        "interface": len(interface_candidates),
        "storage": len(storage_candidates),
        "op_binding": len(op_candidates),
        "dataflow_config": len(dataflow_candidates),
    }
    total_plan_target_count = (
        len(_normalize_pragma_plan_targets(plan))
        + len(_normalize_interface_plan_targets(plan))
        + len(_normalize_storage_plan_targets(plan))
        + len(_normalize_op_plan_targets(plan))
        + len(_normalize_dataflow_plan_targets(plan))
    )
    matched_target_count = sum(1 for value in selection_mode_by_family.values() if value == "plan")

    return {
        "generator": "pragma-tuning/gen_candidates.py",
        "selection_mode": _family_mode_summary(selection_mode_by_family),
        "selection_mode_by_family": selection_mode_by_family,
        "plan_path": str(plan_path.resolve()) if plan_path and plan_path.exists() else "",
        "plan_target_count": total_plan_target_count,
        "matched_target_count": matched_target_count,
        "unmatched_plan_targets": unmatched_plan_targets,
        "source_code_path": str(code_path),
        "source_variant_kind": resolved_variant_kind,
        "source_turn": resolved_source_turn,
        "based_on_hardware_rewrite": based_on_hardware_rewrite,
        "pragma_site_count": len(sites),
        "tunable_site_count": sum(1 for site in site_summaries if site["generated_variants"]),
        "candidate_count": len(candidates),
        "candidate_family_counts": family_candidate_counts,
        "pragma_sites": site_summaries,
        "interface_sites": interface_sites,
        "storage_sites": storage_sites,
        "op_sites": op_sites,
        "dataflow_sites": dataflow_sites,
        "skipped_pragmas": skipped_pragmas,
        "skipped_interfaces": skipped_interfaces,
        "skipped_storage": skipped_storage,
        "skipped_ops": skipped_ops,
        "skipped_dataflow": skipped_dataflow,
        "candidates": candidates,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate tuning candidates from existing pragmas and related HLS Tcl knobs")
    ap.add_argument("--code", required=True, help="Path to C/C++ source")
    ap.add_argument("--output", default="pragma_candidates.json", help="Output JSON path")
    ap.add_argument("--max-pragmas", type=int, default=24)
    ap.add_argument("--variant-kind", default="optimized", help="Source variant kind, e.g. hardware/software")
    ap.add_argument("--source-turn", type=int, help="Source optimized code turn number")
    ap.add_argument("--plan", help="Optional tuning plan JSON produced by hardware rewrite")
    args = ap.parse_args()

    code_path = Path(args.code).resolve()
    plan_path = Path(args.plan).resolve() if args.plan else None
    payload = build_candidate_payload(
        code_path=code_path,
        variant_kind=args.variant_kind,
        source_turn=args.source_turn,
        max_pragmas=args.max_pragmas,
        plan_path=plan_path,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": args.output,
                "candidate_count": payload["candidate_count"],
                "selection_mode": payload["selection_mode"],
                "candidate_family_counts": payload["candidate_family_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
