#!/usr/bin/env python3
"""Differential Original-C vs Rewrite-C verification with AMD Vitis C-sim."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from typing import Any


STATUS_VALUES = {"PASS", "FAIL", "ERROR", "TIMEOUT"}


def _split_parameters(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depths = {"(": 0, "[": 0, "<": 0}
    closing = {")": "(", "]": "[", ">": "<"}
    for char in text:
        if char in depths:
            depths[char] += 1
        elif char in closing and depths[closing[char]] > 0:
            depths[closing[char]] -= 1
        if char == "," and not any(depths.values()):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _find_matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _strip_comments(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _extract_signature(source: str, top: str) -> dict[str, Any]:
    clean = _strip_comments(source)
    match = re.search(rf"\b{re.escape(top)}\s*\(", clean)
    if not match:
        raise ValueError(f"top function '{top}' was not found")
    open_paren = clean.find("(", match.start())
    close_paren = _find_matching(clean, open_paren, "(", ")")
    if close_paren < 0:
        raise ValueError(f"top function '{top}' has an unterminated parameter list")
    suffix = clean[close_paren + 1 : close_paren + 256]
    if "{" not in suffix:
        next_match = re.search(rf"\b{re.escape(top)}\s*\(", clean[close_paren + 1 :])
        if next_match:
            offset = close_paren + 1 + next_match.start()
            return _extract_signature(clean[offset:], top)
        raise ValueError(f"top function '{top}' has no definition")

    prefix = clean[: match.start()].rstrip()
    boundary = max(prefix.rfind(";"), prefix.rfind("}"), prefix.rfind("{"), prefix.rfind("\n"))
    return_type = prefix[boundary + 1 :].strip()
    return_type = re.sub(r"\b(?:static|inline|extern|constexpr)\b", "", return_type)
    return_type = " ".join(return_type.split())
    if not return_type:
        raise ValueError(f"could not determine return type for '{top}'")

    params: list[dict[str, str]] = []
    parameter_text = clean[open_paren + 1 : close_paren].strip()
    if parameter_text and parameter_text != "void":
        for raw in _split_parameters(parameter_text):
            raw = re.sub(r"\s*=.*$", "", raw).strip()
            parsed = re.match(
                r"(?P<type>.+?)(?P<name>[A-Za-z_]\w*)\s*(?P<arrays>(?:\[[^\]]*\]\s*)*)$",
                raw,
            )
            if not parsed:
                raise ValueError(f"unsupported parameter declaration: {raw}")
            type_text = parsed.group("type").strip()
            arrays = parsed.group("arrays").strip()
            if not type_text:
                raise ValueError(f"missing type in parameter declaration: {raw}")
            params.append(
                {
                    "raw": raw,
                    "type": type_text,
                    "name": parsed.group("name"),
                    "arrays": arrays,
                }
            )
    return {"return_type": return_type, "params": params}


def _extract_integer_defines(source: str) -> dict[str, int]:
    raw_values: dict[str, str] = {}
    for name, value in re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([^\r\n]+)", source, re.MULTILINE):
        value = value.strip()
        if "(" in name:
            continue
        raw_values[name] = value

    resolved: dict[str, int] = {}
    for _ in range(len(raw_values) + 1):
        changed = False
        for name, expression in raw_values.items():
            if name in resolved:
                continue
            try:
                resolved[name] = _eval_integer_expression(expression, resolved)
                changed = True
            except ValueError:
                pass
        if not changed:
            break
    return resolved


def _eval_integer_expression(expression: str, names: dict[str, int]) -> int:
    expression = re.sub(r"\b([A-Za-z_]\w*)\b", lambda match: str(names[match.group(1)]) if match.group(1) in names else match.group(1), expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(str(exc)) from exc

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod)):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.FloorDiv, ast.Div)):
                return left // right
            return left % right
        raise ValueError("expression is not a supported integer constant")

    return visit(tree)


def _extract_aliases(source: str) -> dict[str, str]:
    aliases = {
        alias: target.strip()
        for target, alias in re.findall(r"\btypedef\s+([^;]+?)\s+([A-Za-z_]\w*)\s*;", source)
    }
    aliases.update(
        {
            alias: target.strip()
            for alias, target in re.findall(r"\busing\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", source)
        }
    )
    return aliases


def _base_type(param: dict[str, str]) -> str:
    value = param["type"]
    value = value.replace("*", " ").replace("&", " ")
    value = re.sub(r"\b(?:const|volatile|restrict|__restrict|__restrict__)\b", "", value)
    return " ".join(value.split())


def _resolved_base_type(param: dict[str, str], aliases: dict[str, str]) -> str:
    value = _base_type(param)
    seen: set[str] = set()
    while value in aliases and value not in seen:
        seen.add(value)
        value = " ".join(aliases[value].split())
    return value


def _is_pointer_like(param: dict[str, str]) -> bool:
    return bool(param["arrays"] or "*" in param["type"])


def _array_extent(param: dict[str, str], defines: dict[str, int], buf_size: int) -> int:
    if not param["arrays"]:
        return buf_size if "*" in param["type"] else 0
    dimensions = re.findall(r"\[([^\]]*)\]", param["arrays"])
    extent = 1
    for dimension in dimensions:
        if not dimension.strip():
            return buf_size
        value = _eval_integer_expression(dimension.strip(), defines)
        if value <= 0:
            raise ValueError(f"non-positive array dimension '{dimension}' for {param['name']}")
        extent *= value
    return extent


def _is_float_type(type_name: str) -> bool:
    lowered = type_name.lower()
    return any(token in lowered for token in ("float", "double", "ap_fixed", "ap_ufixed", "half"))


def _is_supported_numeric_type(type_name: str) -> bool:
    lowered = type_name.lower()
    if any(token in lowered for token in ("struct ", "class ", "hls::stream", "std::", "enum ")):
        return False
    return bool(re.fullmatch(r"[A-Za-z_:][\w:<>,\s]*", type_name))


def _normalize_type(type_name: str) -> str:
    value = re.sub(r"\b(?:const|volatile|restrict|__restrict|__restrict__)\b", "", type_name)
    return re.sub(r"\s+", "", value).lower()


def _validate_signatures(
    original: dict[str, Any],
    rewrite: dict[str, Any],
    original_source: str,
    rewrite_source: str,
) -> None:
    if len(original["params"]) != len(rewrite["params"]):
        raise ValueError("top-function parameter count changed after rewrite")
    if _normalize_type(original["return_type"]) != _normalize_type(rewrite["return_type"]):
        raise ValueError(
            f"top-function return type changed: {original['return_type']} -> {rewrite['return_type']}"
        )
    original_aliases = _extract_aliases(original_source)
    rewrite_aliases = _extract_aliases(rewrite_source)
    for index, (lhs, rhs) in enumerate(zip(original["params"], rewrite["params"])):
        if _is_pointer_like(lhs) != _is_pointer_like(rhs):
            raise ValueError(f"parameter {index} changed between scalar and array/pointer")
        lhs_type = _normalize_type(_resolved_base_type(lhs, original_aliases))
        rhs_type = _normalize_type(_resolved_base_type(rhs, rewrite_aliases))
        if lhs_type != rhs_type:
            raise ValueError(f"parameter {index} base type changed: {lhs_type} -> {rhs_type}")


def _cpp_string(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _build_design_source(
    source_name: str,
    trace_path: Path,
    signature: dict[str, Any],
    source_text: str,
    top: str,
    trials: int,
    seed: int,
    buf_size: int,
) -> str:
    defines = _extract_integer_defines(source_text)
    aliases = _extract_aliases(source_text)
    declarations: list[str] = []
    initializers: list[str] = []
    arguments: list[str] = []
    trace_lines: list[str] = []

    for index, param in enumerate(signature["params"]):
        base = _resolved_base_type(param, aliases)
        if not _is_supported_numeric_type(base):
            raise ValueError(f"unsupported C-sim parameter type for {param['name']}: {base}")
        floating = _is_float_type(base)
        name = f"arg_{index}"
        arguments.append(name)
        if _is_pointer_like(param):
            extent = _array_extent(param, defines, buf_size)
            declarations.append(f"    {_base_type(param)} {name}[{extent}];")
            if floating:
                value = f"({_base_type(param)})(_csim_uniform() * 4.0 - 2.0)"
            elif any(token in param["name"].lower() for token in ("addr", "idx", "index", "col")):
                value = f"({_base_type(param)})(k % {max(1, min(extent, buf_size))})"
            else:
                value = f"({_base_type(param)})((long long)(_csim_uniform() * 17.0) - 8)"
            initializers.append(f"    for (int k = 0; k < {extent}; ++k) {name}[k] = {value};")
            fmt = "F %.17g" if floating else "I %lld"
            cast = f"(double){name}[k]" if floating else f"(long long){name}[k]"
            trace_lines.append(
                f'    for (int k = 0; k < {extent}; ++k) std::fprintf(trace, "T %d A {index} %d {fmt}\\n", trial, k, {cast});'
            )
        else:
            param_type = param["type"]
            if floating:
                value = f"({param_type})(_csim_uniform() * 4.0 - 2.0)"
            else:
                value = f"({param_type})(_csim_uniform() * ({max(2, buf_size)} + 1))"
            declarations.append(f"    {param_type} {name} = {value};")

    return_type = signature["return_type"]
    call = f"{top}({', '.join(arguments)})"
    if _normalize_type(return_type) == "void":
        call_lines = [f"    {call};"]
    else:
        return_float = _is_float_type(return_type)
        call_lines = [f"    {return_type} result = {call};"]
        if return_float:
            call_lines.append('    std::fprintf(trace, "T %d R F %.17g\\n", trial, (double)result);')
        else:
            call_lines.append('    std::fprintf(trace, "T %d R I %lld\\n", trial, (long long)result);')

    return f'''#include "{source_name}"
#include <cstdint>
#include <cstdio>

static unsigned long long _csim_state = 1ULL;
static double _csim_uniform() {{
    _csim_state = _csim_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((_csim_state >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
}}

int verification_top() {{
  _csim_state = {seed}ULL;
  std::FILE *trace = std::fopen("{_cpp_string(trace_path)}", "w");
  if (!trace) return 2;
  for (int trial = 0; trial < {trials}; ++trial) {{
{os.linesep.join(declarations)}
{os.linesep.join(initializers)}
{os.linesep.join(call_lines)}
{os.linesep.join(trace_lines)}
  }}
  std::fclose(trace);
  return 0;
}}
'''


def _build_tcl(part: str, clock_period: str) -> str:
    return f'''open_project csim_project
set_top verification_top
add_files equiv_design.cpp
add_files -tb equiv_tb.cpp
open_solution solution -flow_target vivado
set_part {{{part}}}
create_clock -period {{{clock_period}}} -name default
csim_design -clean
close_project
exit
'''


def _run_vitis(run_dir: Path, timeout: int) -> dict[str, Any]:
    vitis_run = os.environ.get("VITIS_RUN", "vitis-run")
    executable = shutil.which(vitis_run)
    if not executable:
        return {"status": "ERROR", "reason": f"'{vitis_run}' was not found in PATH"}
    command = [executable, "--mode", "hls", "--tcl", "run.tcl"]
    try:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        (run_dir / "vitis_csim.log").write_text((stdout or "") + (stderr or ""), encoding="utf-8")
        return {"status": "TIMEOUT", "reason": f"Vitis C-sim exceeded {timeout}s"}

    combined = (stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")
    log_path = run_dir / "vitis_csim.log"
    log_path.write_text(combined, encoding="utf-8")
    if process.returncode != 0:
        error_lines = [line.strip() for line in combined.splitlines() if "ERROR:" in line.upper()]
        reason = error_lines[-1] if error_lines else f"vitis-run returned {process.returncode}"
        return {"status": "ERROR", "reason": reason[:800], "log": str(log_path)}
    return {"status": "PASS", "reason": "C-sim completed", "log": str(log_path)}


def _parse_trace(path: Path) -> list[tuple[tuple[str, ...], str, float | int]]:
    records: list[tuple[tuple[str, ...], str, float | int]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if len(parts) < 5 or parts[0] != "T" or parts[-2] not in {"F", "I"}:
            raise ValueError(f"malformed trace line {line_number}: {raw[:160]}")
        kind = parts[-2]
        value = float(parts[-1]) if kind == "F" else int(parts[-1])
        records.append((tuple(parts[:-2]), kind, value))
    return records


def _compare_traces(original: Path, rewrite: Path, atol: float, rtol: float) -> dict[str, Any]:
    lhs = _parse_trace(original)
    rhs = _parse_trace(rewrite)
    if len(lhs) != len(rhs):
        return {"status": "FAIL", "reason": f"trace length mismatch: original={len(lhs)}, rewrite={len(rhs)}"}
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        left_key, left_kind, left_value = left
        right_key, right_kind, right_value = right
        if left_key != right_key or left_kind != right_kind:
            return {
                "status": "FAIL",
                "reason": f"trace schema mismatch at record {index}",
                "mismatch": {"original": left, "rewrite": right},
            }
        if left_kind == "I":
            equal = left_value == right_value
        else:
            left_float = float(left_value)
            right_float = float(right_value)
            equal = (
                (math.isnan(left_float) and math.isnan(right_float))
                or left_float == right_float
                or math.isclose(left_float, right_float, rel_tol=rtol, abs_tol=atol)
            )
        if not equal:
            return {
                "status": "FAIL",
                "reason": f"output mismatch at {' '.join(left_key)}",
                "mismatch": {
                    "record": index,
                    "key": list(left_key),
                    "kind": left_kind,
                    "original": left_value,
                    "rewrite": right_value,
                    "atol": atol,
                    "rtol": rtol,
                },
            }
    return {"status": "PASS", "reason": f"all {len(lhs)} trace records matched", "record_count": len(lhs)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    original_path = Path(args.original).resolve()
    rewrite_path = Path(args.rewrite).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not original_path.is_file() or not rewrite_path.is_file():
        raise ValueError("original and rewrite source files must exist")
    if args.trials < 1:
        raise ValueError("trials must be at least 1")
    if args.buf_size < 1:
        raise ValueError("buf-size must be at least 1")
    if args.timeout < 1:
        raise ValueError("timeout must be at least 1 second")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be non-negative")

    original_text = original_path.read_text(encoding="utf-8", errors="replace")
    rewrite_text = rewrite_path.read_text(encoding="utf-8", errors="replace")
    original_signature = _extract_signature(original_text, args.top)
    rewrite_signature = _extract_signature(rewrite_text, args.top)
    _validate_signatures(original_signature, rewrite_signature, original_text, rewrite_text)

    work_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Any] = {}
    for label, source_path in (
        ("original", original_path),
        ("rewrite", rewrite_path),
    ):
        run_dir = work_dir / label
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir()
        source_copy = run_dir / "design_under_test.cc"
        shutil.copy2(source_path, source_copy)
        trace_path = run_dir / "csim_trace.txt"
        design_source = _build_design_source(
            source_copy.name,
            trace_path,
            original_signature,
            original_text,
            args.top,
            args.trials,
            args.seed,
            args.buf_size,
        )
        (run_dir / "equiv_design.cpp").write_text(design_source, encoding="utf-8")
        (run_dir / "equiv_tb.cpp").write_text(
            "extern int verification_top();\nint main() { return verification_top(); }\n",
            encoding="utf-8",
        )
        (run_dir / "run.tcl").write_text(_build_tcl(args.part, args.clock_period), encoding="utf-8")
        result = _run_vitis(run_dir, args.timeout)
        result["trace"] = str(trace_path)
        runs[label] = result
        if result["status"] != "PASS":
            return {
                "status": result["status"],
                "reason": f"{label} C-sim failed: {result['reason']}",
                "runs": runs,
            }
        if not trace_path.is_file():
            return {"status": "ERROR", "reason": f"{label} C-sim produced no trace", "runs": runs}

    comparison = _compare_traces(
        Path(runs["original"]["trace"]),
        Path(runs["rewrite"]["trace"]),
        args.atol,
        args.rtol,
    )
    return {
        **comparison,
        "top": args.top,
        "trials": args.trials,
        "seed": args.seed,
        "buf_size": args.buf_size,
        "atol": args.atol,
        "rtol": args.rtol,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewrite", help="rewritten C/C++ or HLS-C source")
    parser.add_argument("original", help="original C/C++ source")
    parser.add_argument("top", help="top function name")
    parser.add_argument("--part", required=True, help="target FPGA part")
    parser.add_argument("--clock-period", required=True, help="Vitis clock period, such as 10ns")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--report")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buf-size", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "ERROR", "reason": str(exc)}
    status = str(result.get("status", "ERROR")).upper()
    if status not in STATUS_VALUES:
        status = "ERROR"
        result["status"] = status
    report_path = Path(args.report).resolve() if args.report else Path(args.work_dir).resolve() / "csim_equiv_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"[csim_status]={status}")
    print(f"[csim_reason]={result.get('reason', '')}")
    print(f"[csim_report]={report_path}")
    if result.get("mismatch"):
        print(f"[csim_mismatch]={json.dumps(result['mismatch'], separators=(',', ':'))}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
