#!/usr/bin/env python3
"""
Evaluate pragma candidates with Vitis HLS.

This script reuses the profiling skill implementation for TCL generation,
Vitis invocation, and csynth.xml parsing, then ranks pragma candidates by QoR.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILING_SCRIPTS = REPO_ROOT / "skills" / "profiling" / "scripts"
PRAGMA_TUNING_SCRIPTS = REPO_ROOT / "skills" / "pragma-tuning" / "scripts"
if str(PROFILING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROFILING_SCRIPTS))
if str(PRAGMA_TUNING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PRAGMA_TUNING_SCRIPTS))

from gen_tcl import generate_tcl_script, save_tcl_script  # type: ignore  # noqa: E402
from gen_candidates import build_candidate_payload  # type: ignore  # noqa: E402
from parse_csynth import parse_csynth  # type: ignore  # noqa: E402
from run_vitis import run_vitis_tcl_with_status  # type: ignore  # noqa: E402


FOR_RE = re.compile(r"\bfor\s*\(")
HLS_PRAGMA_RE = re.compile(r"#\s*pragma\s+HLS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _pragma_kind(line: str) -> Optional[str]:
    match = HLS_PRAGMA_RE.search(line)
    if not match:
        return None
    return match.group(1).lower()


def _collect_preceding_pragmas(lines: List[str], loop_index: int) -> List[Dict[str, Any]]:
    attached: List[Dict[str, Any]] = []
    idx = loop_index - 1
    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("//"):
            idx -= 1
            continue
        kind = _pragma_kind(stripped)
        if kind:
            attached.append({"line": idx + 1, "kind": kind, "text": stripped})
            idx -= 1
            continue
        break
    attached.reverse()
    return attached


def _collect_body_pragmas(lines: List[str], loop_index: int) -> List[Dict[str, Any]]:
    attached: List[Dict[str, Any]] = []
    idx = loop_index + 1
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or stripped == "{" or stripped.startswith("//"):
            idx += 1
            continue
        kind = _pragma_kind(stripped)
        if kind:
            attached.append({"line": idx + 1, "kind": kind, "text": stripped})
            idx += 1
            continue
        break
    return attached


def _find_loop_body_bounds(lines: List[str], loop_index: int) -> Optional[tuple[int, int]]:
    brace_depth = 0
    saw_open_brace = False
    start = loop_index
    for idx in range(loop_index, len(lines)):
        line = lines[idx]
        for char in line:
            if char == "{":
                brace_depth += 1
                saw_open_brace = True
            elif char == "}":
                brace_depth -= 1
                if saw_open_brace and brace_depth == 0:
                    return start, idx
        if saw_open_brace and idx == loop_index:
            start = idx + 1
    return None


def _loop_nesting_depth(lines: List[str], loop_index: int) -> int:
    depth = 0
    for idx in range(loop_index):
        if not FOR_RE.search(lines[idx]):
            continue
        bounds = _find_loop_body_bounds(lines, idx)
        if bounds and bounds[0] <= loop_index <= bounds[1]:
            depth += 1
    return depth


def find_loops(code: str, max_loops: int) -> List[Dict[str, Any]]:
    lines = code.splitlines()
    loops: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not FOR_RE.search(line):
            continue
        loops.append(
            {
                "line": idx + 1,
                "existing_pragmas": _collect_preceding_pragmas(lines, idx) + _collect_body_pragmas(lines, idx),
                "nesting_depth": _loop_nesting_depth(lines, idx),
            }
        )
        if len(loops) >= max_loops:
            break
    return loops


def auto_generate_candidates(code_path: Path, max_loops: int) -> Dict[str, Any]:
    payload = build_candidate_payload(
        code_path=code_path,
        variant_kind="auto",
        source_turn=None,
        max_pragmas=max_loops,
    )
    payload["generator"] = "pragma-dse/run_pragma_dse.py:auto_generate_candidates"
    payload["based_on_hardware_rewrite"] = False
    return payload


def load_candidates(code_path: Path, candidates_path: Optional[Path], max_loops: int) -> Dict[str, Any]:
    if candidates_path:
        return json.loads(candidates_path.read_text(encoding="utf-8"))
    return auto_generate_candidates(code_path, max_loops=max_loops)


def _normalize_tcl_overrides(candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = candidate.get("tcl_overrides", {}) or {}
    normalized = {
        "config_compile": {},
        "config_interface": {},
        "config_dataflow": {},
        "config_rtl": {},
        "config_schedule": {},
        "config_op": [],
    }
    for key in ("config_compile", "config_interface", "config_dataflow", "config_rtl", "config_schedule"):
        value = raw.get(key, {})
        if not isinstance(value, dict):
            continue
        normalized[key] = {str(opt): opt_value for opt, opt_value in value.items()}
    config_op_value = raw.get("config_op", [])
    if isinstance(config_op_value, list):
        normalized["config_op"] = [str(item) for item in config_op_value if str(item).strip()]
    return normalized


def _render_tcl_commands(commands: List[str], top_func: str) -> List[str]:
    rendered: List[str] = []
    for command in commands:
        text = str(command).strip()
        if not text:
            continue
        try:
            rendered.append(text.format(top_func=top_func))
        except Exception:
            rendered.append(text.replace("{top_func}", top_func))
    return rendered


def _merge_tcl_overrides(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "config_compile": {},
        "config_interface": {},
        "config_dataflow": {},
        "config_rtl": {},
        "config_schedule": {},
        "config_op": [],
    }
    for candidate in candidates:
        overrides = _normalize_tcl_overrides(candidate)
        for key in ("config_compile", "config_interface", "config_dataflow", "config_rtl", "config_schedule"):
            merged[key].update(overrides.get(key, {}))
        for command in overrides.get("config_op", []):
            if command not in merged["config_op"]:
                merged["config_op"].append(command)
    return merged


def _candidate_signature_items(candidate: Dict[str, Any]) -> List[tuple[str, str]]:
    items: List[tuple[str, str]] = []
    for edit in candidate.get("edits", []):
        items.append(("edit", json.dumps(edit, sort_keys=True)))
    for directive in candidate.get("directives", []):
        items.append(("directive", str(directive)))
    for command in candidate.get("tcl_directives", []):
        items.append(("tcl_directive", str(command)))
    overrides = _normalize_tcl_overrides(candidate)
    for key in ("config_compile", "config_interface", "config_dataflow", "config_rtl", "config_schedule"):
        if overrides.get(key):
            items.append((key, json.dumps(overrides[key], sort_keys=True)))
    for command in overrides.get("config_op", []):
        items.append(("config_op", str(command)))
    return items


def extract_insertions(candidate: Dict[str, Any]) -> Dict[int, List[str]]:
    insertions: Dict[int, List[str]] = {}
    for directive in candidate.get("directives", []):
        if not isinstance(directive, str):
            continue
        line_no = candidate.get("insert_before_line")
        pragma = directive
        if "//" in pragma:
            pragma = pragma.split("//", 1)[0].rstrip()
        if line_no is None:
            match = re.search(r"before line\s+(\d+)", directive, re.IGNORECASE)
            if match:
                line_no = int(match.group(1))
        if line_no is None:
            continue
        insertions.setdefault(int(line_no), []).append(pragma)
    return insertions


def apply_candidate(code_path: Path, candidate: Dict[str, Any], output_path: Path) -> None:
    lines = code_path.read_text(encoding="utf-8").splitlines()
    rendered = list(lines)
    edits = list(candidate.get("edits", []))
    if edits:
        for edit in edits:
            if edit.get("type") != "replace_line":
                continue
            line_no = int(edit["line"])
            if line_no < 1 or line_no > len(rendered):
                raise ValueError(f"Candidate edit line {line_no} is out of range")
            rendered[line_no - 1] = str(edit["new_text"])

    insertions = extract_insertions(candidate)
    rendered_with_insertions: List[str] = []
    for idx, line in enumerate(rendered, start=1):
        for pragma in insertions.get(idx, []):
            rendered_with_insertions.append(pragma)
        rendered_with_insertions.append(line)
    output_path.write_text("\n".join(rendered_with_insertions) + "\n", encoding="utf-8")


def infer_clock_period(clock_period: Optional[str], target_freq_mhz: Optional[float]) -> str:
    if clock_period:
        return clock_period
    if target_freq_mhz and target_freq_mhz > 0:
        return f"{1000.0 / target_freq_mhz:.3f}"
    raise ValueError("Provide either --clock-period or --target-freq-mhz")


def find_csynth_report(run_dir: Path, project_name: str) -> Optional[Path]:
    exact = run_dir / project_name / "solution" / "syn" / "report" / "csynth.xml"
    if exact.is_file():
        return exact
    matches = sorted(run_dir.glob("**/syn/report/csynth.xml"))
    return matches[0] if matches else None


def _effective_interval(latency_info: Dict[str, Any]) -> float:
    for key in ("interval", "interval_min", "interval_max"):
        value = latency_info.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0:
            return parsed
    return 0.0


def score_report(
    report: Dict[str, Any],
    goal: str,
    resource_caps: Dict[str, Optional[float]],
    baseline_ii: float = 0.0,
    baseline_latency: float = 0.0,
) -> Dict[str, float]:
    """Score a synthesis report.

    When *baseline_ii* and *baseline_latency* are both > 0, the II and latency
    penalties are normalised by those reference values so that each design
    starts from 1.0 and improvements produce equal-scale deltas regardless of
    the absolute magnitudes.  This prevents high-II designs (e.g. getTanh with
    II ≈ 186 000) from having their latency signal crushed by the II term.

    When either reference is zero (e.g. when scoring the baseline itself),
    the original absolute formula is used as a fall-back.
    """
    latency_info = report.get("latency", {})
    latency = float(latency_info.get("best_case") or latency_info.get("worst_case") or 0.0)
    ii = _effective_interval(latency_info)
    timing = report.get("timing", {})
    target_clock = float(timing.get("target_clock") or 0.0)
    estimated_clock = float(timing.get("estimated_clock") or 0.0)
    constraint_summary = report.get("constraint_summary", {}) or {}
    worst_slack = constraint_summary.get("worst_slack")
    try:
        worst_slack = float(worst_slack)
    except (TypeError, ValueError):
        worst_slack = timing.get("slack")
        try:
            worst_slack = float(worst_slack)
        except (TypeError, ValueError):
            worst_slack = 0.0

    timing_feasible_raw = constraint_summary.get("timing_feasible")
    if timing_feasible_raw is None:
        timing_feasible = bool(worst_slack >= 0.0)
    else:
        timing_feasible = bool(timing_feasible_raw)
    timing_violation = 0.0 if timing_feasible else 1.0

    resources = report.get("resources", {})
    pressure_terms: List[float] = []
    for key, cap in resource_caps.items():
        used = float(resources.get(key, 0.0))
        if cap and cap > 0:
            pressure_terms.append(used / cap)
    resource_pressure = sum(pressure_terms) / len(pressure_terms) if pressure_terms else 0.0

    goal_text = goal.lower()
    use_normalized = baseline_ii > 0.0 and baseline_latency > 0.0
    if timing_feasible:
        score = 1_000_000.0
        if use_normalized:
            score -= (ii / baseline_ii) * (120.0 if "throughput" in goal_text else 80.0)
            score -= (latency / baseline_latency) * (40.0 if "latency" in goal_text else 10.0)
        else:
            score -= ii * (120.0 if "throughput" in goal_text else 80.0)
            score -= latency * (0.04 if "latency" in goal_text else 0.01)
        score -= resource_pressure * (80.0 if "resource" in goal_text else 40.0)
        score += max(0.0, worst_slack) * 0.1
    else:
        score = -1_000_000.0
        score += worst_slack * 1_000.0
        if use_normalized:
            score -= (ii / baseline_ii) * 0.001
            score -= (latency / baseline_latency) * 0.000001
        else:
            score -= ii * 0.001
            score -= latency * 0.000001
        score -= resource_pressure * 1.0

    return {
        "score": round(score, 4),
        "latency": latency,
        "ii": ii,
        "timing_violation": timing_violation,
        "timing_feasible": 1.0 if timing_feasible else 0.0,
        "worst_slack": round(worst_slack, 4),
        "estimated_clock": round(estimated_clock, 4) if estimated_clock else 0.0,
        "target_clock": round(target_clock, 4) if target_clock else 0.0,
        "resource_pressure": round(resource_pressure, 4),
    }


def summarize_candidate(
    candidate: Dict[str, Any],
    run_dir: Path,
    report: Optional[Dict[str, Any]],
    metrics: Dict[str, float],
    success: bool,
    elapsed_sec: float,
    error: str = "",
    warnings: Optional[List[str]] = None,
    vitis_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = {
        "id": candidate.get("id"),
        "family": candidate.get("family", ""),
        "kind": candidate.get("kind", ""),
        "success": success,
        "elapsed_sec": round(elapsed_sec, 2),
        "directives": candidate.get("directives", []),
        "tcl_directives": candidate.get("tcl_directives", []),
        "tcl_overrides": _normalize_tcl_overrides(candidate),
        "metrics": metrics,
        "run_dir": str(run_dir),
        "report_path": "",
        "timing": {},
        "resources": {},
        "error": error,
        "warnings": list(warnings or []),
        "vitis_status": vitis_status or {},
    }
    if report:
        summary["report_path"] = str(report.get("_report_path", ""))
        summary["timing"] = report.get("timing", {})
        summary["resources"] = report.get("resources", {})
    return summary


def _metric_value(result: Optional[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not result:
        return default
    metrics = result.get("metrics", {})
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def _delta_if_present(best: Optional[Dict[str, Any]], baseline: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not best or not baseline:
        return None
    return round(_metric_value(best, key) - _metric_value(baseline, key), 4)


def summarize_search_outcome(
    successful: List[Dict[str, Any]],
    successful_tuned: List[Dict[str, Any]],
    baseline_result: Optional[Dict[str, Any]],
    best_tuned: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    score_epsilon = 1e-6
    baseline_score = _metric_value(baseline_result, "score", -1e9) if baseline_result and baseline_result.get("success") else None
    improving = 0
    tied = 0
    regressing = 0
    if baseline_score is not None:
        for item in successful_tuned:
            score_delta = _metric_value(item, "score", -1e9) - baseline_score
            if score_delta > score_epsilon:
                improving += 1
            elif abs(score_delta) <= score_epsilon:
                tied += 1
            else:
                regressing += 1

    if not successful:
        search_outcome = "all_failed"
    elif not successful_tuned:
        search_outcome = "baseline_only_success"
    elif improving > 0:
        search_outcome = "improved"
    elif tied > 0:
        search_outcome = "baseline_tied"
    else:
        search_outcome = "baseline_best"

    return {
        "search_outcome": search_outcome,
        "score_epsilon": score_epsilon,
        "improving_tuned_candidate_count": improving,
        "baseline_tie_tuned_candidate_count": tied,
        "regressing_tuned_candidate_count": regressing,
        "baseline_metrics": baseline_result.get("metrics", {}) if baseline_result else {},
        "best_tuned_metrics": best_tuned.get("metrics", {}) if best_tuned else {},
        "best_tuned_score_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "score"),
        "best_tuned_latency_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "latency"),
        "best_tuned_ii_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "ii"),
        "best_tuned_timing_violation_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "timing_violation"),
        "best_tuned_timing_feasible_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "timing_feasible"),
        "best_tuned_worst_slack_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "worst_slack"),
        "best_tuned_resource_pressure_delta_vs_baseline": _delta_if_present(best_tuned, baseline_result, "resource_pressure"),
    }


def evaluate_candidate(
    candidate: Dict[str, Any],
    code_path: Path,
    top_func: str,
    part: str,
    clock_period: str,
    work_dir: Path,
    goal: str,
    resource_caps: Dict[str, Optional[float]],
    candidate_timeout_sec: int,
    baseline_ii: float = 0.0,
    baseline_latency: float = 0.0,
) -> Dict[str, Any]:
    candidate_id = candidate.get("id", "unknown")
    candidate_dir = work_dir / f"candidate_{candidate_id}"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    variant_path = candidate_dir / code_path.name
    apply_candidate(code_path, candidate, variant_path)

    project_name = f"pragma_dse_{candidate_id}"
    tcl_path = candidate_dir / "run.tcl"
    candidate_overrides = _normalize_tcl_overrides(candidate)
    config_compile_params = {"-pipeline_loops": "0"}
    config_compile_params.update(candidate_overrides.get("config_compile", {}))
    config_dataflow_params = {"-strict_mode": "warning"}
    config_dataflow_params.update(candidate_overrides.get("config_dataflow", {}))
    config_rtl_params = {"-enable_maxiConservative": "1"}
    config_rtl_params.update(candidate_overrides.get("config_rtl", {}))
    config_interface_params = candidate_overrides.get("config_interface") or None
    config_schedule_params = candidate_overrides.get("config_schedule") or None
    config_op_params = list(candidate_overrides.get("config_op", [])) or None
    directive_commands = _render_tcl_commands(list(candidate.get("tcl_directives", [])), top_func=top_func) or None
    tcl_script = generate_tcl_script(
        top_file_name=str(variant_path),
        top_func_name=top_func,
        set_part=part,
        create_clock_period=clock_period,
        project_name=project_name,
        config_compile_params=config_compile_params,
        config_interface_params=config_interface_params,
        config_dataflow_params=config_dataflow_params,
        config_rtl_params=config_rtl_params,
        config_schedule_params=config_schedule_params,
        config_op_params=config_op_params,
        directive_commands=directive_commands,
    )
    save_tcl_script(str(tcl_path), tcl_script)

    start = time.time()
    report: Optional[Dict[str, Any]] = None
    metrics = {
        "score": -1e9,
        "latency": 0.0,
        "ii": 0.0,
        "timing_violation": 1.0,
        "timing_feasible": 0.0,
        "worst_slack": -1e9,
        "estimated_clock": 0.0,
        "target_clock": 0.0,
        "resource_pressure": 0.0,
    }
    error = ""
    warnings: List[str] = []

    vitis_status: Dict[str, Any] = {}
    success = False
    try:
        vitis_status = run_vitis_tcl_with_status(
            str(tcl_path),
            cwd=str(candidate_dir),
            log_path=str(candidate_dir / "vitis_run.log"),
            timeout_sec=candidate_timeout_sec,
            treat_warnings_as_errors=False,
        )
        success = bool(vitis_status.get("success"))
        warnings = list(vitis_status.get("warnings", []) or [])

        report_path = find_csynth_report(candidate_dir, project_name)
        if report_path and report_path.is_file():
            report = parse_csynth(str(report_path))
            report["_report_path"] = str(report_path)
            metrics = score_report(
                report,
                goal=goal,
                resource_caps=resource_caps,
                baseline_ii=baseline_ii,
                baseline_latency=baseline_latency,
            )
        if vitis_status.get("timed_out"):
            error = f"Vitis HLS timed out after {vitis_status.get('timeout_sec', 0)}s"
            success = False
        elif vitis_status.get("errors"):
            error = "; ".join(str(item).strip() for item in vitis_status.get("errors", [])[:3] if str(item).strip())
            success = False
        elif not report_path or not report_path.is_file():
            error = "csynth.xml not found after Vitis HLS execution"
            success = False
        else:
            success = True
    except Exception as exc:
        error = f"Exception while evaluating candidate: {exc}"
        vitis_status = {
            "returncode": None,
            "timed_out": False,
            "timeout_sec": candidate_timeout_sec,
            "log_path": str(candidate_dir / "vitis_run.log"),
            "engine_log_path": str(candidate_dir / "logs" / "hls_run_tcl.log"),
            "primary_log_path": str(candidate_dir / "vitis_run.log"),
        }
        success = False

    elapsed_sec = time.time() - start
    return summarize_candidate(
        candidate=candidate,
        run_dir=candidate_dir,
        report=report,
        metrics=metrics,
        success=success,
        elapsed_sec=elapsed_sec,
        error=error,
        warnings=warnings,
        vitis_status={
            "returncode": vitis_status.get("returncode"),
            "timed_out": bool(vitis_status.get("timed_out")),
            "timeout_sec": vitis_status.get("timeout_sec"),
            "log_path": vitis_status.get("log_path", ""),
            "engine_log_path": vitis_status.get("engine_log_path", ""),
            "primary_log_path": vitis_status.get("primary_log_path", vitis_status.get("log_path", "")),
        },
    )


def build_resource_caps(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    return {
        "bram": args.max_bram,
        "dsp": args.max_dsp,
        "ff": args.max_ff,
        "lut": args.max_lut,
        "uram": args.max_uram,
    }


def candidate_site_key(candidate: Dict[str, Any]) -> tuple[str, str]:
    family = str(candidate.get("family") or candidate.get("kind") or "candidate").lower()
    target_key = str(candidate.get("target_key") or f"{candidate.get('target_line', 0)}:{candidate.get('kind', '')}")
    return (family, target_key)


def candidate_sort_key(candidate: Dict[str, Any]) -> tuple[int, int, str]:
    priority = str(candidate.get("priority", "")).lower()
    family_order = {
        "pragma": 0,
        "storage": 1,
        "interface": 2,
        "op_binding": 3,
        "dataflow_config": 4,
        "baseline": 5,
        "combo": 6,
    }
    family = str(candidate.get("family") or candidate.get("kind") or "").lower()
    return (
        PRIORITY_ORDER.get(priority, 3),
        family_order.get(family, 7),
        int(candidate.get("target_line") or 0),
        str(candidate.get("id")),
    )


def make_baseline_candidate() -> Dict[str, Any]:
    return {
        "id": "baseline",
        "family": "baseline",
        "kind": "baseline",
        "target_key": "baseline",
        "parameter_kind": "",
        "parameter_value": None,
        "target_line": 0,
        "source_pragma": "",
        "tuned_pragma": "",
        "directives": [],
        "edits": [],
        "tcl_directives": [],
        "tcl_overrides": {},
        "intent": "baseline",
        "risk": "low",
        "selection_source": "baseline",
        "conflict_keys": [],
        "site_keys": [["baseline", "baseline"]],
    }


def make_combo_candidate(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(members, key=candidate_sort_key)
    combo_id = "combo_" + "_".join(str(item.get("id")) for item in ordered)
    edits: List[Dict[str, Any]] = []
    directives: List[str] = []
    tcl_directives: List[str] = []
    priorities = []
    families: List[str] = []
    conflict_keys: List[str] = []
    for item in ordered:
        edits.extend(list(item.get("edits", [])))
        directives.extend(list(item.get("directives", [])))
        tcl_directives.extend(list(item.get("tcl_directives", [])))
        priorities.append(str(item.get("priority", "")).lower())
        families.append(str(item.get("family") or item.get("kind") or "").lower())
        for key in list(item.get("conflict_keys", [])):
            if key not in conflict_keys:
                conflict_keys.append(str(key))
    priority = "low"
    if "high" in priorities:
        priority = "high"
    elif "medium" in priorities:
        priority = "medium"
    return {
        "id": combo_id,
        "family": "combo",
        "kind": "combo",
        "target_key": combo_id,
        "parameter_kind": "multi",
        "parameter_value": None,
        "target_line": min(int(item.get("target_line") or 0) for item in ordered),
        "source_pragma": "",
        "tuned_pragma": "",
        "directives": directives,
        "edits": edits,
        "tcl_directives": tcl_directives,
        "tcl_overrides": _merge_tcl_overrides(ordered),
        "intent": "multi_site",
        "risk": "high" if len(ordered) > 1 else "medium",
        "selection_source": "progressive_combo",
        "priority": priority,
        "members": [item.get("id") for item in ordered],
        "site_keys": [list(candidate_site_key(item)) for item in ordered],
        "conflict_keys": conflict_keys,
        "combo_families": families,
    }


def combo_signature(candidate: Dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_candidate_signature_items(candidate)))


def _boltzmann_select(
    pool: List[Dict[str, Any]],
    beam_width: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Select *beam_width* states from *pool* using Boltzmann (softmax) weighting.

    At high *temperature* the selection is near-uniform (exploration).
    As *temperature* approaches 0 it converges to deterministic top-K (exploitation).
    Scores are shifted by the pool maximum before exponentiation for numerical stability.
    """
    if not pool:
        return []
    n = min(beam_width, len(pool))
    if temperature <= 0.0:
        return sorted(pool, key=lambda s: s["result"].get("metrics", {}).get("score", -1e9), reverse=True)[:n]

    scores = [float(s["result"].get("metrics", {}).get("score", -1e9)) for s in pool]
    max_score = max(scores)
    weights = [math.exp((s - max_score) / max(temperature, 1e-10)) for s in scores]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return sorted(pool, key=lambda s: s["result"].get("metrics", {}).get("score", -1e9), reverse=True)[:n]

    selected: List[Dict[str, Any]] = []
    remaining_idx = list(range(len(pool)))
    rem_weights = list(weights)
    for _ in range(n):
        if not remaining_idx:
            break
        tw = sum(rem_weights)
        if tw <= 0.0:
            break
        probs = [w / tw for w in rem_weights]
        pick = random.choices(range(len(remaining_idx)), weights=probs, k=1)[0]
        selected.append(pool[remaining_idx[pick]])
        remaining_idx.pop(pick)
        rem_weights.pop(pick)
    return selected


def progressive_search_candidates(
    candidates: List[Dict[str, Any]],
    max_candidates: int,
    max_combos: int,
    top_per_site: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sorted_candidates = sorted(candidates, key=candidate_sort_key)[:max_candidates]
    candidate_lookup = {str(candidate.get("id")): candidate for candidate in sorted_candidates}
    search_meta = {
        "input_candidate_count": len(candidates),
        "single_candidate_count": len(sorted_candidates),
        "max_combos": max_combos,
        "top_per_site": top_per_site,
    }
    return sorted_candidates, {"candidate_lookup": candidate_lookup, "search_meta": search_meta}


def run_search(
    candidates: List[Dict[str, Any]],
    code_path: Path,
    top_func: str,
    part: str,
    clock_period: str,
    work_dir: Path,
    goal: str,
    resource_caps: Dict[str, Optional[float]],
    jobs: int,
    candidate_timeout_sec: int,
    search_strategy: str,
    max_combos: int,
    top_per_site: int,
    beam_width: int,
    beam_temperature: float = 50.0,
    beam_cooling: float = 0.85,
    random_combo_fraction: float = 0.3,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    single_candidates, helper = progressive_search_candidates(
        candidates=candidates,
        max_candidates=len(candidates),
        max_combos=max_combos,
        top_per_site=top_per_site,
    )
    candidate_lookup = helper["candidate_lookup"]
    stages: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Stage 1: Baseline                                                    #
    # ------------------------------------------------------------------ #
    baseline_candidate = make_baseline_candidate()
    baseline_result = evaluate_candidate(
        candidate=baseline_candidate,
        code_path=code_path,
        top_func=top_func,
        part=part,
        clock_period=clock_period,
        work_dir=work_dir,
        goal=goal,
        resource_caps=resource_caps,
        candidate_timeout_sec=candidate_timeout_sec,
    )

    # Extract normalization factors from baseline for relative scoring.
    # When either factor is 0 (synthesis failed), score_report falls back
    # to the original absolute formula — no regression in behaviour.
    _bi = max(float((baseline_result.get("metrics") or {}).get("ii") or 0), 1.0) if baseline_result.get("success") else 0.0
    _bl = max(float((baseline_result.get("metrics") or {}).get("latency") or 0), 1.0) if baseline_result.get("success") else 0.0

    # Re-score baseline with the normalized formula so that its score is
    # on the same scale as all subsequent candidate scores.
    if _bi > 0.0 and _bl > 0.0:
        m = baseline_result["metrics"]
        goal_text = goal.lower()
        ii_w = 120.0 if "throughput" in goal_text else 80.0
        lat_w = 40.0 if "latency" in goal_text else 10.0
        res_w = 80.0 if "resource" in goal_text else 40.0
        ws = float(m.get("worst_slack", 0.0))
        rp = float(m.get("resource_pressure", 0.0))
        if bool(m.get("timing_feasible")):
            norm_score = 1_000_000.0 - ii_w - lat_w - rp * res_w + max(0.0, ws) * 0.1
        else:
            norm_score = -1_000_000.0 + ws * 1_000.0 - 0.001 - 0.000001 - rp
        baseline_result["metrics"] = dict(m)
        baseline_result["metrics"]["score"] = round(norm_score, 4)

    all_results.append(baseline_result)
    stages.append(
        {
            "stage": "baseline",
            "planned": [baseline_candidate["id"]],
            "evaluated": [baseline_result["id"]],
        }
    )

    # ------------------------------------------------------------------ #
    # Stage 2: Single-site candidates (parallel)                           #
    # ------------------------------------------------------------------ #
    single_results = evaluate_candidates(
        candidates=single_candidates,
        code_path=code_path,
        top_func=top_func,
        part=part,
        clock_period=clock_period,
        work_dir=work_dir,
        goal=goal,
        resource_caps=resource_caps,
        jobs=jobs,
        candidate_timeout_sec=candidate_timeout_sec,
        baseline_ii=_bi,
        baseline_latency=_bl,
    )
    all_results.extend(single_results)
    stages.append(
        {
            "stage": "single_site",
            "planned": [candidate.get("id") for candidate in single_candidates],
            "evaluated": [result.get("id") for result in single_results],
        }
    )

    if search_strategy != "progressive" or max_combos <= 0:
        return all_results, {"stages": stages, **helper["search_meta"]}

    # ------------------------------------------------------------------ #
    # Stage 3: Progressive combo search with Boltzmann beam               #
    # ------------------------------------------------------------------ #
    # Split the combo budget: the first (1 - random_combo_fraction) goes
    # to the deterministic Boltzmann beam; the remainder goes to a random
    # sampling phase that explores combinations orthogonal to the beam path.
    random_combo_count = max(0, int(round(max_combos * max(0.0, min(1.0, random_combo_fraction)))))
    beam_combo_count = max_combos - random_combo_count

    baseline_score = float(baseline_result.get("metrics", {}).get("score", -1e9))
    successful_by_site: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
    for result in single_results:
        if not result.get("success"):
            continue
        candidate = candidate_lookup.get(str(result.get("id")))
        if not candidate:
            continue
        improvement = float(result.get("metrics", {}).get("score", -1e9)) - baseline_score
        if improvement <= 0:
            continue
        site = candidate_site_key(candidate)
        item = {
            "candidate": candidate,
            "result": result,
            "improvement": improvement,
        }
        successful_by_site.setdefault(site, []).append(item)

    ranked_sites = sorted(
        successful_by_site.items(),
        key=lambda item: (
            PRIORITY_ORDER.get(str(item[1][0]["candidate"].get("priority", "")).lower(), 3),
            -max(entry["improvement"] for entry in item[1]),
            item[0][0],
            item[0][1],
        ),
    )
    beam_states: List[Dict[str, Any]] = [{"members": [], "result": baseline_result}]
    combo_candidates: List[Dict[str, Any]] = []
    seen_single_signatures = {
        signature
        for signature in (combo_signature(candidate) for candidate in single_candidates)
        if signature
    }
    seen_signatures = set(seen_single_signatures)
    remaining_beam_budget = beam_combo_count
    current_beam_temperature = beam_temperature

    for _site_key, site_entries in ranked_sites:
        if remaining_beam_budget <= 0:
            break
        top_entries = sorted(site_entries, key=lambda item: item["result"]["metrics"]["score"], reverse=True)[:top_per_site]
        stage_batch: List[Dict[str, Any]] = []
        member_map: Dict[str, List[Dict[str, Any]]] = {}
        virtual_states: List[Dict[str, Any]] = []
        virtual_signatures = set()
        for state in beam_states:
            used_sites = {candidate_site_key(member) for member in state["members"]}
            used_conflicts = {
                str(conflict_key)
                for member in state["members"]
                for conflict_key in list(member.get("conflict_keys", []))
            }
            for entry in top_entries:
                candidate = entry["candidate"]
                site = candidate_site_key(candidate)
                if site in used_sites:
                    continue
                candidate_conflicts = {str(conflict_key) for conflict_key in list(candidate.get("conflict_keys", []))}
                if used_conflicts.intersection(candidate_conflicts):
                    continue
                members = list(state["members"]) + [candidate]
                combo_candidate = make_combo_candidate(members)
                signature = combo_signature(combo_candidate)
                if not signature:
                    continue
                if len(members) == 1 and signature in seen_single_signatures:
                    if signature in virtual_signatures:
                        continue
                    virtual_signatures.add(signature)
                    virtual_states.append({"members": members, "result": entry["result"]})
                    continue
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                stage_batch.append(combo_candidate)
                member_map[str(combo_candidate["id"])] = members
                if len(stage_batch) >= remaining_beam_budget:
                    break
            if len(stage_batch) >= remaining_beam_budget:
                break
        beam_pool = list(beam_states)
        beam_pool.extend(virtual_states)
        if stage_batch:
            combo_results = evaluate_candidates(
                candidates=stage_batch,
                code_path=code_path,
                top_func=top_func,
                part=part,
                clock_period=clock_period,
                work_dir=work_dir,
                goal=goal,
                resource_caps=resource_caps,
                jobs=jobs,
                candidate_timeout_sec=candidate_timeout_sec,
                baseline_ii=_bi,
                baseline_latency=_bl,
            )
            all_results.extend(combo_results)
            combo_candidates.extend(stage_batch)
            stages.append(
                {
                    "stage": "combo_site",
                    "planned": [candidate.get("id") for candidate in stage_batch],
                    "evaluated": [result.get("id") for result in combo_results],
                }
            )
            remaining_beam_budget -= len(stage_batch)
            for result in combo_results:
                if not result.get("success"):
                    continue
                members = member_map.get(str(result.get("id")), [])
                beam_pool.append({"members": members, "result": result})
        elif not virtual_states:
            continue

        # Boltzmann beam update — replaces hard top-K with temperature-weighted
        # sampling.  Temperature decays each round, transitioning from exploration
        # (early) to exploitation (late).
        beam_states = _boltzmann_select(beam_pool, max(1, beam_width), current_beam_temperature)
        current_beam_temperature *= beam_cooling

    # ------------------------------------------------------------------ #
    # Stage 4: Random combo fallback                                       #
    # ------------------------------------------------------------------ #
    # Spend any remaining budget on randomly sampled multi-site combinations
    # from the winner pool.  This "genetic crossover" step reaches combos that
    # the beam path would only find in later expansion rounds.
    if random_combo_count > 0 and len(successful_by_site) >= 2:
        winner_by_site: Dict[tuple[str, str], Dict[str, Any]] = {}
        for site, entries in successful_by_site.items():
            best_entry = max(entries, key=lambda e: e["result"]["metrics"]["score"])
            winner_by_site[site] = best_entry["candidate"]
        winner_sites = list(winner_by_site.keys())

        random_batch: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = random_combo_count * 10
        while len(random_batch) < random_combo_count and attempts < max_attempts:
            attempts += 1
            if len(winner_sites) < 2:
                break
            k = random.randint(2, min(4, len(winner_sites)))
            chosen_sites = random.sample(winner_sites, k)
            members = [winner_by_site[s] for s in chosen_sites]
            conflict_ok = True
            all_conflict_keys: set = set()
            for m in members:
                m_conflicts = {str(ck) for ck in m.get("conflict_keys", [])}
                if all_conflict_keys.intersection(m_conflicts):
                    conflict_ok = False
                    break
                all_conflict_keys.update(m_conflicts)
            if not conflict_ok:
                continue
            combo = make_combo_candidate(members)
            combo["selection_source"] = "random_combo"
            sig = combo_signature(combo)
            if not sig or sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            random_batch.append(combo)

        if random_batch:
            random_results = evaluate_candidates(
                candidates=random_batch,
                code_path=code_path,
                top_func=top_func,
                part=part,
                clock_period=clock_period,
                work_dir=work_dir,
                goal=goal,
                resource_caps=resource_caps,
                jobs=jobs,
                candidate_timeout_sec=candidate_timeout_sec,
                baseline_ii=_bi,
                baseline_latency=_bl,
            )
            all_results.extend(random_results)
            combo_candidates.extend(random_batch)
            stages.append(
                {
                    "stage": "random_combo",
                    "planned": [c.get("id") for c in random_batch],
                    "evaluated": [r.get("id") for r in random_results],
                }
            )

    meta = {
        "stages": stages,
        "combo_candidate_count": len(combo_candidates),
        "beam_combo_count": beam_combo_count,
        "random_combo_count": random_combo_count,
        "beam_temperature": beam_temperature,
        "beam_cooling": beam_cooling,
        "random_combo_fraction": random_combo_fraction,
        **helper["search_meta"],
    }
    return all_results, meta


def evaluate_candidates(
    candidates: List[Dict[str, Any]],
    code_path: Path,
    top_func: str,
    part: str,
    clock_period: str,
    work_dir: Path,
    goal: str,
    resource_caps: Dict[str, Optional[float]],
    jobs: int,
    candidate_timeout_sec: int,
    baseline_ii: float = 0.0,
    baseline_latency: float = 0.0,
) -> List[Dict[str, Any]]:
    if jobs <= 1 or len(candidates) <= 1:
        return [
            evaluate_candidate(
                candidate=candidate,
                code_path=code_path,
                top_func=top_func,
                part=part,
                clock_period=clock_period,
                work_dir=work_dir,
                goal=goal,
                resource_caps=resource_caps,
                candidate_timeout_sec=candidate_timeout_sec,
                baseline_ii=baseline_ii,
                baseline_latency=baseline_latency,
            )
            for candidate in candidates
        ]

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                evaluate_candidate,
                candidate,
                code_path,
                top_func,
                part,
                clock_period,
                work_dir,
                goal,
                resource_caps,
                candidate_timeout_sec,
                baseline_ii,
                baseline_latency,
            ): candidate.get("id")
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "id": candidate_id,
                        "success": False,
                        "elapsed_sec": 0.0,
                        "directives": [],
                        "metrics": {
                            "score": -1e9,
                            "latency": 0.0,
                            "ii": 0.0,
                            "timing_violation": 1.0,
                            "timing_feasible": 0.0,
                            "worst_slack": -1e9,
                            "estimated_clock": 0.0,
                            "target_clock": 0.0,
                            "resource_pressure": 0.0,
                        },
                        "run_dir": str(work_dir / f"candidate_{candidate_id}"),
                        "report_path": "",
                        "timing": {},
                        "resources": {},
                        "error": f"Unhandled evaluation error: {exc}",
                    }
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pragma DSE with Vitis HLS QoR evaluation")
    parser.add_argument("--code", required=True, help="Path to current optimized code")
    parser.add_argument("--top-func", required=True, help="Top function name")
    parser.add_argument("--part", required=True, help="Target FPGA part")
    parser.add_argument("--clock-period", help="Clock period passed to create_clock -period")
    parser.add_argument("--target-freq-mhz", type=float, help="Target frequency in MHz; converted to ns if clock period is omitted")
    parser.add_argument("--goal", default="", help="Optimization goal such as latency/throughput/resource")
    parser.add_argument("--candidates", help="Candidate JSON path from pragma-tuning; auto-generated if omitted")
    parser.add_argument("--work-dir", default="pragma_dse_runs", help="Directory for candidate runs")
    parser.add_argument("--output", default="pragma_dse_report.json", help="Output JSON report")
    parser.add_argument("--max-candidates", type=int, default=8, help="Maximum number of candidates to evaluate")
    parser.add_argument(
        "--search-strategy",
        choices=["singles", "progressive"],
        default="progressive",
        help="Search strategy. 'singles' evaluates baseline + single-site candidates. 'progressive' also evaluates small combinations.",
    )
    parser.add_argument("--max-combos", type=int, default=4, help="Maximum number of multi-site combo candidates to evaluate in progressive search")
    parser.add_argument("--top-per-site", type=int, default=1, help="How many successful single-site winners to keep per pragma site for combo expansion")
    parser.add_argument("--beam-width", type=int, default=2, help="Beam width for progressive combo search")
    parser.add_argument(
        "--beam-temperature",
        type=float,
        default=50.0,
        help="Initial temperature for Boltzmann beam selection (higher = more exploration). Default: 50.0.",
    )
    parser.add_argument(
        "--beam-cooling",
        type=float,
        default=0.85,
        help="Multiplicative cooling factor applied to beam temperature each combo round. Default: 0.85.",
    )
    parser.add_argument(
        "--random-combo-fraction",
        type=float,
        default=0.3,
        help=(
            "Fraction of --max-combos budget reserved for random multi-site sampling after the beam phase. "
            "E.g. 0.3 means 30%% of combos are random crossover candidates. Default: 0.3."
        ),
    )
    parser.add_argument("--max-loops", type=int, default=8, help="Maximum existing pragma sites to scan when auto-generating fallback candidates")
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of candidates to evaluate in parallel. Default: 4.",
    )
    parser.add_argument(
        "--candidate-timeout-sec",
        type=int,
        default=0,
        help="Per-candidate Vitis HLS timeout in seconds. Use 0 to disable the timeout. Default: 0.",
    )
    parser.add_argument("--max-bram", type=float, help="Optional BRAM budget")
    parser.add_argument("--max-dsp", type=float, help="Optional DSP budget")
    parser.add_argument("--max-ff", type=float, help="Optional FF budget")
    parser.add_argument("--max-lut", type=float, help="Optional LUT budget")
    parser.add_argument("--max-uram", type=float, help="Optional URAM budget")
    args = parser.parse_args()

    code_path = Path(args.code).resolve()
    if not code_path.is_file():
        raise FileNotFoundError(f"Code file not found: {code_path}")

    candidates_path = Path(args.candidates).resolve() if args.candidates else None
    if candidates_path and not candidates_path.is_file():
        raise FileNotFoundError(f"Candidate file not found: {candidates_path}")

    clock_period = infer_clock_period(args.clock_period, args.target_freq_mhz)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    payload = load_candidates(code_path=code_path, candidates_path=candidates_path, max_loops=args.max_loops)
    candidates = list(payload.get("candidates", []))[: args.max_candidates]
    resource_caps = build_resource_caps(args)

    jobs = args.jobs if args.jobs and args.jobs > 0 else 4
    results, search_meta = run_search(
        candidates=candidates,
        code_path=code_path,
        top_func=args.top_func,
        part=args.part,
        clock_period=clock_period,
        work_dir=work_dir,
        goal=args.goal,
        resource_caps=resource_caps,
        jobs=jobs,
        candidate_timeout_sec=args.candidate_timeout_sec,
        search_strategy=args.search_strategy,
        max_combos=args.max_combos,
        top_per_site=args.top_per_site,
        beam_width=args.beam_width,
        beam_temperature=args.beam_temperature,
        beam_cooling=args.beam_cooling,
        random_combo_fraction=args.random_combo_fraction,
    )

    ranked = sorted(results, key=lambda item: item.get("metrics", {}).get("score", -1e9), reverse=True)
    successful = [item for item in ranked if item.get("success")]
    successful_tuned = [item for item in successful if item.get("id") != "baseline"]
    best = successful[0] if successful else None
    best_tuned = successful_tuned[0] if successful_tuned else None
    baseline_result = next((item for item in results if item.get("id") == "baseline"), None)
    search_summary = summarize_search_outcome(
        successful=successful,
        successful_tuned=successful_tuned,
        baseline_result=baseline_result,
        best_tuned=best_tuned,
    )
    report = {
        "skill": "pragma-dse",
        "evaluator": "vitis-hls",
        "code": str(code_path),
        "top_func": args.top_func,
        "part": args.part,
        "clock_period": clock_period,
        "goal": args.goal,
        "search_strategy": args.search_strategy,
        "max_combos": args.max_combos,
        "top_per_site": args.top_per_site,
        "beam_width": args.beam_width,
        "beam_temperature": args.beam_temperature,
        "beam_cooling": args.beam_cooling,
        "random_combo_fraction": args.random_combo_fraction,
        "jobs": jobs,
        "candidate_timeout_sec": args.candidate_timeout_sec,
        "candidate_count": len(candidates),
        "evaluated_count": len(results),
        "successful_candidate_count": len(successful),
        "successful_tuned_candidate_count": len(successful_tuned),
        "baseline_success": bool(baseline_result and baseline_result.get("success")),
        "best_candidate_id": best.get("id") if best else None,
        "best_tuned_candidate_id": best_tuned.get("id") if best_tuned else None,
        "search_meta": search_meta,
        **search_summary,
        "top_candidates": ranked[: min(5, len(ranked))],
        "results": ranked,
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "candidate_count": len(candidates),
                "best_candidate_id": report["best_candidate_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
