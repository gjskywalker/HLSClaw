#!/usr/bin/env python3
"""
Parse Vitis HLS csynth.xml and extract structured performance/resource data.
Outputs JSON report with bottleneck analysis and optimization recommendations.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


def _strip_text(value: Optional[str]) -> str:
    return (value or "").strip()


def _text(node: Optional[ET.Element], path: str, default: str = "") -> str:
    if node is None:
        return default
    found = node.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _to_int(value: str) -> Optional[int]:
    value = _strip_text(value)
    if not value or value == "-":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _to_float(value: str) -> Optional[float]:
    value = _strip_text(value)
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clean_issue(value: str) -> str:
    value = _strip_text(value)
    return "" if value == "-" else value


def _compute_timing_slack(target: Optional[float], uncertainty: Optional[float], estimated: Optional[float]) -> Optional[float]:
    if target is None or estimated is None:
        return None
    return target - (uncertainty or 0.0) - estimated


def extract_latency_info(perf_node: Optional[ET.Element]) -> Dict[str, Any]:
    """Extract latency information from a PerformanceEstimates node."""
    latency: Dict[str, Any] = {}
    overall = perf_node.find("SummaryOfOverallLatency") if perf_node is not None else None
    if overall is None:
        return latency

    latency["best_case"] = _to_int(_text(overall, "Best-caseLatency"))
    latency["worst_case"] = _to_int(_text(overall, "Worst-caseLatency"))
    latency["average_case"] = _to_int(_text(overall, "Average-caseLatency"))
    latency["interval"] = _to_int(_text(overall, "PipelineInitiationInterval"))
    latency["interval_min"] = _to_int(_text(overall, "Interval-min"))
    latency["interval_max"] = _to_int(_text(overall, "Interval-max"))
    latency["pipeline_depth"] = _to_int(_text(overall, "PipelineDepth"))
    latency["pipeline_type"] = _text(overall, "PipelineType", default="no")
    return latency


def extract_loop_info(perf_node: Optional[ET.Element]) -> List[Dict[str, Any]]:
    """Extract loop hierarchy and performance info."""
    loops: List[Dict[str, Any]] = []
    loop_section = perf_node.find("SummaryOfLoopLatency") if perf_node is not None else None
    if loop_section is None:
        return loops

    for loop_elem in list(loop_section):
        if not isinstance(loop_elem.tag, str):
            continue
        loop_info = {
            "name": _text(loop_elem, "Name", default=loop_elem.tag),
            "latency": _to_int(_text(loop_elem, "Latency")) or 0,
            "iteration_latency": _to_int(_text(loop_elem, "IterationLatency")) or 0,
            "trip_count": _to_int(_text(loop_elem, "TripCount")) or 0,
            "pipelined": _text(loop_elem, "PipelineType", default="no"),
            "pipeline_ii": _to_int(_text(loop_elem, "PipelineII")),
            "pipeline_depth": _to_int(_text(loop_elem, "PipelineDepth")),
            "slack": _to_float(_text(loop_elem, "Slack")),
            "absolute_time_latency": _text(loop_elem, "AbsoluteTimeLatency"),
        }
        loops.append(loop_info)
    return loops


def extract_resource_info(area_node: Optional[ET.Element]) -> Dict[str, Any]:
    """Extract resource utilization."""
    resources: Dict[str, Any] = {}
    if area_node is None:
        return resources

    current = area_node.find("Resources")
    if current is not None:
        resources["bram"] = _to_int(_text(current, "BRAM_18K")) or 0
        resources["dsp"] = _to_int(_text(current, "DSP")) or 0
        resources["ff"] = _to_int(_text(current, "FF")) or 0
        resources["lut"] = _to_int(_text(current, "LUT")) or 0
        resources["uram"] = _to_int(_text(current, "URAM")) or 0

    available = area_node.find("AvailableResources")
    if available is not None:
        resources["available_bram"] = _to_int(_text(available, "BRAM_18K")) or 0
        resources["available_dsp"] = _to_int(_text(available, "DSP")) or 0
        resources["available_ff"] = _to_int(_text(available, "FF")) or 0
        resources["available_lut"] = _to_int(_text(available, "LUT")) or 0
        resources["available_uram"] = _to_int(_text(available, "URAM")) or 0

    return resources


def extract_timing_info(assignments_node: Optional[ET.Element], perf_node: Optional[ET.Element]) -> Dict[str, Any]:
    """Extract timing analysis."""
    timing: Dict[str, Any] = {}
    timing["target_clock"] = _to_float(_text(assignments_node, "TargetClockPeriod"))
    timing["uncertainty"] = _to_float(_text(assignments_node, "ClockUncertainty"))

    timing_section = perf_node.find("SummaryOfTimingAnalysis") if perf_node is not None else None
    timing["estimated_clock"] = _to_float(_text(timing_section, "EstimatedClockPeriod"))
    timing["effective_target_clock"] = None
    if timing["target_clock"] is not None:
        timing["effective_target_clock"] = timing["target_clock"] - (timing["uncertainty"] or 0.0)
    timing["slack"] = _compute_timing_slack(
        timing.get("target_clock"),
        timing.get("uncertainty"),
        timing.get("estimated_clock"),
    )
    return timing


def extract_violation_info(violations_node: Optional[ET.Element]) -> Dict[str, Any]:
    """Extract issue / violation summary."""
    issue_type = _clean_issue(_text(violations_node, "IssueType"))
    violation_type = _clean_issue(_text(violations_node, "ViolationType"))
    source_location = _text(violations_node, "SourceLocation")

    loop_violations: List[Dict[str, Any]] = []
    if violations_node is not None:
        loop_violations_root = violations_node.find("SummaryOfLoopViolations")
        if loop_violations_root is not None:
            for loop_elem in list(loop_violations_root):
                if not isinstance(loop_elem.tag, str):
                    continue
                loop_issue = _clean_issue(_text(loop_elem, "IssueType"))
                loop_violation = _clean_issue(_text(loop_elem, "ViolationType"))
                if not loop_issue and not loop_violation and not _text(loop_elem, "SourceLocation"):
                    continue
                loop_violations.append(
                    {
                        "name": _text(loop_elem, "Name", default=loop_elem.tag),
                        "issue_type": loop_issue,
                        "violation_type": loop_violation,
                        "source_location": _text(loop_elem, "SourceLocation"),
                    }
                )

    return {
        "issue_type": issue_type,
        "violation_type": violation_type,
        "source_location": source_location,
        "loop_violations": loop_violations,
    }


def extract_module_info(root: ET.Element, assignments_node: Optional[ET.Element]) -> List[Dict[str, Any]]:
    modules: List[Dict[str, Any]] = []
    for module_elem in root.findall("./ModuleInformation/Module"):
        name = _text(module_elem, "Name", default="unknown")
        perf = module_elem.find("PerformanceEstimates")
        area = module_elem.find("AreaEstimates")
        module_timing = extract_timing_info(assignments_node, perf)
        module_latency = extract_latency_info(perf)
        module_loops = extract_loop_info(perf)
        module_violations = extract_violation_info(perf.find("SummaryOfViolations") if perf is not None else None)

        loop_violation_map = {item["name"]: item for item in module_violations["loop_violations"]}
        for loop in module_loops:
            loop_violation = loop_violation_map.get(loop["name"])
            if loop_violation:
                loop["issue_type"] = loop_violation.get("issue_type", "")
                loop["violation_type"] = loop_violation.get("violation_type", "")
                loop["source_location"] = loop_violation.get("source_location", "")

        modules.append(
            {
                "name": name,
                "timing": module_timing,
                "latency": module_latency,
                "loops": module_loops,
                "resources": extract_resource_info(area),
                "violations": module_violations,
            }
        )
    return modules


def parse_performance_table_from_rpt(rpt_path: Path) -> List[Dict[str, Any]]:
    """Parse the performance/resource table from csynth.rpt when available."""
    if not rpt_path.exists():
        return []

    text = rpt_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"\* Performance & Resource Estimates:\s*(.+?)\n\s*Name Prefix:",
        text,
        re.DOTALL,
    )
    if not match:
        return []

    rows: List[Dict[str, Any]] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) != 15:
            continue
        if (
            parts[0].startswith("Modules")
            or parts[0] == "& Loops"
            or parts[1] == "Type"
            or set("".join(parts)) <= {"-", "+"}
        ):
            continue

        raw_name = parts[0]
        prefix = raw_name.lstrip()[:1]
        kind = {"+" : "module", "o": "loop", "*": "dataflow"}.get(prefix, "item")
        name = raw_name.lstrip("+o* ").rstrip()
        rows.append(
            {
                "name": name,
                "kind": kind,
                "raw_name": raw_name,
                "issue_type": _clean_issue(parts[1]),
                "violation_type": _clean_issue(parts[2]),
                "iteration_latency": _to_int(parts[3]),
                "interval": _to_int(parts[4]),
                "trip_count": _to_int(parts[5]),
                "pipelined": parts[6],
                "latency_cycles": _to_int(parts[7]),
                "latency_ns": _to_float(parts[8]),
                "slack": _to_float(parts[9]),
                "bram": parts[10],
                "dsp": parts[11],
                "ff": parts[12],
                "lut": parts[13],
                "uram": parts[14],
            }
        )
    return rows


def merge_rpt_rows(report: Dict[str, Any], rpt_rows: List[Dict[str, Any]]) -> None:
    """Overlay richer table data from csynth.rpt onto parsed XML data."""
    if not rpt_rows:
        return

    report["performance_table"] = rpt_rows

    top_name = report.get("module_name")
    modules_by_name = {module["name"]: module for module in report.get("modules", [])}
    top_loops_by_name = {loop["name"]: loop for loop in report.get("loops", [])}

    for row in rpt_rows:
        if row["name"] == top_name and row["kind"] in {"module", "dataflow", "item"}:
            report["timing"]["table_slack"] = row.get("slack")
            report["violations"]["issue_type"] = row.get("issue_type", report["violations"].get("issue_type", ""))
            report["violations"]["violation_type"] = row.get("violation_type", report["violations"].get("violation_type", ""))
            report["latency"]["interval"] = row.get("interval") or report["latency"].get("interval")
        module = modules_by_name.get(row["name"])
        if module:
            module["table_issue_type"] = row.get("issue_type", "")
            module["table_violation_type"] = row.get("violation_type", "")
            module["table_slack"] = row.get("slack")
            continue
        loop = top_loops_by_name.get(row["name"])
        if loop:
            loop["issue_type"] = row.get("issue_type", loop.get("issue_type", ""))
            loop["violation_type"] = row.get("violation_type", loop.get("violation_type", ""))
            if row.get("slack") is not None:
                loop["slack"] = row["slack"]

        for module in report.get("modules", []):
            for module_loop in module.get("loops", []):
                if module_loop.get("name") == row["name"]:
                    module_loop["issue_type"] = row.get("issue_type", module_loop.get("issue_type", ""))
                    module_loop["violation_type"] = row.get("violation_type", module_loop.get("violation_type", ""))
                    if row.get("slack") is not None:
                        module_loop["slack"] = row["slack"]


def build_constraint_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    timing = report.get("timing", {})
    perf_rows = report.get("performance_table", [])
    modules = report.get("modules", [])

    timing_violations: List[Dict[str, Any]] = []
    ii_violations: List[Dict[str, Any]] = []
    worst_slack = timing.get("slack")

    top_issue = report.get("violations", {}).get("issue_type", "")
    top_violation = report.get("violations", {}).get("violation_type", "")
    if "Timing" in top_issue or "Timing" in top_violation or (timing.get("slack") is not None and timing["slack"] < 0):
        timing_violations.append(
            {
                "name": report.get("module_name", "unknown"),
                "issue_type": top_issue,
                "violation_type": top_violation,
                "slack": timing.get("slack"),
                "source_location": report.get("violations", {}).get("source_location", ""),
            }
        )

    for module in modules:
        module_slack = module.get("timing", {}).get("slack")
        table_issue = module.get("table_issue_type", "")
        table_violation = module.get("table_violation_type", "")
        xml_issue = module.get("violations", {}).get("issue_type", "")
        xml_violation = module.get("violations", {}).get("violation_type", "")

        if module_slack is not None:
            worst_slack = module_slack if worst_slack is None else min(worst_slack, module_slack)

        if (
            module_slack is not None and module_slack < 0
        ) or "Timing" in table_issue or "Timing" in xml_issue:
            timing_violations.append(
                {
                    "name": module["name"],
                    "issue_type": table_issue or xml_issue,
                    "violation_type": table_violation or xml_violation,
                    "slack": module_slack if module_slack is not None else module.get("table_slack"),
                    "source_location": module.get("violations", {}).get("source_location", ""),
                }
            )

        for loop in module.get("loops", []):
            issue_type = loop.get("issue_type", "")
            violation_type = loop.get("violation_type", "")
            if issue_type == "II" or "Resource Limitation" in violation_type:
                ii_violations.append(
                    {
                        "name": f"{module['name']}::{loop['name']}",
                        "issue_type": issue_type,
                        "violation_type": violation_type,
                        "pipeline_ii": loop.get("pipeline_ii"),
                        "slack": loop.get("slack"),
                    }
                )

    for row in perf_rows:
        row_slack = row.get("slack")
        if row_slack is not None:
            worst_slack = row_slack if worst_slack is None else min(worst_slack, row_slack)
        if row.get("issue_type") == "II" or "Resource Limitation" in row.get("violation_type", ""):
            ii_violations.append(
                {
                    "name": row["name"],
                    "issue_type": row.get("issue_type", ""),
                    "violation_type": row.get("violation_type", ""),
                    "pipeline_ii": row.get("interval"),
                    "slack": row.get("slack"),
                }
            )

    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for item in items:
            key = (
                item.get("name", ""),
                item.get("issue_type", ""),
                item.get("violation_type", ""),
                item.get("pipeline_ii"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    timing_violations = _dedupe(timing_violations)
    ii_violations = _dedupe(ii_violations)

    return {
        "timing_feasible": worst_slack is None or worst_slack >= 0,
        "worst_slack": worst_slack,
        "timing_violations": timing_violations,
        "ii_violations": ii_violations,
    }


def analyze_bottlenecks(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze design and suggest optimizations."""
    loops = report.get("loops", [])
    timing = report.get("timing", {})
    constraint_summary = report.get("constraint_summary", {})
    bottlenecks: List[Dict[str, Any]] = []

    unpipelined = [loop for loop in loops if loop.get("pipelined") == "no"]
    if unpipelined:
        bottlenecks.append(
            {
                "type": "pipelining",
                "severity": "high",
                "description": f"{len(unpipelined)} loop(s) not pipelined",
                "affected_loops": [loop["name"] for loop in unpipelined],
                "recommendation": "Add #pragma HLS pipeline II=1 to enable loop pipelining",
            }
        )

    for loop in loops:
        if loop.get("iteration_latency", 0) > 10:
            bottlenecks.append(
                {
                    "type": "iteration_latency",
                    "severity": "medium",
                    "description": f"Loop {loop['name']} has high iteration latency ({loop['iteration_latency']} cycles)",
                    "recommendation": "Consider loop unrolling or reducing loop body complexity",
                }
            )

    if constraint_summary.get("timing_violations"):
        offenders = constraint_summary["timing_violations"][:3]
        desc = ", ".join(
            f"{item['name']} (slack={item.get('slack')})"
            for item in offenders
        )
        bottlenecks.append(
            {
                "type": "timing",
                "severity": "critical",
                "description": f"Timing infeasible under current clock target; worst offenders: {desc}",
                "recommendation": "Reduce critical-path depth, lower aggressive parallelism, or restructure arithmetic/memory access in the offending module(s)",
            }
        )
    elif timing.get("estimated_clock") and timing.get("target_clock") and timing["estimated_clock"] > timing["target_clock"]:
        bottlenecks.append(
            {
                "type": "timing",
                "severity": "critical",
                "description": f"Timing violation: estimated {timing['estimated_clock']}ns > target {timing['target_clock']}ns",
                "recommendation": "Reduce combinational logic depth or increase clock period",
            }
        )

    if constraint_summary.get("ii_violations"):
        offenders = constraint_summary["ii_violations"][:3]
        desc = ", ".join(
            f"{item['name']} ({item.get('violation_type') or item.get('issue_type')})"
            for item in offenders
        )
        bottlenecks.append(
            {
                "type": "ii_violation",
                "severity": "high",
                "description": f"Pipeline II is constrained by resource or scheduling limits; offenders: {desc}",
                "recommendation": "Reduce unroll/parallelism pressure, improve memory banking/partitioning, or relax the target II where necessary",
            }
        )

    return bottlenecks


def parse_csynth(xml_path: str) -> Dict[str, Any]:
    """Parse csynth.xml and return structured report."""
    xml_file = Path(xml_path)
    root = ET.parse(xml_file).getroot()
    assignments = root.find("UserAssignments")
    perf = root.find("PerformanceEstimates")
    area = root.find("AreaEstimates")

    report: Dict[str, Any] = {
        "module_name": _text(assignments, "TopModelName", default=_text(root, "Name", default="unknown")),
        "latency": extract_latency_info(perf),
        "loops": extract_loop_info(perf),
        "resources": extract_resource_info(area),
        "timing": extract_timing_info(assignments, perf),
        "violations": extract_violation_info(perf.find("SummaryOfViolations") if perf is not None else None),
        "modules": extract_module_info(root, assignments),
    }

    rpt_rows = parse_performance_table_from_rpt(xml_file.with_suffix(".rpt"))
    merge_rpt_rows(report, rpt_rows)
    report["constraint_summary"] = build_constraint_summary(report)
    report["bottlenecks"] = analyze_bottlenecks(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Parse Vitis HLS csynth.xml report")
    parser.add_argument("xml_path", type=str, help="Path to csynth.xml")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed report")
    parser.add_argument(
        "--kernel",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    report = parse_csynth(args.xml_path)
    json_output = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).write_text(json_output, encoding="utf-8")
        print(f"Report saved to: {args.output}")

    if args.verbose or not args.output:
        print(json_output)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Module: {report['module_name']}")
    print(f"Latency: {report['latency'].get('best_case', 'N/A')} cycles")
    print(
        "Resources: "
        f"DSP={report['resources'].get('dsp', 0)}, "
        f"FF={report['resources'].get('ff', 0)}, "
        f"LUT={report['resources'].get('lut', 0)}"
    )

    worst_slack = report.get("constraint_summary", {}).get("worst_slack")
    if worst_slack is not None:
        print(f"Worst slack: {worst_slack:.3f} ns")

    if report["bottlenecks"]:
        print(f"\nBottlenecks found: {len(report['bottlenecks'])}")
        for bottleneck in report["bottlenecks"]:
            print(f"  [{bottleneck['severity'].upper()}] {bottleneck['type']}: {bottleneck['description']}")
            print(f"           -> {bottleneck['recommendation']}")
    else:
        print("\nNo major bottlenecks detected.")


if __name__ == "__main__":
    main()
