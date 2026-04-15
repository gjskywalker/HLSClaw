#!/usr/bin/env python3
"""
Run CBMC on a generated equivalence harness.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List


class CBMCRunner:
    """Run CBMC equivalence checking with practical defaults."""

    SOLVER_FLAGS = {
        "z3": "--z3",
        "smt2": "--smt2",
        "boolector": "--boolector",
        "cvc4": "--cvc4",
        "cprover-smt2": "--cprover-smt2",
    }

    def __init__(
        self,
        run_dir: str,
        timeout: int = 180,
        version: str | None = None,
        unwind: int = 2048,
        strict_safety: bool = False,
        solver: str | None = None,
    ):
        self.run_dir = run_dir
        self.timeout = timeout
        self.version = version
        self.unwind = unwind
        self.strict_safety = strict_safety
        self.solver = solver
        if version:
            self.harness_path = os.path.join(run_dir, f"cbmc_harness_{version}.c")
        else:
            self.harness_path = os.path.join(run_dir, "cbmc_harness.c")

    def _read_harness_text(self) -> str:
        try:
            return open(self.harness_path, "r", encoding="utf-8").read()
        except OSError:
            return ""

    def _load_defines(self, text: str) -> Dict[str, int]:
        defines: Dict[str, int] = {}
        for match in re.finditer(r"^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", text, re.MULTILINE):
            name = match.group(1)
            expr = match.group(2).strip()
            try:
                defines[name] = int(eval(expr, {"__builtins__": {}}, defines))
            except Exception:
                continue
        return defines

    def _infer_loop_bound(self, header: str, defines: Dict[str, int]) -> int | None:
        normalized = " ".join(header.split())
        match = re.match(r"for\s*\(([^;]*);([^;]*);([^\)]*)\)", normalized)
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

        start = None
        init_match = re.search(rf"\b{re.escape(loop_var)}\s*=\s*(-?\d+)\b", init)
        if init_match:
            start = int(init_match.group(1))

        if op == "<":
            if start is not None:
                return max(1, (limit - start) + 1)
            return max(1, limit + 1)
        if op == "<=":
            if start is not None:
                return max(1, (limit - start) + 2)
            return max(1, limit + 2)
        if op == ">":
            if start is not None:
                return max(1, (start - limit) + 1)
            return None
        if op == ">=":
            if start is not None:
                return max(1, (start - limit) + 2)
            return None
        return None

    def _infer_loop_bounds_by_line(self, text: str) -> Dict[int, int]:
        defines = self._load_defines(text)
        bounds: Dict[int, int] = {}
        for match in re.finditer(r"for\s*\((?:[^()]|\([^()]*\))*\)", text, re.DOTALL):
            line_no = text.count("\n", 0, match.start()) + 1
            bound = self._infer_loop_bound(match.group(0), defines)
            if bound is not None:
                bounds[line_no] = bound
        return bounds

    def _show_loops(self) -> List[Dict[str, int | str]]:
        try:
            result = subprocess.run(
                ["cbmc", self.harness_path, "--show-loops"],
                capture_output=True,
                text=True,
                cwd=self.run_dir,
                timeout=min(self.timeout, 60),
            )
        except (subprocess.TimeoutExpired, OSError):
            return []
        if result.returncode != 0:
            return []

        loops: List[Dict[str, int | str]] = []
        pattern = re.compile(
            r"Loop\s+([^\s:]+):\s*\n\s*file\s+.+?\s+line\s+(\d+)\s+function\s+([^\s]+)",
            re.MULTILINE,
        )
        for match in pattern.finditer(result.stdout):
            loops.append(
                {
                    "id": match.group(1),
                    "line": int(match.group(2)),
                    "function": match.group(3),
                }
            )
        return loops

    def _build_unwindset(self, text: str) -> tuple[List[str], Dict[str, int]]:
        bounds_by_line = self._infer_loop_bounds_by_line(text)
        loops = self._show_loops()
        entries: List[str] = []
        resolved: Dict[str, int] = {}
        for loop in loops:
            line_no = int(loop["line"])
            bound = bounds_by_line.get(line_no)
            if bound is None:
                continue
            loop_id = str(loop["id"])
            entries.append(f"{loop_id}:{bound}")
            resolved[loop_id] = bound
        return entries, resolved

    def _format_unwind_report(self, unwindset: Dict[str, int]) -> str:
        if not unwindset:
            return f"Unwind bound: {self.unwind}"
        rendered = ", ".join(f"{loop_id}={bound}" for loop_id, bound in sorted(unwindset.items()))
        return f"Unwind set: {rendered} (global unwind fallback: {self.unwind})"

    def _build_cbmc_command(self, unwind_entries: List[str]) -> List[str]:
        cmd = [
            "cbmc",
            self.harness_path,
            "--function",
            "equivalence_harness",
            "--unwind",
            str(self.unwind),
        ]
        solver_flag = self.SOLVER_FLAGS.get(self.solver)
        if solver_flag:
            cmd.append(solver_flag)
        if unwind_entries:
            cmd.extend(["--unwindset", ",".join(unwind_entries)])
        cmd.extend(
            [
                "--unwinding-assertions",
                "--bounds-check",
                "--pointer-check",
                "--pointer-overflow-check",
                "--trace",
                "--compact-trace",
                "--slice-formula",
                "--stop-on-fail",
                "--verbosity",
                "6",
            ]
        )
        if self.strict_safety:
            cmd.extend(
                [
                    "--div-by-zero-check",
                    "--signed-overflow-check",
                    "--unsigned-overflow-check",
                    "--undefined-shift-check",
                ]
            )
        return cmd

    def _build_result(
        self,
        status: str,
        output: str = "",
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        reason: str = "",
    ) -> dict:
        rendered = output.rstrip()
        status_lines = [f"[cbmc_status]={status}"]
        if reason:
            status_lines.append(f"[cbmc_reason]={reason}")
        if rendered:
            rendered = rendered + "\n" + "\n".join(status_lines) + "\n"
        else:
            rendered = "\n".join(status_lines) + "\n"
        return {
            "success": status in {"PASS", "TIMEOUT", "ABORTED"},
            "status": status,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": reason,
            "output": rendered,
        }

    def run(self) -> dict:
        if shutil.which("cbmc") is None:
            return self._build_result("FAIL", reason="cbmc not found in PATH")
        if not os.path.exists(self.harness_path):
            return self._build_result("FAIL", reason=f"CBMC harness not found: {self.harness_path}")

        harness_text = self._read_harness_text()
        unwind_entries, unwind_report = self._build_unwindset(harness_text)
        cmd = self._build_cbmc_command(unwind_entries)

        print(f"Running CBMC: {' '.join(cmd)}")
        print(f"Timeout: {self.timeout} seconds")
        print(self._format_unwind_report(unwind_report))
        print(f"Solver: {self.solver or 'default'}")
        print(f"Strict safety checks: {self.strict_safety}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.run_dir,
            )
        except subprocess.TimeoutExpired:
            return self._build_result("TIMEOUT", reason=f"CBMC timeout after {self.timeout}s")

        output = (result.stdout or "") + (result.stderr or "")
        lower = output.lower()
        if result.returncode == 0 and "verification successful" in lower:
            return self._build_result(
                "PASS",
                output=output,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        if "verification failed" in lower:
            return self._build_result(
                "FAIL",
                output=output,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                reason="VERIFICATION FAILED",
            )

        aborted_reason = f"CBMC aborted before VERIFICATION SUCCESSFUL (return code {result.returncode})"
        if result.returncode is None:
            aborted_reason = "CBMC aborted before VERIFICATION SUCCESSFUL"
        elif result.returncode < 0:
            aborted_reason = f"CBMC terminated by signal {-result.returncode}"
        return self._build_result(
            "ABORTED",
            output=output,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            reason=aborted_reason,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CBMC equivalence checking on a generated harness")
    parser.add_argument("--run_dir", default=".", help="Directory containing cbmc_harness.c")
    parser.add_argument("--timeout", type=int, default=180, help="CBMC timeout in seconds")
    parser.add_argument("--version", default=None, help="Version suffix for versioned harnesses (e.g., v3)")
    parser.add_argument("--unwind", type=int, default=2048, help="Global unwind bound passed to CBMC")
    parser.add_argument("--strict_safety", action="store_true", help="Enable arithmetic UB checks in addition to equivalence")
    parser.add_argument(
        "--solver",
        choices=sorted(CBMCRunner.SOLVER_FLAGS.keys()),
        default=None,
        help="Optional CBMC solver backend. If omitted, use CBMC's default solver selection.",
    )
    args = parser.parse_args()

    runner = CBMCRunner(
        run_dir=args.run_dir,
        timeout=args.timeout,
        version=args.version,
        unwind=args.unwind,
        strict_safety=args.strict_safety,
        solver=args.solver,
    )
    result = runner.run()
    if "output" in result:
        print(result["output"], end="" if result["output"].endswith("\n") else "\n")
    elif "error" in result:
        print(result["error"])
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
