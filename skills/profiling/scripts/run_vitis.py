#!/usr/bin/env python3
"""Run Vitis HLS through `vitis-run --mode hls --tcl`."""

import os
import re
import shutil
import signal
import subprocess
import sys
from typing import Any, Dict, List


WARNING_RE = re.compile(r"\bWARNING\s*:", re.IGNORECASE)
ERROR_RE = re.compile(r"\bERROR\s*:", re.IGNORECASE)


def _issue_lines(text: str, pattern: re.Pattern[str]) -> List[str]:
    issues: List[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not pattern.search(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        issues.append(line)
    return issues


def _write_debug_log(path: str, content: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _engine_log_path(cwd: str | None) -> str:
    return os.path.join(cwd or os.getcwd(), "logs", "hls_run_tcl.log")


def _load_engine_log(cwd: str | None) -> str:
    path = _engine_log_path(cwd)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _merge_issue_lists(*issue_lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for issue_list in issue_lists:
        for item in issue_list:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _default_debug_log_path(cwd: str | None) -> str:
    return os.path.join(cwd or os.getcwd(), "vitis_run.log")


def run_vitis_tcl_with_status(
    tcl_file: str,
    cwd: str | None = None,
    log_path: str | None = None,
    timeout_sec: int = 0,
    treat_warnings_as_errors: bool = False,
) -> Dict[str, Any]:
    """Run Vitis HLS and return a structured execution status."""

    debug_log = log_path or _default_debug_log_path(cwd)
    status: Dict[str, Any] = {
        "success": False,
        "returncode": None,
        "timed_out": False,
        "timeout_sec": timeout_sec,
        "log_path": debug_log,
        "engine_log_path": "",
        "primary_log_path": debug_log,
        "warnings": [],
        "errors": [],
    }

    if not os.path.exists(tcl_file):
        message = f"Error: TCL file not found: {tcl_file}"
        print(message, file=sys.stderr)
        status["errors"] = [message]
        _write_debug_log(debug_log, message + "\n")
        return status

    vitis_run = os.environ.get("VITIS_RUN", "vitis-run")
    if shutil.which(vitis_run):
        cmd = [vitis_run, "--mode", "hls", "--tcl", tcl_file]
    else:
        message = "Error: 'vitis-run' not found in PATH"
        hint = "Please source Xilinx settings: source /tools/Xilinx/Vitis/2023.2/settings64.sh"
        print(message, file=sys.stderr)
        print(hint, file=sys.stderr)
        status["errors"] = [message, hint]
        _write_debug_log(debug_log, message + "\n" + hint + "\n")
        return status

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=(os.name != "nt"),
        )
        if timeout_sec and timeout_sec > 0:
            stdout, stderr = process.communicate(timeout=timeout_sec)
        else:
            stdout, stderr = process.communicate()
        combined = (stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")
        _write_debug_log(debug_log, combined)

        engine_log_path = _engine_log_path(cwd)
        engine_log = _load_engine_log(cwd)
        wrapper_warnings = _issue_lines(combined, WARNING_RE)
        wrapper_errors = _issue_lines(combined, ERROR_RE)
        engine_warnings = _issue_lines(engine_log, WARNING_RE)
        engine_errors = _issue_lines(engine_log, ERROR_RE)
        warnings = _merge_issue_lists(engine_warnings, wrapper_warnings)
        errors = _merge_issue_lists(engine_errors, wrapper_errors)
        status.update(
            {
                "returncode": process.returncode,
                "engine_log_path": engine_log_path if engine_log else "",
                "primary_log_path": engine_log_path if engine_log else debug_log,
                "warnings": warnings,
                "errors": errors,
            }
        )

        status["success"] = (
            process.returncode == 0
            and not errors
            and (not treat_warnings_as_errors or not warnings)
        )
        if status["success"]:
            print("SUCCESS")
        else:
            print(f"FAILED - Debug log saved to {debug_log}", file=sys.stderr)
        return status

    except subprocess.TimeoutExpired as exc:
        if "process" in locals():
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            except Exception:
                process.kill()
            tail_stdout, tail_stderr = process.communicate()
        else:
            tail_stdout, tail_stderr = "", ""
        partial_stdout = _coerce_text(exc.stdout)
        partial_stderr = _coerce_text(exc.stderr)
        stdout = partial_stdout + _coerce_text(tail_stdout)
        stderr = partial_stderr + _coerce_text(tail_stderr)
        combined = (stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")
        message = f"Vitis HLS timed out after {timeout_sec}s"
        combined = combined + ("\n" if combined and not combined.endswith("\n") else "") + f"[TIMEOUT] {message}\n"
        _write_debug_log(debug_log, combined)
        engine_log_path = _engine_log_path(cwd)
        engine_log = _load_engine_log(cwd)
        wrapper_warnings = _issue_lines(combined, WARNING_RE)
        wrapper_errors = _issue_lines(combined, ERROR_RE)
        engine_warnings = _issue_lines(engine_log, WARNING_RE)
        engine_errors = _issue_lines(engine_log, ERROR_RE)
        warnings = _merge_issue_lists(engine_warnings, wrapper_warnings)
        errors = _merge_issue_lists(engine_errors, wrapper_errors)
        errors.append(message)
        status.update(
            {
                "timed_out": True,
                "returncode": None if "process" not in locals() else process.returncode,
                "engine_log_path": engine_log_path if engine_log else "",
                "primary_log_path": engine_log_path if engine_log else debug_log,
                "warnings": warnings,
                "errors": errors,
            }
        )
        print(f"FAILED - {message}. Debug log saved to {debug_log}", file=sys.stderr)
        return status
    except Exception as exc:
        message = f"Error running vitis-run: {exc}"
        _write_debug_log(debug_log, message + "\n")
        status["errors"] = [message]
        print(message, file=sys.stderr)
        return status


def run_vitis_tcl(
    tcl_file: str,
    cwd: str | None = None,
    log_path: str | None = None,
    timeout_sec: int = 0,
    treat_warnings_as_errors: bool = False,
) -> bool:
    """Backward-compatible wrapper that returns only success/fail."""

    status = run_vitis_tcl_with_status(
        tcl_file=tcl_file,
        cwd=cwd,
        log_path=log_path,
        timeout_sec=timeout_sec,
        treat_warnings_as_errors=treat_warnings_as_errors,
    )
    return bool(status.get("success"))


def run_vitis_hls_with_status(
    tcl_file: str,
    cwd: str | None = None,
    log_path: str | None = None,
    timeout_sec: int = 0,
    treat_warnings_as_errors: bool = False,
) -> Dict[str, Any]:
    return run_vitis_tcl_with_status(
        tcl_file=tcl_file,
        cwd=cwd,
        log_path=log_path,
        timeout_sec=timeout_sec,
        treat_warnings_as_errors=treat_warnings_as_errors,
    )


def run_vitis_hls(
    tcl_file: str,
    cwd: str | None = None,
    log_path: str | None = None,
    timeout_sec: int = 0,
    treat_warnings_as_errors: bool = False,
) -> bool:
    return run_vitis_tcl(
        tcl_file=tcl_file,
        cwd=cwd,
        log_path=log_path,
        timeout_sec=timeout_sec,
        treat_warnings_as_errors=treat_warnings_as_errors,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_vitis.py <tcl_file>", file=sys.stderr)
        sys.exit(1)
    
    tcl_file = sys.argv[1]
    success = run_vitis_tcl(tcl_file)
    sys.exit(0 if success else 1)
