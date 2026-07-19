from __future__ import annotations

import json
import os
import re
import shutil
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None

try:
    from .agent import (
        Agent,
        Candidate,
        FatalRunError,
        Scratchpad,
        SkillInfo,
        _c,
        _log_info,
        _log_phase,
        _log_skill,
        _log_warn,
    )
except ImportError:
    from agent import (
        Agent,
        Candidate,
        FatalRunError,
        Scratchpad,
        SkillInfo,
        _c,
        _log_info,
        _log_phase,
        _log_skill,
        _log_warn,
    )


class AgentState(TypedDict, total=False):
    design_name: str
    run_id: str
    thread_id: str
    run_dir: str
    log_file: str
    input_mode: str
    skills: list[dict[str, str]]
    available_skills_xml: str
    profiling_analysis: str
    rag_analysis: str
    rewrite_analysis: str
    best_software_turn: int
    hardware_turn: int
    final_code_turn: int
    hardware_attempt: int
    hardware_feedback: str
    pragma_candidate_count: int
    pragma_dse_success: bool
    hardware_loop_continue: bool
    best_hardware_turn: int
    best_pragma_dse_score: float
    best_pragma_dse_candidate_id: str
    best_pragma_dse_report_path: str
    best_pragma_candidates_path: str
    analysis_turn: int
    command_turn: int
    reference_turn: int
    skill_turn: int
    optimized_code_turn: int
    scratchpad: dict[str, Any]
    candidates: list[dict[str, Any]]
    conversation_history: list[dict[str, str]]
    total_tokens_used: int
    stage_results: dict[str, Any]
    errors: list[str]
    final_summary: dict[str, Any]


class LangGraphAgent(Agent):
    """LangGraph-based orchestration that reuses the existing direct skill runners."""

    def __init__(
        self,
        config: dict[str, Any],
        system_prompt: str,
        code_path: str,
        req_path: str,
        top_function_name: str | None = None,
        input_mode: str = "plain_c",
        skills_dir: str | None = None,
        out_dir: str | None = None,
        interrupt_after: list[str] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            system_prompt=system_prompt,
            code_path=code_path,
            req_path=req_path,
            top_function_name=top_function_name,
            input_mode=input_mode,
            skills_dir=skills_dir,
            out_dir=out_dir,
        )
        self.interrupt_after = tuple(interrupt_after or [])
        self.checkpoint_backend = "memory"
        self.checkpoint_db_path = ""
        self._checkpointer_cm = None
        self.checkpointer = self._build_checkpointer()
        self.graph = self._build_graph()

    def _build_checkpointer(self) -> Any:
        if SqliteSaver is None:
            return InMemorySaver()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.getenv(
            "HLSCLAW_CHECKPOINT_DB",
            os.path.join(base_dir, "checkpoints", "langgraph_checkpoints.sqlite"),
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._checkpointer_cm = SqliteSaver.from_conn_string(db_path)
        self.checkpoint_backend = "sqlite"
        self.checkpoint_db_path = db_path
        return self._checkpointer_cm.__enter__()

    def close(self) -> None:
        if self._checkpointer_cm is not None:
            self._checkpointer_cm.__exit__(None, None, None)
            self._checkpointer_cm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("initialize_run", self._node_initialize_run)
        graph.add_node("discover_skills", self._node_discover_skills)
        graph.add_node("profiling", self._node_profiling)
        graph.add_node("kg_rag", self._node_kg_rag)
        graph.add_node("rewrite_guidance", self._node_rewrite_guidance)
        graph.add_node("software_rewrite", self._node_software_rewrite)
        graph.add_node("hardware_rewrite", self._node_hardware_rewrite)
        graph.add_node("pragma_tuning", self._node_pragma_tuning)
        graph.add_node("pragma_dse", self._node_pragma_dse)
        graph.add_node("finalize", self._node_finalize)

        graph.add_edge(START, "initialize_run")
        graph.add_edge("initialize_run", "discover_skills")
        graph.add_conditional_edges(
            "discover_skills",
            self._route_after_discover,
            {
                "profiling": "profiling",
                "kg_rag": "kg_rag",
                "software_rewrite": "software_rewrite",
            },
        )
        graph.add_conditional_edges(
            "profiling",
            self._route_after_profiling,
            {
                "kg_rag": "kg_rag",
                "software_rewrite": "software_rewrite",
            },
        )
        graph.add_conditional_edges(
            "software_rewrite",
            self._route_after_software_rewrite,
            {
                "kg_rag": "kg_rag",
                "hardware_rewrite": "hardware_rewrite",
                "finalize": "finalize",
            },
        )
        graph.add_edge("kg_rag", "hardware_rewrite")
        graph.add_conditional_edges(
            "hardware_rewrite",
            self._route_after_hardware_rewrite,
            {
                "kg_rag": "kg_rag",
                "pragma_tuning": "pragma_tuning",
                "pragma_dse": "pragma_dse",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "pragma_tuning",
            self._route_after_pragma_tuning,
            {
                "kg_rag": "kg_rag",
                "pragma_dse": "pragma_dse",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "pragma_dse",
            self._route_after_pragma_dse,
            {
                "kg_rag": "kg_rag",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        interrupt_after = list(self.interrupt_after) if self.interrupt_after else None
        return graph.compile(
            checkpointer=self.checkpointer,
            interrupt_after=interrupt_after,
            name="hlsclaw_graph",
        )

    def run_CoT(self) -> None:
        self.run_langgraph()

    def run_langgraph(
        self,
        initial_state: AgentState | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        state: AgentState = {"errors": [], "stage_results": {}}
        if initial_state:
            state.update(initial_state)
        resolved_thread_id = thread_id or state.get("thread_id") or f"hlsclaw-{int(time.time() * 1000)}"
        state["thread_id"] = resolved_thread_id
        final_state = self.graph.invoke(
            state,
            config=self._checkpoint_config(resolved_thread_id),
        )
        return self._finalize_checkpoint_artifacts(final_state)

    def resume_langgraph(self, thread_id: str) -> AgentState:
        try:
            record = self.get_thread_record(thread_id)
        except Exception:
            record = {}
        status = str(record.get("execution_status", "")).strip().lower()
        if status == "running":
            raise ValueError(f"Thread {thread_id} is still running and cannot be resumed.")
        if status == "fatal_error" and not bool(record.get("resumable", False)):
            raise ValueError(f"Thread {thread_id} ended with fatal_error and is not resumable.")

        snapshot = None
        values: dict[str, Any] = {}
        try:
            snapshot = self.get_checkpoint_state(thread_id)
            values = dict(getattr(snapshot, "values", {}) or {})
        except Exception:
            snapshot = None

        if values:
            values["thread_id"] = thread_id
            if getattr(snapshot, "next", ()):
                self._persist_running_thread(values, list(getattr(snapshot, "next", ()) or ()))
                final_state = self.graph.invoke(None, config=self._checkpoint_config(thread_id))
                return self._finalize_checkpoint_artifacts(final_state)
            return self._finalize_checkpoint_artifacts(values)

        state_path = self._find_state_path_for_thread(thread_id)
        if not state_path:
            raise ValueError(f"No checkpoint or serialized state found for thread_id={thread_id}")
        loaded_state = self._load_json_file(state_path)
        loaded_state["thread_id"] = thread_id
        summary = deepcopy(loaded_state.get("final_summary", {}))
        pending_nodes = list(summary.get("pending_nodes", []))
        if pending_nodes:
            self._persist_running_thread(loaded_state, pending_nodes)
        return self._resume_from_serialized_state(loaded_state)

    def _checkpoint_config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def get_checkpoint_state(self, thread_id: str) -> Any:
        return self.graph.get_state(config=self._checkpoint_config(thread_id))

    def get_checkpoint_history(self, thread_id: str) -> list[Any]:
        return list(self.graph.get_state_history(config=self._checkpoint_config(thread_id)))

    def _infer_run_identity(self, run_dir: str) -> tuple[str, str]:
        name = os.path.basename(run_dir.rstrip(os.sep))
        marker = "_run_"
        if marker in name:
            design_name, run_id = name.split(marker, 1)
            return design_name, run_id
        return name, ""

    def _infer_log_file(self, run_dir: str) -> str:
        if not run_dir or not os.path.isdir(run_dir):
            return ""
        matches = sorted(
            (
                os.path.join(run_dir, entry)
                for entry in os.listdir(run_dir)
                if entry.startswith("log_") and entry.endswith(".log")
            ),
            key=os.path.getmtime,
        )
        return matches[-1] if matches else ""

    def _infer_pending_node_from_state(self, state: AgentState) -> str | None:
        stage_results = dict(state.get("stage_results", {}) or {})
        input_mode = self._input_mode(state)
        if not stage_results:
            return "discover_skills"
        if input_mode == "plain_c" and "software_rewrite" not in stage_results:
            return "software_rewrite"
        if input_mode == "hls_native":
            if "profiling" not in stage_results and self._has_skill(state, "profiling") and self._check_vitis_available():
                return "profiling"
            if "kg_rag" not in stage_results and ("hardware_rewrite" not in stage_results):
                return "kg_rag" if self._has_skill(state, "kg-rag") or self._has_skill(state, "rag") else "hardware_rewrite"
        if state.get("pragma_dse_success"):
            return "finalize"
        if state.get("hardware_feedback") and int(state.get("hardware_attempt", 0) or 0) < self._max_hardware_opt_rounds():
            return "kg_rag" if self._has_skill(state, "kg-rag") or self._has_skill(state, "rag") else "hardware_rewrite"
        if "hardware_rewrite" not in stage_results:
            if ("kg_rag" in stage_results) or not (self._has_skill(state, "kg-rag") or self._has_skill(state, "rag")):
                return "hardware_rewrite"
            return "kg_rag"
        if self._has_skill(state, "pragma-tuning") and "pragma_tuning" not in stage_results:
            return "pragma_tuning"
        if self._has_skill(state, "pragma-dse") and self._check_vitis_available() and "pragma_dse" not in stage_results:
            return "pragma_dse"
        return "finalize"

    def persist_interrupted_thread(
        self,
        thread_id: str,
        reason: str = "Interrupted by user (Ctrl+C)",
        status_reason: str = "keyboard_interrupt",
    ) -> AgentState | None:
        snapshot = None
        values: dict[str, Any] = {}
        try:
            snapshot = self.get_checkpoint_state(thread_id)
            values = dict(getattr(snapshot, "values", {}) or {})
        except Exception:
            snapshot = None

        state: AgentState = deepcopy(values) if values else {}
        state.update(self._capture_runtime_state())
        state["thread_id"] = thread_id

        run_dir = str(state.get("run_dir") or self.scratchpad.run_dir or "")
        if not run_dir:
            return None
        state["run_dir"] = run_dir
        if not state.get("log_file"):
            state["log_file"] = self._infer_log_file(run_dir)
        design_name, run_id = self._infer_run_identity(run_dir)
        if design_name:
            state.setdefault("design_name", design_name)
        if run_id:
            state.setdefault("run_id", run_id)

        errors = list(state.get("errors", []))
        if reason not in errors:
            errors.append(reason)
        state["errors"] = errors

        self._restore_runtime_from_state(state)
        log_file = str(state.get("log_file", "") or "")
        if log_file:
            self._append_text(log_file, f"[WARN] {reason}\n")
            _log_warn(reason)

        pending_nodes = list(getattr(snapshot, "next", ()) or ()) if snapshot is not None else []
        if not pending_nodes:
            pending_nodes = list(state.get("final_summary", {}).get("pending_nodes", []) or [])
        if not pending_nodes:
            inferred = self._infer_pending_node_from_state(state)
            if inferred:
                pending_nodes = [inferred]

        if snapshot is not None:
            checkpoint_path = os.path.join(run_dir, "langgraph_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(self._snapshot_to_json(snapshot), f, indent=2)
                f.write("\n")
            self.scratchpad.stage_artifacts["langgraph_checkpoint"] = checkpoint_path
            if self.checkpoint_db_path:
                self.scratchpad.stage_artifacts["langgraph_checkpoint_db"] = self.checkpoint_db_path

        return self._finalize_state_only_artifacts(
            state,
            execution_status="interrupted",
            pending_nodes=pending_nodes,
            status_reason=status_reason,
            resumable=True,
        )

    def persist_fatal_error_thread(
        self,
        thread_id: str,
        reason: str,
        *,
        status_reason: str = "fatal_error",
        failing_stage: str = "",
        resumable: bool = False,
    ) -> AgentState | None:
        snapshot = None
        values: dict[str, Any] = {}
        try:
            snapshot = self.get_checkpoint_state(thread_id)
            values = dict(getattr(snapshot, "values", {}) or {})
        except Exception:
            snapshot = None

        state: AgentState = deepcopy(values) if values else {}
        state.update(self._capture_runtime_state())
        state["thread_id"] = thread_id

        run_dir = str(state.get("run_dir") or self.scratchpad.run_dir or "")
        if not run_dir:
            return None
        state["run_dir"] = run_dir
        if not state.get("log_file"):
            state["log_file"] = self._infer_log_file(run_dir)
        design_name, run_id = self._infer_run_identity(run_dir)
        if design_name:
            state.setdefault("design_name", design_name)
        if run_id:
            state.setdefault("run_id", run_id)

        errors = list(state.get("errors", []))
        if reason not in errors:
            errors.append(reason)
        state["errors"] = errors

        self._restore_runtime_from_state(state)
        log_file = str(state.get("log_file", "") or "")
        if log_file:
            self._append_text(log_file, f"[FATAL] {reason}\n")
            _log_warn(reason)

        pending_nodes = list(getattr(snapshot, "next", ()) or ()) if snapshot is not None else []
        if not resumable:
            pending_nodes = []
        elif not pending_nodes:
            pending_nodes = list(state.get("final_summary", {}).get("pending_nodes", []) or [])
        if resumable and not pending_nodes:
            inferred = self._infer_pending_node_from_state(state)
            if inferred:
                pending_nodes = [inferred]

        if snapshot is not None:
            checkpoint_path = os.path.join(run_dir, "langgraph_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(self._snapshot_to_json(snapshot), f, indent=2)
                f.write("\n")
            self.scratchpad.stage_artifacts["langgraph_checkpoint"] = checkpoint_path
            if self.checkpoint_db_path:
                self.scratchpad.stage_artifacts["langgraph_checkpoint_db"] = self.checkpoint_db_path

        return self._finalize_state_only_artifacts(
            state,
            execution_status="fatal_error",
            pending_nodes=pending_nodes,
            status_reason=status_reason,
            failing_stage=failing_stage,
            resumable=resumable,
        )

    def _reset_runtime_state(self) -> None:
        self.analysis_turn = 0
        self.command_turn = 0
        self.reference_turn = 0
        self.skill_turn = 0
        self.optimized_code_turn = 0
        self.client.total_tokens_used = 0
        self.scratchpad = Scratchpad(
            goal="",
            fpga="",
            target_frequency_mhz="",
            top_function_name=self.top_function_name,
            run_dir="",
            stage_artifacts={},
            analysis={},
            command={},
            reference={},
            json_artifact={},
            skill={},
            optimized_code={},
            optimized_code_file={},
            input_mode=self.input_mode,
        )
        self.candidates = []
        self.conversation_history = []

    def _append_error(self, state: AgentState, label: str, exc: Exception) -> dict[str, Any]:
        message = f"{label}: {exc}"
        log_file = state.get("log_file", "")
        fatal = self._classify_fatal_exception(label, exc)
        if fatal is not None:
            status_reason, resumable = fatal
            failing_stage = self._normalize_stage_label(label)
            if log_file:
                self._append_text(log_file, f"[FATAL] {message}\n")
            _log_warn(f"{label} failed fatally: {exc}")
            raise FatalRunError(
                message,
                status_reason=status_reason,
                failing_stage=failing_stage,
                resumable=resumable,
            )
        if log_file:
            self._append_text(log_file, f"[ERROR] {message}\n")
        _log_warn(f"{label} failed, continuing: {exc}")
        errors = list(state.get("errors", []))
        errors.append(message)
        return {"errors": errors}

    def _normalize_stage_label(self, label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
        return normalized or "unknown"

    def _classify_fatal_exception(self, label: str, exc: Exception) -> tuple[str, bool] | None:
        text = f"{label}: {exc}".lower()
        quota_signatures = (
            "insufficient_quota",
            "insufficient balance",
            "out of credit",
            "out of credits",
            "exceeded your current quota",
            "payment required",
            "billing",
        )
        if any(sig in text for sig in quota_signatures):
            return ("provider_quota_exceeded", False)

        auth_signatures = (
            "unauthorized",
            "forbidden",
            "invalid api key",
            "authentication",
            "auth token",
            "incorrect api key",
            "api key not provided",
        )
        if any(sig in text for sig in auth_signatures):
            return ("provider_auth_error", False)

        model_signatures = (
            "unknown_model",
            "model not found",
            "unsupported model",
            "model_not_supported",
            "unsupported_api_for_model",
        )
        if any(sig in text for sig in model_signatures):
            return ("provider_model_error", False)

        rate_limit_signatures = (
            "rate limit",
            "http 429",
            "too many requests",
        )
        if any(sig in text for sig in rate_limit_signatures):
            return ("provider_rate_limit", True)

        return None

    def _is_process_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _normalize_thread_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        status = str(normalized.get("execution_status", "")).strip().lower()
        if status == "running":
            pid = 0
            try:
                pid = int(normalized.get("process_pid", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            if pid > 0 and not self._is_process_alive(pid):
                normalized["execution_status"] = "external_abort"
                normalized.setdefault("status_reason", "process_terminated_unexpectedly")
                normalized["resumable"] = bool(normalized.get("state_path")) or self._has_checkpoint_for_thread(normalized.get("thread_id", ""))
        status = str(normalized.get("execution_status", "")).strip().lower()
        if "resumable" not in normalized:
            if status in {"interrupted", "external_abort"}:
                normalized["resumable"] = bool(normalized.get("state_path")) or self._has_checkpoint_for_thread(normalized.get("thread_id", ""))
            elif status == "fatal_error":
                normalized["resumable"] = False
            else:
                normalized["resumable"] = False
        if not normalized.get("pending_nodes") and bool(normalized.get("resumable", False)):
            try:
                snapshot = self.get_checkpoint_state(str(normalized.get("thread_id", "")))
                normalized["pending_nodes"] = list(getattr(snapshot, "next", ()) or ())
            except Exception:
                pass
        return normalized

    def _has_skill(self, state: AgentState, keyword: str) -> bool:
        return bool(self._find_skill_name(self._skills_from_state(state), keyword))

    def _skills_from_state(self, state: AgentState) -> list[SkillInfo]:
        skills = state.get("skills", [])
        return [SkillInfo(**item) for item in skills]

    def _serialize_skills(self, skills: list[SkillInfo]) -> list[dict[str, str]]:
        return [asdict(skill) for skill in skills]

    def _copy_stage_results(self, state: AgentState) -> dict[str, Any]:
        return deepcopy(state.get("stage_results", {}))

    def _results_root(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "results")

    def _copy_if_exists(self, src: str, dst: str) -> bool:
        if not src or not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True

    def _copy_dir_if_exists(self, src: str, dst: str) -> bool:
        if not src or not os.path.isdir(src):
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return True

    def _select_best_pragma_dse_result(self, report: dict[str, Any]) -> dict[str, Any]:
        best_candidate_id = report.get("best_candidate_id")
        for item in report.get("results", []) or []:
            if item.get("id") == best_candidate_id and item.get("success"):
                return item
        for item in report.get("results", []) or []:
            if item.get("success"):
                return item
        return {}

    def _best_pragma_dse_score(self, report: dict[str, Any]) -> float | None:
        best = self._select_best_pragma_dse_result(report)
        if not best:
            return None
        try:
            return float((best.get("metrics") or {}).get("score"))
        except (TypeError, ValueError):
            return None

    def _resolved_hardware_context_turn(self, state: AgentState) -> int:
        best_hardware_turn = int(state.get("best_hardware_turn", 0) or 0)
        if best_hardware_turn > 0:
            return best_hardware_turn
        hardware_turn = int(state.get("hardware_turn", 0) or 0)
        if hardware_turn > 0:
            return hardware_turn
        return self._resolved_best_software_turn(state)

    def _analyze_profile_output(self, output: dict[str, Any], code_turn: int, log_file: str) -> str:
        profiling_result = self._generate_llm_prompt(output)
        saved_turn = self.optimized_code_turn
        self.optimized_code_turn = code_turn
        try:
            raw = self._chat_with_history([
                {
                    "role": "system",
                    "content": (
                        "Analyze the profiling results only and identify bottlenecks. "
                        "Do not optimize any source code. "
                        "Prioritize QoR analysis in this order: "
                        "(1) latency / throughput / hotspot loops or modules, "
                        "(2) resource pressure and utilization, "
                        "(3) constraint feasibility signals such as timing_feasible, timing_violations, and ii_violations. "
                        "Always keep latency, II, and resource information central. "
                        "Do not treat slack as an optimization target; use timing fields only to flag feasibility or infeasibility under the fixed clock target. "
                        "If timing_feasible is false or timing_violations are present, explicitly state that the current design is infeasible under the fixed clock target. "
                        "If timing_feasible is true but ii_violations exist, explicitly state that the design is timing-feasible but throughput-limited by II/resource constraints. "
                        "If timing and II are both acceptable, focus on cycle latency and resource bottlenecks. "
                        "Structure the analysis with concise sections for QoR Summary, Constraint Status, Top Bottlenecks, and Optimization Priorities. "
                        "Mark analysis content with <analysis></analysis>."
                    ),
                },
                {
                    "role": "user",
                    "content": profiling_result + "\n Current Design \n" + self.scratchpad.optimized_code[self.optimized_code_turn],
                },
            ])["content"].strip()
        finally:
            self.optimized_code_turn = saved_turn
        self._log_block(log_file, "Profiling analysis output", raw, max_lines=20, max_chars=1400)
        parsed = self._parse_artifacts(raw, log_file)
        return self._generate_llm_prompt(parsed)

    def _refresh_best_qor_context_from_pragma_dse(
        self,
        state: AgentState,
        report: dict[str, Any],
        best_turn: int,
        log_file: str,
    ) -> dict[str, Any]:
        best = self._select_best_pragma_dse_result(report)
        if not best:
            raise ValueError("No successful pragma-DSE candidate was available for QoR refresh")
        report_path = str(best.get("report_path", "")).strip()
        if not report_path or not os.path.exists(report_path):
            run_dir = str(best.get("run_dir", "")).strip()
            report_path = self._find_csynth_report(run_dir) if run_dir else ""
        if not report_path or not os.path.exists(report_path):
            raise FileNotFoundError("Best pragma-DSE csynth.xml was not found for QoR refresh")

        best_candidate_id = str(best.get("id", report.get("best_candidate_id", "unknown")))
        _log_info(f"Current best one: {best_candidate_id} on v{best_turn}, do new profiling.")
        self._append_text(
            log_file,
            f"[INFO] Current best one: {best_candidate_id} on v{best_turn}, do new profiling from {report_path}\n",
        )

        script_dir = os.path.join(self.skills_dir, "profiling", "scripts")
        refreshed_report_path = os.path.join(
            self.scratchpad.run_dir,
            f"hardware_qor_refresh_v{best_turn}_{str(best_candidate_id).replace(os.sep, '_')}.json",
        )
        rc, command_output = self._run_python_script(
            os.path.join(script_dir, "parse_csynth.py"),
            [report_path, "-o", refreshed_report_path],
            log_file,
            "hardware-qor-refresh-parse",
            cwd=self.scratchpad.run_dir,
        )
        if rc != 0 or not os.path.exists(refreshed_report_path):
            raise ValueError("Hardware QoR refresh failed: parse_csynth did not produce JSON output")
        data = self._load_json_file(refreshed_report_path)
        output = self._store_json_artifact("hardware_qor_refresh_report", data, refreshed_report_path, log_file)
        output["analysis"] = "Refreshed QoR summary parsed from the current best pragma-DSE csynth report."
        output["command_result"] = command_output
        profiling_analysis = self._analyze_profile_output(output, best_turn, log_file)
        _log_info(f"Current best one: {best_candidate_id} on v{best_turn}, new profiling completed.")
        self._append_text(
            log_file,
            f"[INFO] Current best one: {best_candidate_id} on v{best_turn}, new profiling completed.\n",
        )
        return {"output": output, "profiling_analysis": profiling_analysis}

    def _score_improved(self, candidate_score: float | None, best_score: float | None, epsilon: float) -> bool:
        if candidate_score is None:
            return False
        if best_score is None:
            return True
        return candidate_score > best_score + epsilon

    def _snapshot_best_hardware_round(self, state: AgentState, report_path: str) -> dict[str, str]:
        run_dir = state.get("run_dir", "")
        snapshots: dict[str, str] = {}
        if run_dir and report_path and os.path.exists(report_path):
            best_report_path = os.path.join(run_dir, "best_pragma_dse_report.json")
            shutil.copy2(report_path, best_report_path)
            snapshots["best_pragma_dse_report"] = best_report_path
        candidates_path = self.scratchpad.stage_artifacts.get("pragma_candidates", "")
        if run_dir and candidates_path and os.path.exists(candidates_path):
            best_candidates_path = os.path.join(run_dir, "best_pragma_candidates.json")
            shutil.copy2(candidates_path, best_candidates_path)
            snapshots["best_pragma_candidates"] = best_candidates_path
        return snapshots

    def _build_hardware_progress_feedback(
        self,
        report: dict[str, Any],
        round_best: dict[str, Any],
        best_score_before: float | None,
        improved: bool,
    ) -> str:
        best_candidate_id = round_best.get("id") or report.get("best_candidate_id") or "unknown"
        round_score = self._best_pragma_dse_score(report)
        search_outcome = str(report.get("search_outcome", "")).strip() or "unknown"
        if improved:
            prior = "none" if best_score_before is None else f"{best_score_before:.4f}"
            current = "unknown" if round_score is None else f"{round_score:.4f}"
            return (
                f"Current best hardware/DSE result improved in attempt {int(report.get('_hardware_attempt', 0) or 0)}: "
                f"candidate {best_candidate_id} with score {current} (previous best {prior}). "
                "Start another kg-rag -> hardware rewrite round only if it can beat this new best QoR."
            )
        current_best = "unknown" if best_score_before is None else f"{best_score_before:.4f}"
        return (
            "Latest hardware optimization round did not improve the current best QoR. "
            f"Search outcome: {search_outcome}; latest best candidate: {best_candidate_id}; current best score: {current_best}."
        )

    def _export_results(self, run_dir: str, final_code_path: str, log_file: str, report_path: str = "", candidates_path: str = "") -> dict[str, str]:
        run_rel_path = self._run_relative_path(run_dir)
        results_dir = os.path.join(self._results_root(), run_rel_path)
        os.makedirs(results_dir, exist_ok=True)
        exported: dict[str, str] = {"results_dir": results_dir}

        resolved_candidates_path = candidates_path or os.path.join(run_dir, "pragma_candidates.json")
        if self._copy_if_exists(resolved_candidates_path, os.path.join(results_dir, "pragma_candidates.json")):
            exported["pragma_candidates_json"] = os.path.join(results_dir, "pragma_candidates.json")

        resolved_report_path = report_path or os.path.join(run_dir, "pragma_dse_report.json")
        if self._copy_if_exists(resolved_report_path, os.path.join(results_dir, "pragma_dse_report.json")):
            exported["pragma_dse_report_json"] = os.path.join(results_dir, "pragma_dse_report.json")

        if final_code_path:
            dst = os.path.join(results_dir, os.path.basename(final_code_path))
            if self._copy_if_exists(final_code_path, dst):
                exported["final_code"] = dst

        if not os.path.exists(resolved_report_path):
            manifest_path = os.path.join(results_dir, "results_manifest.json")
            manifest = {
                "status": "partial",
                "reason": "pragma_dse_report.json not found",
                "run_dir": run_dir,
                "exported": exported,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.write("\n")
            exported["results_manifest"] = manifest_path
            self._append_text(log_file, f"[WARN] Results exported partially to {results_dir}; pragma_dse_report.json not found.\n")
            return exported

        report = self._load_json_file(resolved_report_path)
        best = self._select_best_pragma_dse_result(report)
        if not best:
            manifest_path = os.path.join(results_dir, "results_manifest.json")
            manifest = {
                "status": "partial",
                "reason": "No successful pragma DSE candidate was available for export",
                "run_dir": run_dir,
                "exported": exported,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.write("\n")
            exported["results_manifest"] = manifest_path
            self._append_text(log_file, f"[WARN] Results exported partially to {results_dir}; no successful pragma DSE candidate was found.\n")
            return exported
        candidate_run_dir = str(best.get("run_dir", ""))
        candidate_code_name = os.path.basename(str(report.get("code", final_code_path or "")))

        if candidate_run_dir and os.path.isdir(candidate_run_dir):
            if candidate_code_name:
                src = os.path.join(candidate_run_dir, candidate_code_name)
                dst = os.path.join(results_dir, candidate_code_name)
                if self._copy_if_exists(src, dst):
                    exported["optimized_code"] = dst
            self._copy_if_exists(
                os.path.join(candidate_run_dir, "run.tcl"),
                os.path.join(results_dir, "run.tcl"),
            )
            if os.path.exists(os.path.join(results_dir, "run.tcl")):
                exported["run_tcl"] = os.path.join(results_dir, "run.tcl")
            copied_vitis_log = self._copy_if_exists(
                os.path.join(candidate_run_dir, "vitis_run.log"),
                os.path.join(results_dir, "vitis_run.log"),
            )
            if not copied_vitis_log:
                copied_vitis_log = self._copy_if_exists(
                    os.path.join(candidate_run_dir, "vitis_hls.log"),
                    os.path.join(results_dir, "vitis_run.log"),
                )
            if os.path.exists(os.path.join(results_dir, "vitis_run.log")):
                exported["vitis_run_log"] = os.path.join(results_dir, "vitis_run.log")
            self._copy_if_exists(
                os.path.join(candidate_run_dir, "logs", "hls_run_tcl.log"),
                os.path.join(results_dir, "hls_run_tcl.log"),
            )
            if os.path.exists(os.path.join(results_dir, "hls_run_tcl.log")):
                exported["hls_run_tcl_log"] = os.path.join(results_dir, "hls_run_tcl.log")

            csynth_path = self._find_csynth_report(candidate_run_dir)
            if csynth_path:
                dst = os.path.join(results_dir, "csynth.xml")
                if self._copy_if_exists(csynth_path, dst):
                    exported["csynth_xml"] = dst

            vitis_project_dir = ""
            for entry in sorted(os.listdir(candidate_run_dir)):
                full = os.path.join(candidate_run_dir, entry)
                if os.path.isdir(full) and entry.startswith("pragma_dse_"):
                    vitis_project_dir = full
                    break
            if vitis_project_dir:
                dst = os.path.join(results_dir, os.path.basename(vitis_project_dir))
                if self._copy_dir_if_exists(vitis_project_dir, dst):
                    exported["vitis_project"] = dst

        manifest_path = os.path.join(results_dir, "results_manifest.json")
        manifest = {
            "status": "ok",
            "run_dir": run_dir,
            "best_candidate_id": report.get("best_candidate_id"),
            "candidate_run_dir": candidate_run_dir,
            "exported": exported,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        exported["results_manifest"] = manifest_path
        self._append_text(log_file, f"[INFO] Exported final results to {results_dir}\n")
        return exported

    def _thread_index_path(self) -> str:
        if self.checkpoint_db_path:
            return os.path.join(os.path.dirname(self.checkpoint_db_path), "thread_index.json")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "checkpoints", "thread_index.json")

    def _load_thread_index(self) -> dict[str, Any]:
        path = self._thread_index_path()
        if not os.path.exists(path):
            return {}
        try:
            return self._load_json_file(path)
        except Exception:
            return {}

    def _write_thread_index(self, index: dict[str, Any]) -> None:
        path = self._thread_index_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
            f.write("\n")

    def _clear_thread_persistence(
        self,
        thread_id: str,
        run_dir: str = "",
        state_path: str = "",
        checkpoint_path: str = "",
    ) -> None:
        if not thread_id:
            return

        if hasattr(self.checkpointer, "delete_thread"):
            try:
                self.checkpointer.delete_thread(thread_id)
            except Exception:
                pass

        candidate_paths = {
            state_path,
            checkpoint_path,
            os.path.join(run_dir, "langgraph_state.json") if run_dir else "",
            os.path.join(run_dir, "langgraph_checkpoint.json") if run_dir else "",
        }
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        index = self._load_thread_index()
        if thread_id in index:
            index.pop(thread_id, None)
            self._write_thread_index(index)

    def _update_thread_index(self, state: AgentState) -> None:
        thread_id = state.get("thread_id", "")
        run_dir = state.get("run_dir", "")
        if not thread_id or not run_dir:
            return
        summary = state.get("final_summary", {})
        execution_status = str(summary.get("execution_status", "")).strip().lower()
        state_path = os.path.join(run_dir, "langgraph_state.json")
        checkpoint_path = os.path.join(run_dir, "langgraph_checkpoint.json")
        if execution_status == "completed":
            self._clear_thread_persistence(
                thread_id=thread_id,
                run_dir=run_dir,
                state_path=state_path,
                checkpoint_path=checkpoint_path,
            )
            return

        index = self._load_thread_index()
        index[thread_id] = {
            "run_dir": run_dir,
            "state_path": state_path,
            "checkpoint_path": checkpoint_path,
            "log_file": state.get("log_file", ""),
            "execution_status": summary.get("execution_status", ""),
            "status_reason": summary.get("status_reason", ""),
            "total_tokens_used": summary.get("total_tokens_used", 0),
            "failing_stage": summary.get("failing_stage", ""),
            "resumable": summary.get("resumable", False),
            "process_pid": summary.get("process_pid", 0),
            "pending_nodes": summary.get("pending_nodes", []),
            "checkpoint_db_path": summary.get("checkpoint_db_path", self.checkpoint_db_path),
            "updated_at": int(time.time()),
        }
        self._write_thread_index(index)

    def _find_state_path_for_thread(self, thread_id: str) -> str:
        index = self._load_thread_index()
        entry = index.get(thread_id, {})
        state_path = entry.get("state_path", "")
        if state_path and os.path.exists(state_path):
            return state_path

        out_dir = os.path.abspath(self.out_dir)
        if not os.path.isdir(out_dir):
            return ""
        for root, _, files in os.walk(out_dir):
            if "langgraph_state.json" not in files:
                continue
            candidate = os.path.join(root, "langgraph_state.json")
            try:
                data = self._load_json_file(candidate)
            except Exception:
                continue
            if data.get("thread_id") == thread_id or data.get("final_summary", {}).get("thread_id") == thread_id:
                return candidate
        return ""

    def list_threads(self, include_completed: bool = False) -> list[dict[str, Any]]:
        index = self._load_thread_index()
        items: list[dict[str, Any]] = []
        for thread_id, entry in sorted(index.items(), key=lambda item: item[0]):
            record = {"thread_id": thread_id}
            if isinstance(entry, dict):
                record.update(entry)
            record = self._normalize_thread_record(record)
            if not include_completed and str(record.get("execution_status", "")) == "completed":
                continue
            items.append(record)
        return items

    def get_thread_record(self, thread_id: str) -> dict[str, Any]:
        index = self._load_thread_index()
        entry = index.get(thread_id)
        if isinstance(entry, dict):
            record = {"thread_id": thread_id}
            record.update(entry)
            return self._normalize_thread_record(record)

        state_path = self._find_state_path_for_thread(thread_id)
        if not state_path:
            raise ValueError(f"No thread record found for thread_id={thread_id}")
        data = self._load_json_file(state_path)
        summary = data.get("final_summary", {})
        return self._normalize_thread_record({
            "thread_id": thread_id,
            "run_dir": data.get("run_dir", ""),
            "state_path": state_path,
            "log_file": data.get("log_file", ""),
            "execution_status": summary.get("execution_status", ""),
            "status_reason": summary.get("status_reason", ""),
            "total_tokens_used": summary.get("total_tokens_used", 0),
            "failing_stage": summary.get("failing_stage", ""),
            "resumable": summary.get("resumable", False),
            "process_pid": summary.get("process_pid", 0),
            "pending_nodes": summary.get("pending_nodes", []),
            "checkpoint_db_path": summary.get("checkpoint_db_path", ""),
            "updated_at": "",
        })

    def _has_checkpoint_for_thread(self, thread_id: str) -> bool:
        try:
            snapshot = self.get_checkpoint_state(thread_id)
        except Exception:
            return False
        values = dict(getattr(snapshot, "values", {}) or {})
        return bool(values or getattr(snapshot, "next", ()))

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        record: dict[str, Any] = {}
        try:
            record = self.get_thread_record(thread_id)
        except Exception:
            record = {"thread_id": thread_id}

        if hasattr(self.checkpointer, "delete_thread"):
            try:
                self.checkpointer.delete_thread(thread_id)
            except Exception:
                pass

        removed_files: list[str] = []
        for key in ("state_path",):
            path = record.get(key, "")
            if path and os.path.exists(path):
                os.remove(path)
                removed_files.append(path)

        run_dir = record.get("run_dir", "")
        for filename in ("langgraph_state.json", "langgraph_checkpoint.json"):
            path = os.path.join(run_dir, filename) if run_dir else ""
            if path and os.path.exists(path) and path not in removed_files:
                os.remove(path)
                removed_files.append(path)

        index = self._load_thread_index()
        index.pop(thread_id, None)
        self._write_thread_index(index)

        return {
            "thread_id": thread_id,
            "removed_files": removed_files,
            "run_dir": run_dir,
        }

    def prune_threads(self) -> list[dict[str, Any]]:
        index = self._load_thread_index()
        removed: list[dict[str, Any]] = []
        for thread_id, entry in list(index.items()):
            state_path = entry.get("state_path", "") if isinstance(entry, dict) else ""
            state_exists = bool(state_path and os.path.exists(state_path))
            checkpoint_exists = self._has_checkpoint_for_thread(thread_id)
            if state_exists or checkpoint_exists:
                continue
            removed.append(
                {
                    "thread_id": thread_id,
                    "run_dir": entry.get("run_dir", "") if isinstance(entry, dict) else "",
                    "state_path": state_path,
                }
            )
            index.pop(thread_id, None)
        if removed:
            self._write_thread_index(index)
        return removed

    def _merge_state(self, base: AgentState, updates: AgentState) -> AgentState:
        merged = deepcopy(base)
        merged.update(updates)
        return merged

    def _next_node_after(self, node_name: str, state: AgentState) -> str | None:
        if node_name == "initialize_run":
            return "discover_skills"
        if node_name == "discover_skills":
            return self._route_after_discover(state)
        if node_name == "profiling":
            return self._route_after_profiling(state)
        if node_name == "kg_rag":
            return "hardware_rewrite"
        if node_name == "rewrite_guidance":
            return "hardware_rewrite" if state.get("best_software_turn") is not None else "software_rewrite"
        if node_name == "software_rewrite":
            return self._route_after_software_rewrite(state)
        if node_name == "hardware_rewrite":
            return self._route_after_hardware_rewrite(state)
        if node_name == "pragma_tuning":
            return self._route_after_pragma_tuning(state)
        if node_name == "pragma_dse":
            return self._route_after_pragma_dse(state)
        if node_name == "finalize":
            return None
        raise ValueError(f"Unknown node name for resume: {node_name}")

    def _capture_runtime_state(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "analysis_turn": self.analysis_turn,
            "command_turn": self.command_turn,
            "reference_turn": self.reference_turn,
            "skill_turn": self.skill_turn,
            "optimized_code_turn": self.optimized_code_turn,
            "scratchpad": asdict(self.scratchpad),
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "conversation_history": deepcopy(self.conversation_history),
            "total_tokens_used": int(getattr(self.client, "total_tokens_used", 0) or 0),
        }

    def _restore_runtime_from_state(self, state: AgentState) -> None:
        self.input_mode = self._normalize_input_mode(state.get("input_mode") or state.get("scratchpad", {}).get("input_mode", "plain_c"))
        self.analysis_turn = int(state.get("analysis_turn", 0))
        self.command_turn = int(state.get("command_turn", 0))
        self.reference_turn = int(state.get("reference_turn", 0))
        self.skill_turn = int(state.get("skill_turn", 0))
        self.optimized_code_turn = int(state.get("optimized_code_turn", 0))
        self.client.total_tokens_used = int(state.get("total_tokens_used", 0) or 0)

        scratchpad_state = state.get("scratchpad")
        if scratchpad_state:
            self.scratchpad = Scratchpad(**deepcopy(scratchpad_state))
        else:
            self.scratchpad = Scratchpad(
                goal="",
                fpga="",
                target_frequency_mhz="",
                top_function_name=self.top_function_name,
                run_dir="",
                stage_artifacts={},
                analysis={},
                command={},
                reference={},
                json_artifact={},
                skill={},
                optimized_code={},
                optimized_code_file={},
                input_mode=self.input_mode,
            )

        candidates_state = state.get("candidates", [])
        self.candidates = [Candidate(**deepcopy(candidate)) for candidate in candidates_state]
        self.conversation_history = deepcopy(state.get("conversation_history", []))

    def _emit_state(self, **updates: Any) -> AgentState:
        payload = self._capture_runtime_state()
        payload.update(updates)
        return payload

    def _normalize_input_mode(self, value: Any) -> str:
        return "hls_native" if str(value).strip().lower() == "hls_native" else "plain_c"

    def _input_mode(self, state: AgentState) -> str:
        return self._normalize_input_mode(state.get("input_mode") or getattr(self.scratchpad, "input_mode", self.input_mode))

    def _write_state_artifact(self, state: AgentState, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")

    def _default_status_reason(self, execution_status: str) -> str:
        mapping = {
            "completed": "run_completed",
            "interrupted": "interrupted",
            "running": "active_run",
            "fatal_error": "fatal_error",
            "external_abort": "process_terminated_unexpectedly",
        }
        return mapping.get(str(execution_status).strip().lower(), "")

    def _default_resumable(self, execution_status: str) -> bool:
        return str(execution_status).strip().lower() in {"interrupted", "external_abort"}

    def _persist_running_thread(self, state: AgentState, pending_nodes: list[str]) -> None:
        thread_id = str(state.get("thread_id", "") or "")
        run_dir = str(state.get("run_dir", "") or "")
        if not thread_id or not run_dir:
            return
        self._finalize_state_only_artifacts(
            deepcopy(state),
            execution_status="running",
            pending_nodes=pending_nodes,
            status_reason="active_run",
            resumable=False,
            process_pid=os.getpid(),
        )

    def _snapshot_to_json(self, snapshot: Any) -> dict[str, Any]:
        return {
            "values": getattr(snapshot, "values", {}),
            "next": list(getattr(snapshot, "next", ()) or ()),
            "config": getattr(snapshot, "config", {}),
            "metadata": getattr(snapshot, "metadata", {}),
            "created_at": getattr(snapshot, "created_at", ""),
            "parent_config": getattr(snapshot, "parent_config", {}),
            "interrupts": [repr(item) for item in getattr(snapshot, "interrupts", ()) or ()],
            "tasks": [repr(item) for item in getattr(snapshot, "tasks", ()) or ()],
        }

    def _record_stage_success(
        self,
        state: AgentState,
        stage_name: str,
        output: dict[str, Any],
        before_artifacts: dict[str, str],
    ) -> dict[str, Any]:
        stage_results = self._copy_stage_results(state)
        after_artifacts = dict(self.scratchpad.stage_artifacts)
        new_artifacts = {
            label: path
            for label, path in after_artifacts.items()
            if before_artifacts.get(label) != path
        }
        stage_results[stage_name] = {
            "status": "completed",
            "analysis": output.get("analysis", ""),
            "artifacts": new_artifacts,
            "json_artifacts": output.get("json_artifacts", []),
            "command_result_excerpt": output.get("command_result", "")[:2000],
            "optimized_code_turn": self.optimized_code_turn,
        }
        return stage_results

    def _record_stage_failure(
        self,
        state: AgentState,
        stage_name: str,
        error_message: str,
    ) -> dict[str, Any]:
        stage_results = self._copy_stage_results(state)
        stage_results[stage_name] = {
            "status": "failed",
            "error": error_message,
            "optimized_code_turn": self.optimized_code_turn,
        }
        return stage_results

    def _record_stage_retry(
        self,
        state: AgentState,
        stage_name: str,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage_results = self._copy_stage_results(state)
        payload = {
            "status": "retry",
            "reason": reason,
            "optimized_code_turn": self.optimized_code_turn,
        }
        if extra:
            payload.update(extra)
        stage_results[stage_name] = payload
        return stage_results

    def _finalize_state_only_artifacts(
        self,
        final_state: AgentState,
        execution_status: str,
        pending_nodes: list[str],
        status_reason: str = "",
        failing_stage: str = "",
        resumable: bool | None = None,
        process_pid: int = 0,
    ) -> AgentState:
        thread_id = final_state.get("thread_id", "")
        run_dir = final_state.get("run_dir", "")
        if not thread_id or not run_dir:
            return final_state

        self._restore_runtime_from_state(final_state)
        state_path = os.path.join(run_dir, "langgraph_state.json")
        self.scratchpad.stage_artifacts["langgraph_state"] = state_path
        if self.checkpoint_db_path:
            self.scratchpad.stage_artifacts.setdefault("langgraph_checkpoint_db", self.checkpoint_db_path)

        final_summary = deepcopy(final_state.get("final_summary", {}))
        if not final_summary:
            final_summary = {
                "design_name": final_state.get("design_name", ""),
                "input_mode": final_state.get("input_mode", self.input_mode),
                "run_dir": run_dir,
                "log_file": final_state.get("log_file", ""),
                "final_code_turn": self.optimized_code_turn,
                "final_code_path": self.scratchpad.optimized_code_file.get(self.optimized_code_turn, ""),
                "best_software_turn": final_state.get("best_software_turn"),
                "hardware_turn": final_state.get("hardware_turn"),
                "stage_results": deepcopy(final_state.get("stage_results", {})),
                "errors": list(final_state.get("errors", [])),
            }
        final_summary["thread_id"] = thread_id
        final_summary["checkpoint_backend"] = self.checkpoint_backend
        final_summary["checkpoint_db_path"] = self.checkpoint_db_path
        final_summary["execution_status"] = execution_status
        final_summary["pending_nodes"] = list(pending_nodes)
        final_summary["status_reason"] = status_reason or self._default_status_reason(execution_status)
        final_summary["failing_stage"] = failing_stage
        final_summary["resumable"] = self._default_resumable(execution_status) if resumable is None else bool(resumable)
        final_summary["process_pid"] = int(process_pid or (os.getpid() if execution_status == "running" else 0))
        final_summary["interrupt_after"] = list(self.interrupt_after)
        final_summary["stage_artifacts"] = dict(self.scratchpad.stage_artifacts)
        final_summary["total_tokens_used"] = int(getattr(self.client, "total_tokens_used", 0) or 0)

        merged = deepcopy(final_state)
        merged.update(self._capture_runtime_state())
        merged["thread_id"] = thread_id
        merged["final_summary"] = final_summary
        self._write_state_artifact(merged, state_path)
        self._update_thread_index(merged)
        return merged

    def _resume_from_serialized_state(self, state: AgentState) -> AgentState:
        summary = deepcopy(state.get("final_summary", {}))
        pending_nodes = list(summary.get("pending_nodes", []))
        if not pending_nodes:
            return self._finalize_state_only_artifacts(
                state,
                execution_status=summary.get("execution_status", "completed"),
                pending_nodes=[],
            )

        current_state = deepcopy(state)
        while pending_nodes:
            node_name = pending_nodes[0]
            handler = getattr(self, f"_node_{node_name}")
            updates = handler(current_state)
            current_state = self._merge_state(current_state, updates)
            next_node = self._next_node_after(node_name, current_state)
            if next_node is None:
                return self._finalize_state_only_artifacts(
                    current_state,
                    execution_status="completed",
                    pending_nodes=[],
                )
            if node_name in self.interrupt_after:
                return self._finalize_state_only_artifacts(
                    current_state,
                    execution_status="interrupted",
                    pending_nodes=[next_node],
                )
            pending_nodes = [next_node]

        return self._finalize_state_only_artifacts(
            current_state,
            execution_status="completed",
            pending_nodes=[],
        )

    def _finalize_checkpoint_artifacts(self, final_state: AgentState) -> AgentState:
        thread_id = final_state.get("thread_id", "")
        run_dir = final_state.get("run_dir", "")
        if not thread_id or not run_dir:
            return final_state

        self._restore_runtime_from_state(final_state)
        snapshot = self.get_checkpoint_state(thread_id)
        pending_nodes = list(getattr(snapshot, "next", ()) or ())
        execution_status = "interrupted" if pending_nodes else "completed"
        checkpoint_path = os.path.join(run_dir, "langgraph_checkpoint.json")
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self._snapshot_to_json(snapshot), f, indent=2)
            f.write("\n")

        self.scratchpad.stage_artifacts["langgraph_checkpoint"] = checkpoint_path
        if self.checkpoint_db_path:
            self.scratchpad.stage_artifacts["langgraph_checkpoint_db"] = self.checkpoint_db_path
        state_path = os.path.join(run_dir, "langgraph_state.json")
        self.scratchpad.stage_artifacts["langgraph_state"] = state_path
        final_summary = deepcopy(final_state.get("final_summary", {}))
        if not final_summary:
            final_summary = {
                "design_name": final_state.get("design_name", ""),
                "input_mode": final_state.get("input_mode", self.input_mode),
                "run_dir": run_dir,
                "log_file": final_state.get("log_file", ""),
                "final_code_turn": self.optimized_code_turn,
                "final_code_path": self.scratchpad.optimized_code_file.get(self.optimized_code_turn, ""),
                "best_software_turn": final_state.get("best_software_turn"),
                "hardware_turn": final_state.get("hardware_turn"),
                "stage_results": deepcopy(final_state.get("stage_results", {})),
                "errors": list(final_state.get("errors", [])),
            }
        final_summary["thread_id"] = thread_id
        final_summary["checkpoint_backend"] = self.checkpoint_backend
        final_summary["checkpoint_db_path"] = self.checkpoint_db_path
        final_summary["execution_status"] = execution_status
        final_summary["pending_nodes"] = pending_nodes
        final_summary["status_reason"] = self._default_status_reason(execution_status)
        final_summary["failing_stage"] = ""
        final_summary["resumable"] = self._default_resumable(execution_status)
        final_summary["process_pid"] = 0
        final_summary["interrupt_after"] = list(self.interrupt_after)
        final_summary["stage_artifacts"] = dict(self.scratchpad.stage_artifacts)
        final_summary["total_tokens_used"] = int(getattr(self.client, "total_tokens_used", 0) or 0)

        merged = deepcopy(final_state)
        merged.update(self._capture_runtime_state())
        merged["thread_id"] = thread_id
        merged["final_summary"] = final_summary
        self._write_state_artifact(merged, state_path)
        self._update_thread_index(merged)
        return merged

    def _node_initialize_run(self, state: AgentState) -> AgentState:
        self._reset_runtime_state()
        code_path = self.code_path
        req_path = self.req_path
        self.load_requirement_to_scratchpad(req_path)
        out_dir = os.path.abspath(self.out_dir)

        if not os.path.exists(code_path):
            raise FileNotFoundError(f"[ERROR] Code path not found: {code_path}")
        if not os.path.exists(req_path):
            raise FileNotFoundError(f"[ERROR] Requirement path not found: {req_path}")
        if not os.path.isfile(code_path):
            raise ValueError(f"[ERROR] Code path must be a file: {code_path}")

        run_id = time.strftime("%Y%m%d_%H%M%S")
        design_name = self._derive_design_name(code_path, os.path.dirname(code_path))
        model_dir_name = self._current_model_dir_name()
        model_out_dir = os.path.join(out_dir, model_dir_name)
        run_dir = os.path.join(model_out_dir, self._run_dir_name(design_name, run_id))
        os.makedirs(run_dir, exist_ok=True)
        log_file = os.path.join(run_dir, f"log_{run_id}.log")

        self.scratchpad.run_dir = run_dir
        self._append_text(log_file, f"[INFO] Run directory created: {run_dir}\n")
        self._append_text(log_file, f"[INFO] Model directory: {model_dir_name}\n")
        self._append_text(log_file, f"[INFO] Design Name: {design_name}\n")
        self._append_text(log_file, f"[INFO] Input mode: {self.input_mode}\n")

        _log_phase(f"Run: {design_name}")
        _log_info(f"Run directory: {_c(run_dir, '1;37')}")
        _log_info(f"Optimization mode: {_c(self.input_mode, '1;37')}")

        code = self._read_text(code_path)
        self.scratchpad.optimized_code[self.optimized_code_turn] = code
        self._create_code_file(run_dir=run_dir, design_name=design_name, log_file=log_file)

        emitted = self._emit_state(
            design_name=design_name,
            run_id=run_id,
            thread_id=state.get("thread_id", ""),
            run_dir=run_dir,
            log_file=log_file,
            input_mode=self.input_mode,
            errors=[],
            hardware_attempt=0,
            hardware_feedback="",
            pragma_candidate_count=0,
            pragma_dse_success=False,
            hardware_loop_continue=False,
            best_hardware_turn=0,
            best_pragma_dse_score=0.0,
            best_pragma_dse_candidate_id="",
            best_pragma_dse_report_path="",
            best_pragma_candidates_path="",
            stage_results=self._copy_stage_results(state),
        )
        self._persist_running_thread(emitted, ["discover_skills"])
        return emitted

    def _node_discover_skills(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        skills = self.discover_skills()
        available_skills_xml = self.skills_to_prompt(skills)
        skill_names = ", ".join(skill.name for skill in skills) if skills else "<none>"
        self._append_text(log_file, f"[INFO] Available skills: {skill_names}\n")
        return self._emit_state(
            skills=self._serialize_skills(skills),
            available_skills_xml=available_skills_xml,
        )

    def _route_after_discover(self, state: AgentState) -> str:
        if self._has_skill(state, "profiling") and self._check_vitis_available():
            return "profiling"
        if self._input_mode(state) == "hls_native":
            return "kg_rag"
        return "software_rewrite"

    def _node_profiling(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        skills = self._skills_from_state(state)
        profiling_skill_name = self._find_skill_name(skills, "profiling")
        if not profiling_skill_name or not self._check_vitis_available():
            return self._emit_state(profiling_analysis="", stage_results=self._copy_stage_results(state))

        _log_phase("Phase: Profiling")
        _log_skill(f"Using skill: {_c(profiling_skill_name, '1;37')}")
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            output = self._run_direct_profiling(log_file)
            stage_results = self._record_stage_success(state, "profiling", output, before_artifacts)
            return self._emit_state(
                profiling_analysis=self._analyze_profile_output(output, self.optimized_code_turn, log_file),
                stage_results=stage_results,
            )
        except Exception as exc:
            updates = self._append_error(state, "Profiling phase", exc)
            updates["profiling_analysis"] = ""
            updates["stage_results"] = self._record_stage_failure(state, "profiling", str(exc))
            return self._emit_state(**updates)

    def _route_after_profiling(self, state: AgentState) -> str:
        if self._input_mode(state) == "hls_native":
            return "kg_rag"
        return "software_rewrite"

    def _resolved_best_software_turn(self, state: AgentState) -> int:
        if state.get("best_software_turn") is not None:
            return int(state["best_software_turn"])
        return int(self.optimized_code_turn)

    def _run_rewrite_guidance(self, state: AgentState, log_file: str, mode: str) -> dict[str, Any]:
        skills = self._skills_from_state(state)
        rewrite_skill_name = self._find_skill_name(skills, "rewrite")
        if not rewrite_skill_name:
            return {}

        if mode == "hardware":
            guidance_turn = self._resolved_hardware_context_turn(state)
            design_label = "Current best hardware design" if guidance_turn != self._resolved_best_software_turn(state) else "Validated software design"
            context = (
                "This rewrite phase is analysis-only and targets the hardware optimization loop after KG-RAG. "
                "Do not emit <optimized_code> blocks. Focus on legal HLS-oriented structural changes, baseline pragma placement, "
                "and tunable pragma parameters for later pragma-tuning/pragma-dse. "
                "When the next move should be a structural dataflow circuit optimization, choose one primary structural hypothesis "
                "and explain the intended stage graph, bank factors, and local array changes before suggesting more pragmas. "
                "Use the profiling summary, KG-RAG output, and the latest hardware-loop feedback when present. "
                "Do not emit <command> blocks.\n\n"
                + state.get("profiling_analysis", "")
                + state.get("rag_analysis", "")
                + (
                    "\nLatest hardware-loop feedback:\n" + str(state.get("hardware_feedback", "")).strip() + "\n"
                    if str(state.get("hardware_feedback", "")).strip()
                    else ""
                )
                + f"\n{design_label}:\n"
                + self.scratchpad.optimized_code.get(guidance_turn, "")
            )
        else:
            guidance_turn = self._resolved_best_software_turn(state)
            context = (
                "This rewrite phase is analysis-only and targets the software rewrite/C-sim verification loop only. "
                "Do not emit <optimized_code> blocks. Focus on plain C/C++ transformations that remain compatible with AMD Vitis C-sim, "
                "preserve the top-function signature, and avoid HLS-only constructs. Do not emit <command> blocks.\n\n"
                + state.get("profiling_analysis", "")
                + "\nCurrent software design:\n"
                + self.scratchpad.optimized_code.get(guidance_turn, "")
            )

        saved_turn = self.optimized_code_turn
        self.optimized_code_turn = guidance_turn
        try:
            skill_messages = self.build_skill_prompt(skills, rewrite_skill_name)
        finally:
            self.optimized_code_turn = saved_turn
        skill_messages.append({"role": "user", "content": context})
        result = self._chat_with_history(skill_messages)
        raw_content = result["content"]
        self._log_block(log_file, f"Rewrite guidance ({mode}) raw output", raw_content, max_lines=20, max_chars=1400)
        return self._parse_artifacts(raw_content, log_file, allow_optimized_code=False)

    def _node_kg_rag(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        skills = self._skills_from_state(state)
        available_skills_xml = state.get("available_skills_xml", "")
        rag_skill_name = self._find_skill_name(skills, "kg-rag") or self._find_skill_name(skills, "rag")
        if not rag_skill_name:
            return self._emit_state(rag_analysis="")

        _log_phase("Phase: KG-RAG")
        _log_skill(f"Using skill: {_c(rag_skill_name, '1;37')}")
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            context_turn = self._resolved_hardware_context_turn(state)
            self.optimized_code_turn = context_turn
            feedback = str(state.get("hardware_feedback", "")).strip()
            raw_vitis_diagnostics = self._collect_recent_vitis_diagnostics()
            raw_vitis_block = ""
            if raw_vitis_diagnostics:
                raw_vitis_block = "\nRaw Vitis diagnostics (verbatim when available):\n" + "\n".join(
                    f"- {item}" for item in raw_vitis_diagnostics
                ) + "\n"
            rag_output = self._run_optional_skill(
                skills=skills,
                available_skills_xml=available_skills_xml,
                preferred_skill=rag_skill_name,
                user_prompt=(
                    "Build/use the Vitis guide RAG and retrieve hardware-oriented optimization strategies for the next "
                    "hardware rewrite / pragma-tuning / pragma-dse iteration.\n"
                    "Base the retrieval question on the current best design already provided in ScratchpadInfo/Current_Code, "
                    "the target board/clock/top function, and raw Vitis diagnostics when available.\n"
                    "Do not ask a generic tutorial-style question. Ask exactly one targeted question that is grounded in the current code shape "
                    "and the observed Vitis failure or bottleneck signature. Seek legal HLS transformations and pragma-placement rules supported "
                    "by retrieved evidence, not prior assumptions.\n\n"
                    + state.get("profiling_analysis", "")
                    + raw_vitis_block
                    + (
                        "\nLatest hardware-loop feedback summary:\n" + feedback + "\n"
                        if feedback
                        else ""
                    )
                    + "\nQuestion construction requirements:\n"
                    + "- Analyze the current best design from ScratchpadInfo/Current_Code before forming the query.\n"
                    + "- Include the exact raw Vitis error text verbatim when available.\n"
                    + "- Tie the question to the actual loop/function/memory pattern in the current design.\n"
                    + "- Ask for legal code patterns, pragma placement rules, and fix candidates relevant to the observed issue.\n"
                    + "- Prefer one precise retrieval query over broad optimization brainstorming.\n"
                ),
                log_file=log_file,
                stage_label="kg-rag",
            )
            stage_results = self._record_stage_success(state, "kg_rag", rag_output, before_artifacts)
            return self._emit_state(rag_analysis=self._generate_llm_prompt(rag_output), stage_results=stage_results)
        except Exception as exc:
            updates = self._append_error(state, "KG-RAG phase", exc)
            updates["rag_analysis"] = ""
            updates["stage_results"] = self._record_stage_failure(state, "kg_rag", str(exc))
            return self._emit_state(**updates)

    def _node_rewrite_guidance(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        if not self._find_skill_name(self._skills_from_state(state), "rewrite"):
            self._append_text(log_file, "[WARN] Rewrite skill not found, skipping rewrite phase.\n")
            _log_warn("Rewrite skill not found, skipping.")
            return self._emit_state(rewrite_analysis="")

        mode = "hardware" if state.get("best_software_turn") is not None else "software"
        _log_phase(f"Phase: Rewrite Guidance ({mode})")
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            output = self._run_rewrite_guidance(state, log_file, mode=mode)
            stage_results = self._record_stage_success(state, "rewrite_guidance", output, before_artifacts)
            return self._emit_state(rewrite_analysis=self._generate_llm_prompt(output), stage_results=stage_results)
        except Exception as exc:
            updates = self._append_error(state, "Rewrite phase", exc)
            updates["rewrite_analysis"] = ""
            updates["stage_results"] = self._record_stage_failure(state, "rewrite_guidance", str(exc))
            return self._emit_state(**updates)

    def _node_software_rewrite(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        skills = self._skills_from_state(state)
        profiling_analysis = state.get("profiling_analysis", "")
        verification_skill_name = self._find_skill_name(skills, "csim-verification")
        rewrite_analysis = ""
        updates: dict[str, Any] = {}
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            try:
                rewrite_guidance = self._run_rewrite_guidance(state, log_file, mode="software")
                rewrite_analysis = self._generate_llm_prompt(rewrite_guidance)
            except Exception as exc:
                updates = self._append_error(state, "Software rewrite guidance", exc)
                rewrite_analysis = ""
            selected_software_turn = self._run_software_rewrite_loop(
                log_file=log_file,
                profiling_analysis=profiling_analysis,
                rag_analysis="",
                rewrite_analysis=rewrite_analysis,
                verification_skill_name=verification_skill_name,
            )
            self.optimized_code_turn = selected_software_turn
            selected_candidate = next((item for item in reversed(self.candidates) if item.turn == selected_software_turn), None)
            summary = f"Selected software candidate v{selected_software_turn} after iterative rewrite and Vitis C-sim verification."
            if selected_candidate is not None and selected_candidate.stage == "seed":
                summary += " No C-sim-equivalent rewrite candidate was found, so the baseline software version was kept."
            stage_results = self._record_stage_success(
                state,
                "software_rewrite",
                {
                    "analysis": summary,
                    "json_artifacts": [],
                    "command_result": "",
                },
                before_artifacts,
            )
        except Exception as exc:
            failure_updates = self._append_error(state, "Software rewrite", exc)
            failure_updates["stage_results"] = self._record_stage_failure(state, "software_rewrite", str(exc))
            return self._emit_state(**failure_updates)

        return self._emit_state(
            best_software_turn=self.optimized_code_turn,
            rewrite_analysis=rewrite_analysis,
            stage_results=stage_results,
            **updates,
        )

    def _route_after_software_rewrite(self, state: AgentState) -> str:
        if self._has_skill(state, "kg-rag") or self._has_skill(state, "rag"):
            return "kg_rag"
        if self._has_skill(state, "pragma-tuning") or self._has_skill(state, "pragma-dse") or self._find_skill_name(
            self._skills_from_state(state), "rewrite"
        ):
            return "hardware_rewrite"
        return "finalize"

    def _node_hardware_rewrite(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        base_turn = self._resolved_hardware_context_turn(state)
        attempt = int(state.get("hardware_attempt", 0) or 0) + 1
        rewrite_analysis = state.get("rewrite_analysis", "")
        updates: dict[str, Any] = {}
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            try:
                rewrite_guidance = self._run_rewrite_guidance(state, log_file, mode="hardware")
                rewrite_analysis = self._generate_llm_prompt(rewrite_guidance)
            except Exception as exc:
                updates = self._append_error(state, "Hardware rewrite guidance", exc)
            result = self._run_hardware_rewrite(
                log_file,
                state.get("profiling_analysis", ""),
                state.get("rag_analysis", ""),
                rewrite_analysis,
                base_turn=base_turn,
                feedback=state.get("hardware_feedback", ""),
                attempt_idx=attempt,
            )
            hardware_turn = result.get("hardware_turn")
            if hardware_turn is not None:
                self.optimized_code_turn = hardware_turn
                stage_results = self._record_stage_success(
                    state,
                    "hardware_rewrite",
                    {
                        "analysis": "Hardware rewrite produced a structurally valid, Vitis C-sim-equivalent HLS-oriented variant.",
                        "json_artifacts": [],
                        "command_result": "",
                    },
                    before_artifacts,
                )
                return self._emit_state(
                    hardware_turn=hardware_turn,
                    final_code_turn=hardware_turn,
                    hardware_attempt=attempt,
                    hardware_feedback="",
                    pragma_candidate_count=0,
                    pragma_dse_success=False,
                    hardware_loop_continue=False,
                    stage_results=stage_results,
                    rewrite_analysis=rewrite_analysis,
                    **updates,
                )
            self.optimized_code_turn = base_turn
            feedback = result.get("feedback", "Hardware rewrite produced no usable HLS variant.")
            stage_results = self._record_stage_retry(
                state,
                "hardware_rewrite",
                feedback,
                {"attempt": attempt, "max_attempts": self._max_hardware_opt_rounds()},
            )
            return self._emit_state(
                hardware_turn=None,
                final_code_turn=base_turn,
                hardware_attempt=attempt,
                hardware_feedback=feedback,
                pragma_candidate_count=0,
                pragma_dse_success=False,
                hardware_loop_continue=False,
                stage_results=stage_results,
                rewrite_analysis=rewrite_analysis,
                **updates,
            )
        except Exception as exc:
            updates = self._append_error(state, "Hardware rewrite", exc)
            self.optimized_code_turn = base_turn
            updates["final_code_turn"] = self.optimized_code_turn
            updates["stage_results"] = self._record_stage_failure(state, "hardware_rewrite", str(exc))
            return self._emit_state(**updates)

    def _route_after_hardware_rewrite(self, state: AgentState) -> str:
        # TODO: Add heuristic early-stop / retry policies for the hardware loop, e.g.
        # stop when the same structural failure signature repeats across rounds, or
        # retry immediately when the new hardware rewrite changes code shape materially.
        if not state.get("hardware_turn"):
            if state.get("hardware_feedback") and int(state.get("hardware_attempt", 0) or 0) < self._max_hardware_opt_rounds():
                return "kg_rag" if self._has_skill(state, "kg-rag") or self._has_skill(state, "rag") else "hardware_rewrite"
            return "finalize"
        if self._has_skill(state, "pragma-tuning"):
            return "pragma_tuning"
        if self._has_skill(state, "pragma-dse") and self._check_vitis_available():
            return "pragma_dse"
        return "finalize"

    def _node_pragma_tuning(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        skills = self._skills_from_state(state)
        pragma_tuning_skill_name = self._find_skill_name(skills, "pragma-tuning")
        if not pragma_tuning_skill_name:
            return self._emit_state(stage_results=self._copy_stage_results(state))

        _log_phase("Phase: Pragma Tuning")
        _log_skill(f"Using skill: {_c(pragma_tuning_skill_name, '1;37')}")
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            output = self._run_direct_pragma_tuning(log_file)
            candidate_path = self.scratchpad.stage_artifacts.get("pragma_candidates", "")
            candidate_payload = self._load_json_file(candidate_path) if candidate_path and os.path.exists(candidate_path) else {}
            candidate_count = int(candidate_payload.get("candidate_count", 0) or 0) if candidate_payload else 0
            if candidate_count == 0:
                feedback = self._summarize_pragma_tuning_dead_end(candidate_payload)
                self.optimized_code_turn = self._resolved_best_software_turn(state)
                stage_results = self._record_stage_retry(
                    state,
                    "pragma_tuning",
                    feedback,
                    {"candidate_count": 0, "attempt": state.get("hardware_attempt", 0)},
                )
                return self._emit_state(
                    hardware_turn=None,
                    final_code_turn=self.optimized_code_turn,
                    hardware_feedback=feedback,
                    pragma_candidate_count=0,
                    pragma_dse_success=False,
                    hardware_loop_continue=False,
                    stage_results=stage_results,
                )
            stage_results = self._record_stage_success(state, "pragma_tuning", output, before_artifacts)
            return self._emit_state(
                stage_results=stage_results,
                pragma_candidate_count=candidate_count,
                hardware_feedback="",
                pragma_dse_success=False,
                hardware_loop_continue=False,
            )
        except Exception as exc:
            updates = self._append_error(state, "Pragma tuning", exc)
            updates["stage_results"] = self._record_stage_failure(state, "pragma_tuning", str(exc))
            return self._emit_state(**updates)

    def _route_after_pragma_tuning(self, state: AgentState) -> str:
        # TODO: Add heuristic trigger rules beyond candidate_count==0, e.g.
        # re-enter hardware rewrite when pragma-tuning only finds high-risk / low-value
        # candidates or when the remaining search space is too narrow to justify DSE.
        if int(state.get("pragma_candidate_count", 0) or 0) <= 0:
            if state.get("hardware_feedback") and int(state.get("hardware_attempt", 0) or 0) < self._max_hardware_opt_rounds():
                return "kg_rag" if self._has_skill(state, "kg-rag") or self._has_skill(state, "rag") else "hardware_rewrite"
            return "finalize"
        if self._has_skill(state, "pragma-dse") and self._check_vitis_available():
            return "pragma_dse"
        return "finalize"

    def _node_pragma_dse(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        skills = self._skills_from_state(state)
        pragma_dse_skill_name = self._find_skill_name(skills, "pragma-dse")
        if not pragma_dse_skill_name:
            return self._emit_state(stage_results=self._copy_stage_results(state))
        if not self._check_vitis_available():
            self._append_text(log_file, "[WARN] Skipping pragma DSE because Vitis HLS is not available on PATH.\n")
            _log_warn("Skipping pragma DSE because Vitis HLS is not available.")
            return self._emit_state(stage_results=self._copy_stage_results(state))

        _log_phase("Phase: Pragma DSE")
        _log_skill(f"Using skill: {_c(pragma_dse_skill_name, '1;37')}")
        try:
            before_artifacts = dict(self.scratchpad.stage_artifacts)
            output = self._run_direct_pragma_dse(
                log_file,
                hardware_attempt=int(state.get("hardware_attempt", 0) or 0),
            )
            report_path = self.scratchpad.stage_artifacts.get("pragma_dse_report", "")
            report = self._load_json_file(report_path) if report_path and os.path.exists(report_path) else {}
            hardware_attempt = int(state.get("hardware_attempt", 0) or 0)
            score_epsilon = 1e-6
            try:
                score_epsilon = float(report.get("score_epsilon", score_epsilon) or score_epsilon)
            except (TypeError, ValueError):
                score_epsilon = 1e-6
            report["_hardware_attempt"] = hardware_attempt
            round_best = self._select_best_pragma_dse_result(report)
            round_best_score = self._best_pragma_dse_score(report)
            successful_count = int(report.get("successful_candidate_count", 0) or 0) if report else 0
            baseline_success = bool(report.get("baseline_success")) if report else False
            best_score_before: float | None = None
            if state.get("best_pragma_dse_report_path"):
                try:
                    best_score_before = float(state.get("best_pragma_dse_score", 0.0))
                except (TypeError, ValueError):
                    best_score_before = None
            improved = self._score_improved(round_best_score, best_score_before, score_epsilon)
            stage_results = self._record_stage_success(state, "pragma_dse", output, before_artifacts)
            stage_results.setdefault("pragma_dse", {})
            stage_results["pragma_dse"].update(
                {
                    "search_outcome": report.get("search_outcome", ""),
                    "best_candidate_id": report.get("best_candidate_id"),
                    "round_best_score": round_best_score,
                    "best_score_before_round": best_score_before,
                    "improved_over_best": improved,
                    "attempt": hardware_attempt,
                }
            )

            if successful_count <= 0:
                feedback = self._summarize_pragma_dse_dead_end(report)
                best_turn = int(state.get("best_hardware_turn", 0) or 0)
                if best_turn > 0:
                    self.optimized_code_turn = best_turn
                else:
                    self.optimized_code_turn = self._resolved_best_software_turn(state)
                stage_results = self._record_stage_retry(
                    state,
                    "pragma_dse",
                    feedback,
                    {
                        "successful_candidate_count": 0,
                        "baseline_success": True,
                        "attempt": hardware_attempt,
                    },
                )
                return self._emit_state(
                    hardware_turn=None,
                    final_code_turn=self.optimized_code_turn,
                    hardware_feedback=feedback,
                    pragma_dse_success=bool(state.get("best_pragma_dse_report_path")),
                    hardware_loop_continue=hardware_attempt < self._max_hardware_opt_rounds(),
                    stage_results=stage_results,
                )

            if not baseline_success:
                feedback = self._summarize_pragma_dse_dead_end(report)
                if feedback:
                    feedback += "\nBaseline candidate failed in pragma DSE; retry hardware optimization with a more stable baseline rewrite."
                else:
                    feedback = "Baseline candidate failed in pragma DSE; retry hardware optimization with a more stable baseline rewrite."
                best_turn = int(state.get("best_hardware_turn", 0) or 0)
                if best_turn > 0:
                    self.optimized_code_turn = best_turn
                else:
                    self.optimized_code_turn = self._resolved_best_software_turn(state)
                stage_results = self._record_stage_retry(
                    state,
                    "pragma_dse",
                    feedback,
                    {
                        "successful_candidate_count": successful_count,
                        "baseline_success": False,
                        "attempt": hardware_attempt,
                    },
                )
                return self._emit_state(
                    hardware_turn=None,
                    final_code_turn=self.optimized_code_turn,
                    hardware_feedback=feedback,
                    pragma_dse_success=bool(state.get("best_pragma_dse_report_path")),
                    hardware_loop_continue=hardware_attempt < self._max_hardware_opt_rounds(),
                    stage_results=stage_results,
                )

            if improved:
                snapshot_paths = self._snapshot_best_hardware_round(state, report_path)
                self.scratchpad.stage_artifacts.update(snapshot_paths)
                best_report_path = snapshot_paths.get("best_pragma_dse_report", report_path)
                best_candidates_path = snapshot_paths.get(
                    "best_pragma_candidates",
                    self.scratchpad.stage_artifacts.get("pragma_candidates", ""),
                )
                best_turn = int(state.get("hardware_turn", 0) or 0)
                self.optimized_code_turn = best_turn or self.optimized_code_turn
                continue_search = hardware_attempt < self._max_hardware_opt_rounds()
                feedback = self._build_hardware_progress_feedback(report, round_best, best_score_before, improved=True)
                refreshed_profiling_analysis = state.get("profiling_analysis", "")
                if continue_search and best_turn > 0:
                    refresh_before_artifacts = dict(self.scratchpad.stage_artifacts)
                    try:
                        refresh_result = self._refresh_best_qor_context_from_pragma_dse(state, report, best_turn, log_file)
                        refreshed_profiling_analysis = refresh_result["profiling_analysis"]
                        stage_results = self._record_stage_success(
                            {**state, "stage_results": stage_results},
                            "hardware_qor_refresh",
                            refresh_result["output"],
                            refresh_before_artifacts,
                        )
                    except Exception as exc:
                        _log_warn(f"Current best one refresh failed on v{best_turn}: {exc}")
                        self._append_text(log_file, f"[WARN] Current best one refresh failed on v{best_turn}: {exc}\n")
                        stage_results = self._record_stage_failure(
                            {**state, "stage_results": stage_results},
                            "hardware_qor_refresh",
                            str(exc),
                        )
                return self._emit_state(
                    hardware_turn=best_turn,
                    final_code_turn=self.optimized_code_turn,
                    best_hardware_turn=best_turn,
                    best_pragma_dse_score=round_best_score if round_best_score is not None else state.get("best_pragma_dse_score", 0.0),
                    best_pragma_dse_candidate_id=str(round_best.get("id", report.get("best_candidate_id", ""))),
                    best_pragma_dse_report_path=best_report_path,
                    best_pragma_candidates_path=best_candidates_path,
                    pragma_dse_success=True,
                    hardware_loop_continue=continue_search,
                    hardware_feedback=feedback if continue_search else "",
                    profiling_analysis=refreshed_profiling_analysis,
                    stage_results=stage_results,
                )

            best_turn = int(state.get("best_hardware_turn", 0) or 0)
            if best_turn > 0:
                self.optimized_code_turn = best_turn
                self.scratchpad.stage_artifacts["hardware_rewrite_turn"] = best_turn
                best_code_path = self.scratchpad.optimized_code_file.get(best_turn, "")
                if best_code_path:
                    self.scratchpad.stage_artifacts["hardware_rewrite_code"] = best_code_path
                best_report_path = str(state.get("best_pragma_dse_report_path", "")).strip()
                if best_report_path:
                    self.scratchpad.stage_artifacts["pragma_dse_report"] = best_report_path
                best_candidates_path = str(state.get("best_pragma_candidates_path", "")).strip()
                if best_candidates_path:
                    self.scratchpad.stage_artifacts["pragma_candidates"] = best_candidates_path
            else:
                self.optimized_code_turn = int(state.get("hardware_turn", 0) or self.optimized_code_turn)
            feedback = self._build_hardware_progress_feedback(report, round_best, best_score_before, improved=False)
            stage_results["pragma_dse"]["termination_reason"] = (
                "Latest hardware optimization round did not improve the current best QoR; stopping hardware loop."
            )
            return self._emit_state(
                hardware_turn=best_turn or state.get("hardware_turn"),
                final_code_turn=self.optimized_code_turn,
                pragma_dse_success=bool(state.get("best_pragma_dse_report_path") or round_best),
                hardware_loop_continue=False,
                hardware_feedback=feedback,
                stage_results=stage_results,
            )
        except Exception as exc:
            updates = self._append_error(state, "Pragma DSE", exc)
            updates["stage_results"] = self._record_stage_failure(state, "pragma_dse", str(exc))
            return self._emit_state(**updates)

    def _route_after_pragma_dse(self, state: AgentState) -> str:
        if state.get("hardware_loop_continue") and int(state.get("hardware_attempt", 0) or 0) < self._max_hardware_opt_rounds():
            return "kg_rag" if self._has_skill(state, "kg-rag") or self._has_skill(state, "rag") else "hardware_rewrite"
        return "finalize"

    def _node_finalize(self, state: AgentState) -> AgentState:
        self._restore_runtime_from_state(state)
        log_file = state["log_file"]
        run_dir = state["run_dir"]
        best_turn = int(state.get("best_hardware_turn", 0) or 0)
        final_turn = best_turn or self.optimized_code_turn
        if best_turn > 0:
            self.optimized_code_turn = best_turn
            best_code_path = self.scratchpad.optimized_code_file.get(best_turn, "")
            if best_code_path:
                self.scratchpad.stage_artifacts["hardware_rewrite_code"] = best_code_path
                self.scratchpad.stage_artifacts["hardware_rewrite_turn"] = best_turn
            best_report_path = str(state.get("best_pragma_dse_report_path", "")).strip()
            if best_report_path:
                self.scratchpad.stage_artifacts["pragma_dse_report"] = best_report_path
            best_candidates_path = str(state.get("best_pragma_candidates_path", "")).strip()
            if best_candidates_path:
                self.scratchpad.stage_artifacts["pragma_candidates"] = best_candidates_path
        final_candidate: Candidate | None = None
        for candidate in reversed(self.candidates):
            if candidate.turn == final_turn:
                final_candidate = candidate
                break

        summary = {
            "design_name": state.get("design_name", ""),
            "input_mode": state.get("input_mode", self.input_mode),
            "thread_id": state.get("thread_id", ""),
            "checkpoint_backend": self.checkpoint_backend,
            "checkpoint_db_path": self.checkpoint_db_path,
            "execution_status": "completed",
            "pending_nodes": [],
            "interrupt_after": list(self.interrupt_after),
            "run_dir": run_dir,
            "log_file": log_file,
            "final_code_turn": final_turn,
            "final_code_path": self.scratchpad.optimized_code_file.get(final_turn, ""),
            "best_software_turn": state.get("best_software_turn"),
            "hardware_turn": best_turn or state.get("hardware_turn"),
            "hardware_attempt": state.get("hardware_attempt", 0),
            "best_pragma_dse_score": state.get("best_pragma_dse_score"),
            "best_pragma_dse_candidate_id": state.get("best_pragma_dse_candidate_id", ""),
            "best_pragma_dse_report_path": state.get("best_pragma_dse_report_path", ""),
            "stage_artifacts": dict(self.scratchpad.stage_artifacts),
            "stage_results": deepcopy(state.get("stage_results", {})),
            "errors": list(state.get("errors", [])),
            "total_tokens_used": int(getattr(self.client, "total_tokens_used", 0) or 0),
        }
        if final_candidate is not None:
            summary["final_candidate"] = {
                "turn": final_candidate.turn,
                "stage": final_candidate.stage,
                "variant_kind": final_candidate.variant_kind,
                "score": final_candidate.score,
                "verification_pass": final_candidate.verification_pass,
                "metrics": final_candidate.metrics,
                "notes": final_candidate.notes,
            }

        self._append_text(log_file, "[INFO] Run completed.\n")
        _log_phase("Run Completed")
        _log_info(f"Output directory: {_c(run_dir, '1;37')}")
        _log_info(f"Log file: {_c(log_file, '1;37')}")
        results_artifacts = self._export_results(
            run_dir=run_dir,
            final_code_path=self.scratchpad.optimized_code_file.get(final_turn, ""),
            log_file=log_file,
            report_path=str(state.get("best_pragma_dse_report_path", "")).strip(),
            candidates_path=str(state.get("best_pragma_candidates_path", "")).strip(),
        )
        self.scratchpad.stage_artifacts.update(results_artifacts)
        try:
            rewrite_artifacts = self._update_rewrite_skill_from_run(
                design_name=state.get("design_name", ""),
                run_dir=run_dir,
                final_summary=summary,
                log_file=log_file,
            )
            self.scratchpad.stage_artifacts.update(rewrite_artifacts)
        except Exception as exc:
            self._append_text(log_file, f"[WARN] Failed to update rewrite skill lessons automatically: {exc}\n")
            _log_warn(f"Failed to update rewrite skill lessons automatically: {exc}")
        final_state_path = os.path.join(run_dir, "langgraph_state.json")
        self.scratchpad.stage_artifacts["langgraph_state"] = final_state_path
        summary["stage_artifacts"] = dict(self.scratchpad.stage_artifacts)
        emitted = self._emit_state(final_summary=summary, final_code_turn=final_turn)
        self._write_state_artifact(emitted, final_state_path)
        return emitted
