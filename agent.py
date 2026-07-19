"""Agent class with SKILL interaction methods."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import html
import random
import re
import shlex
from collections import Counter
from pathlib import Path
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from skills_ref import read_properties, validate, to_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Console output helpers — styled, level-aware terminal logging
# ---------------------------------------------------------------------------
_COLOR_ENABLED = os.isatty(1) and os.getenv("NO_COLOR") is None
_VERBOSE = os.getenv("AGENTSKL_VERBOSE", "0") == "1"
_CONSOLE_STATUS_MANAGER: Any | None = None


def _c(text: str, code: str) -> str:
	if not _COLOR_ENABLED:
		return text
	return f"\033[{code}m{text}\033[0m"


def set_console_status_manager(manager: Any | None) -> None:
	"""Install a terminal status-line manager used by interactive run.py."""
	global _CONSOLE_STATUS_MANAGER
	_CONSOLE_STATUS_MANAGER = manager


def _status_safe_print(*lines: str) -> None:
	manager = _CONSOLE_STATUS_MANAGER
	if manager is not None:
		manager.before_print()
	try:
		for line in lines:
			print(line)
	finally:
		if manager is not None:
			manager.after_print()


def _log_info(msg: str) -> None:
	"""Key progress updates — run folder, selected skill, generated files."""
	_status_safe_print(f"{_c('ℹ', '1;36')} {_c('[INFO]', '36')}  {msg}")


def _log_warn(msg: str) -> None:
	_status_safe_print(f"{_c('⚠', '1;33')} {_c('[WARN]', '33')}  {msg}")


def _log_error(msg: str) -> None:
	_status_safe_print(f"{_c('✖', '1;31')} {_c('[ERROR]', '31')} {msg}")


def _log_debug(msg: str) -> None:
	"""Verbose details — only shown when AGENTSKL_VERBOSE=1."""
	if _VERBOSE:
		_status_safe_print(f"{_c('⋯', '90')} {_c('[DEBUG]', '90')} {msg}")


def _log_skill(msg: str) -> None:
	"""Skill selection / activation events."""
	_status_safe_print(f"{_c('🧩', '0')} {_c('[SKILL]', '35')} {msg}")


def _log_exec(msg: str) -> None:
	"""Command execution events."""
	_status_safe_print(f"{_c('▶', '1;32')} {_c('[EXEC]', '32')}  {msg}")


def _log_file(msg: str) -> None:
	"""File creation / output events."""
	_status_safe_print(f"{_c('📄', '0')} {_c('[FILE]', '34')}  {msg}")


def _log_phase(msg: str) -> None:
	"""Major pipeline phase headers."""
	bar = "─" * 40
	_status_safe_print(
		f"\n{_c(bar, '90')}",
		f"{_c('◆', '1;35')} {_c(msg, '1;35')}",
		f"{_c(bar, '90')}",
	)


def _parse_response_data(resp: requests.Response) -> Dict[str, Any]:
	"""Parse provider response into a JSON object, with SSE fallback."""
	text = (resp.text or "").strip()
	if not text:
		raise ValueError("Empty response body")

	try:
		data = resp.json()
		if isinstance(data, dict):
			return data
		raise ValueError(f"JSON root is not an object: {type(data).__name__}")
	except ValueError:
		pass

	# Some gateways may return server-sent-event lines even when stream=False.
	if "data:" in text:
		sse_chunks: List[Dict[str, Any]] = []
		for line in text.splitlines():
			row = line.strip()
			if not row.startswith("data:"):
				continue
			payload = row[5:].strip()
			if not payload or payload == "[DONE]":
				continue
			try:
				obj = json.loads(payload)
				if isinstance(obj, dict):
					sse_chunks.append(obj)
			except ValueError:
				continue
		if sse_chunks:
			return sse_chunks[-1]

	content_type = resp.headers.get("Content-Type", "")
	preview = text[:240].replace("\n", "\\n")
	raise ValueError(f"Non-JSON response (content-type={content_type}): {preview}")


def _extract_chat_content(data: Dict[str, Any]) -> str:
	"""Extract assistant text from OpenAI-compatible or Anthropic-style payloads."""
	choices = data.get("choices")
	if isinstance(choices, list) and choices:
		first = choices[0]
		if isinstance(first, dict):
			msg = first.get("message", {})
			if isinstance(msg, dict):
				content = msg.get("content", "")
				if isinstance(content, str) and content:
					return content
				if isinstance(content, list):
					texts = []
					for block in content:
						if isinstance(block, dict):
							txt = block.get("text")
							if isinstance(txt, str) and txt:
								texts.append(txt)
					if texts:
						return "\n".join(texts)
			delta = first.get("delta", {})
			if isinstance(delta, dict):
				delta_text = delta.get("content")
				if isinstance(delta_text, str) and delta_text:
					return delta_text

	# Anthropic native style: {"content":[{"type":"text","text":"..."}], ...}
	content_blocks = data.get("content")
	if isinstance(content_blocks, list):
		texts = []
		for block in content_blocks:
			if not isinstance(block, dict):
				continue
			txt = block.get("text")
			if isinstance(txt, str) and txt:
				texts.append(txt)
		if texts:
			return "\n".join(texts)

	output_text = data.get("output_text")
	if isinstance(output_text, str) and output_text:
		return output_text

	raise ValueError("Response JSON does not contain assistant text content")


class LLMClient:
	def __init__(self, config: Dict[str, Any]) -> None:
		self.config = config
		self.provider: str = config.get("provider", "openrouter")
		self._last_call_ts: float = 0.0
		self.total_tokens_used: int = 0
		self._available_models: List[str] | None = None
		self._gemini_client: Any | None = None

	def _build_headers(self) -> Dict[str, str]:
		if self.provider == "anthropic":
			return {
				"x-api-key": self.config.get("api_key", ""),
				"anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
				"Content-Type": "application/json",
			}

		headers = {
			"Authorization": f"Bearer {self.config.get('api_key', '')}",
			"Content-Type": "application/json",
		}
		if self.provider == "openrouter":
			headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
			headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE", "skills-agent")
		elif self.provider == "copilot":
			headers["Editor-Version"] = "vscode/1.96.2"
			headers["Editor-Plugin-Version"] = "copilot-chat/0.26.7"
			headers["User-Agent"] = "GitHubCopilotChat/0.26.7"
			headers["Copilot-Integration-Id"] = "vscode-chat"
		return headers

	def _get_gemini_client(self) -> Any:
		if self._gemini_client is not None:
			return self._gemini_client
		try:
			from google import genai
			from google.genai import types as genai_types
			import httpx
		except ImportError as exc:
			raise RuntimeError(
				"Gemini provider requires the 'google-genai' package in the current Python environment."
			) from exc
		api_key = str(self.config.get("api_key", "")).strip() or os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
		if not api_key:
			raise RuntimeError("Gemini provider selected but GEMINI_API_KEY/GOOGLE_API_KEY is not set.")
		timeout_ms = int(self.config.get("timeout", 300) or 300) * 1000
		http_options = genai_types.HttpOptions(
			timeout=timeout_ms,
			clientArgs={
				"trust_env": False,
				"http2": False,
				"limits": httpx.Limits(max_connections=20, max_keepalive_connections=0),
			},
			asyncClientArgs={
				"trust_env": False,
				"http2": False,
			},
		)
		self._gemini_client = genai.Client(api_key=api_key, http_options=http_options)
		return self._gemini_client

	def _convert_gemini_messages(self, messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, Any]]]:
		system_parts: List[str] = []
		converted: List[Dict[str, Any]] = []
		for msg in messages:
			role = str(msg.get("role", "user")).strip().lower()
			content = msg.get("content", "")
			if isinstance(content, list):
				parts: List[str] = []
				for block in content:
					if isinstance(block, dict):
						txt = block.get("text")
						if isinstance(txt, str) and txt:
							parts.append(txt)
				content = "\n".join(parts)
			elif not isinstance(content, str):
				content = json.dumps(content, ensure_ascii=False)

			if role == "system":
				if content:
					system_parts.append(content)
				continue

			gemini_role = "model" if role == "assistant" else "user"
			converted.append({"role": gemini_role, "parts": [{"text": content}]})

		if not converted:
			converted.append({"role": "user", "parts": [{"text": ""}]})
		return "\n\n".join(system_parts).strip(), converted

	def _chat_gemini(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
		if stream:
			raise RuntimeError("Gemini streaming mode is not implemented in HLSClaw.")

		client = self._get_gemini_client()
		try:
			from google.genai import types as genai_types
		except ImportError as exc:
			raise RuntimeError(
				"Gemini provider requires the 'google-genai' package in the current Python environment."
			) from exc

		system_prompt, contents = self._convert_gemini_messages(messages)
		config = genai_types.GenerateContentConfig(
			systemInstruction=system_prompt or None,
			temperature=self.config.get("temperature", 0.7),
			topP=self.config.get("top_p", 1.0),
			maxOutputTokens=self.config.get("max_tokens", 1000),
		)

		retries = int(self.config.get("max_retries", 3))
		backoff = float(self.config.get("backoff", 2.0))
		backoff_max = float(self.config.get("backoff_max", 60.0))
		attempt = 0
		while attempt <= retries:
			try:
				self._throttle()
				response = client.models.generate_content(
					model=str(self.config.get("model", "")),
					contents=contents,
					config=config,
				)
				content = getattr(response, "text", "")
				if not isinstance(content, str) or not content:
					raw = response.to_json_dict() if hasattr(response, "to_json_dict") else {}
					content = _extract_chat_content(raw)
				raw = response.to_json_dict() if hasattr(response, "to_json_dict") else {"output_text": content}
				usage = getattr(response, "usage_metadata", None)
				if usage is not None:
					total_tokens = getattr(usage, "total_token_count", None)
					if total_tokens is None:
						total_tokens = sum(
							int(getattr(usage, field, 0) or 0)
							for field in (
								"prompt_token_count",
								"candidates_token_count",
								"tool_use_prompt_token_count",
								"thoughts_token_count",
							)
						)
					self.total_tokens_used += int(total_tokens or 0)
				return {"content": content, "raw": raw}
			except Exception as exc:
				detail = str(exc)
				retryable = any(
					sig in detail.lower()
					for sig in (
						"rate limit",
						"429",
						"500",
						"502",
						"503",
						"504",
						"deadline exceeded",
						"timed out",
						"server disconnected without sending a response",
						"remoteprotocolerror",
						"connection reset",
						"connection aborted",
						"temporarily unavailable",
					)
				)
				if (not retryable) or attempt >= retries:
					raise RuntimeError(f"Gemini request failed: {detail}") from exc
				# Rebuild the underlying client on transport failures to avoid stale keep-alive sockets.
				self._gemini_client = None
				time.sleep(min(backoff_max, backoff ** attempt))
				client = self._get_gemini_client()
				attempt += 1

		raise RuntimeError("Gemini request failed after retries.")

	def _normalize_model_id(self, model_id: str) -> str:
		return re.sub(r"[^a-z0-9]+", "", model_id.strip().lower())

	def _pick_supported_model(self, requested_model: str, available_models: List[str]) -> str:
		if not available_models:
			return requested_model
		if requested_model in available_models:
			return requested_model

		req_norm = self._normalize_model_id(requested_model)
		if req_norm:
			for model in available_models:
				if self._normalize_model_id(model) == req_norm:
					return model

		req_tokens = [t for t in re.split(r"[^a-z0-9]+", requested_model.strip().lower()) if len(t) >= 2]
		best_model = ""
		best_score = -1
		for model in available_models:
			model_lower = model.lower()
			score = sum(1 for token in req_tokens if token in model_lower)
			if score > best_score:
				best_score = score
				best_model = model
		if best_model and best_score > 0:
			return best_model
		return available_models[0]

	def _fetch_available_models(self) -> List[str]:
		if self._available_models is not None:
			return self._available_models

		if self.provider == "gemini":
			try:
				client = self._get_gemini_client()
				models: List[str] = []
				for item in client.models.list():
					name = str(getattr(item, "name", "") or "").strip()
					actions = getattr(item, "supported_actions", None) or []
					if actions and "generateContent" not in actions:
						continue
					if name.startswith("models/"):
						name = name.split("/", 1)[1]
					if name:
						models.append(name)
				self._available_models = sorted(set(models))
			except Exception:
				self._available_models = []
			return self._available_models

		api_base = str(self.config.get("api_base", "")).rstrip("/")
		if not api_base:
			self._available_models = []
			return self._available_models
		if self.provider == "anthropic" and not api_base.endswith("/v1"):
			api_base = api_base + "/v1"

		try:
			resp = requests.get(
				f"{api_base}/models",
				headers=self._build_headers(),
				timeout=int(self.config.get("timeout", 60)),
			)
			resp.raise_for_status()
			data = resp.json()
			raw_models = data.get("data", []) if isinstance(data, dict) else []
			models: List[str] = []
			for item in raw_models:
				if not isinstance(item, dict):
					continue
				model_id = str(item.get("id", "")).strip()
				if model_id:
					models.append(model_id)
			self._available_models = sorted(set(models))
		except (requests.RequestException, ValueError, TypeError):
			self._available_models = []
		return self._available_models

	def _is_transient_overload(self, status: int, body: str) -> bool:
		if status not in (500, 502, 503, 504):
			return False
		text = (body or "").lower()
		if not text:
			return True
		signatures = (
			"model_temporarily_unavailable",
			"temporarily unavailable",
			"unexpectedly high load",
			"overloaded",
			"rate limit",
		)
		return any(sig in text for sig in signatures)

	def _pick_load_shed_model(self, current_model: str, available_models: List[str]) -> str:
		if not available_models:
			return current_model
		cur = current_model.lower()
		if self.provider == "gemini" and "pro" in cur:
			for model in available_models:
				if "flash" in model.lower():
					return model
		if self.provider == "anthropic" and "opus" in cur:
			for model in available_models:
				if "sonnet" in model.lower():
					return model
			for model in available_models:
				if "haiku" in model.lower():
					return model
		return current_model

	def _resolve_chat_base(self) -> str:
		api_base = str(self.config.get("api_base", "")).strip().rstrip("/")
		if self.provider == "anthropic" and api_base and not api_base.endswith("/v1"):
			api_base = api_base + "/v1"
		return api_base

	def _convert_anthropic_messages(self, messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
		system_parts: List[str] = []
		converted: List[Dict[str, str]] = []
		for msg in messages:
			role = str(msg.get("role", "user")).strip().lower()
			content = msg.get("content", "")
			if isinstance(content, list):
				parts: List[str] = []
				for block in content:
					if isinstance(block, dict):
						txt = block.get("text")
						if isinstance(txt, str) and txt:
							parts.append(txt)
				content = "\n".join(parts)
			elif not isinstance(content, str):
				content = json.dumps(content, ensure_ascii=False)

			if role == "system":
				if content:
					system_parts.append(content)
				continue
			if role not in ("user", "assistant"):
				role = "user"
			converted.append({"role": role, "content": content})

		if not converted:
			converted.append({"role": "user", "content": ""})
		return "\n\n".join(system_parts).strip(), converted

	def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
		if self.provider == "gemini":
			return self._chat_gemini(messages, stream=stream)

		headers = self._build_headers()

		if self.provider == "anthropic":
			url = self._resolve_chat_base() + "/messages"
			system_prompt, anthropic_messages = self._convert_anthropic_messages(messages)
			payload = {
				"model": self.config.get("model", ""),
				"messages": anthropic_messages,
				"max_tokens": self.config.get("max_tokens", 1000),
				"stream": stream,
			}
			if system_prompt:
				payload["system"] = system_prompt
			temperature = self.config.get("temperature")
			if temperature is not None:
				payload["temperature"] = temperature
			top_p = self.config.get("top_p")
			if top_p is not None:
				payload["top_p"] = top_p
		else:
			url = self._resolve_chat_base() + "/chat/completions"
			payload = {
				"model": self.config.get("model", ""),
				"messages": messages,
				"temperature": self.config.get("temperature", 0.7),
				"top_p": self.config.get("top_p", 1.0),
				"max_tokens": self.config.get("max_tokens", 1000),
				"stream": stream,
			}

		retries = int(self.config.get("max_retries", 3))
		transient_extra_retries = int(self.config.get("transient_extra_retries", 6))
		transient_min_wait = float(self.config.get("transient_min_wait", 3.0))
		transient_max_retries = retries + max(0, transient_extra_retries)
		attempt = 0
		unknown_model_retried = False
		load_shed_retried = False
		while attempt <= transient_max_retries:
			try:
				self._throttle()
				resp = requests.post(url, headers=headers, json=payload, timeout=int(self.config.get("timeout", 60)))
				if resp.status_code == 429:
					wait = self._compute_wait(resp, attempt)
					if attempt == retries:
						resp.raise_for_status()
					time.sleep(wait)
					attempt += 1
					continue
				resp.raise_for_status()
				try:
					data = _parse_response_data(resp)
					content = _extract_chat_content(data)
				except ValueError as exc:
					detail = f"Invalid response format from {url}: {exc}"
					if attempt == retries:
						raise RuntimeError(detail) from exc
					backoff_max = float(self.config.get("backoff_max", 60.0))
					backoff = float(self.config.get("backoff", 2.0))
					time.sleep(min(backoff_max, backoff ** attempt))
					attempt += 1
					continue

				# Track token usage
				usage = data.get("usage", {}) if isinstance(data, dict) else {}
				if isinstance(usage, dict):
					total_tokens = usage.get("total_tokens")
					if total_tokens is None:
						total_tokens = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
					self.total_tokens_used += int(total_tokens or 0)
				return {"content": content, "raw": data}
			except requests.HTTPError as exc:
				status = exc.response.status_code if exc.response is not None else -1
				body = ""
				if exc.response is not None:
					body = (exc.response.text or "").strip()
					if len(body) > 800:
						body = body[:800] + "...(truncated)"
				transient_overload = self._is_transient_overload(status, body)
				error_code = ""
				if exc.response is not None:
					try:
						error_obj = exc.response.json().get("error", {})
						if isinstance(error_obj, dict):
							error_code = str(error_obj.get("code", "")).strip().lower()
					except ValueError:
						pass

				if status == 400 and error_code == "unknown_model" and not unknown_model_retried:
					available_models = self._fetch_available_models()
					fallback_model = self._pick_supported_model(str(payload.get("model", "")), available_models)
					current_model = str(payload.get("model", ""))
					if fallback_model and fallback_model != current_model:
						logger.warning(
							"Model '%s' unavailable on provider '%s'; fallback to '%s'.",
							current_model,
							self.provider,
							fallback_model,
						)
						payload["model"] = fallback_model
						self.config["model"] = fallback_model
						unknown_model_retried = True
						continue

				if transient_overload and not load_shed_retried:
					current_model = str(payload.get("model", ""))
					available_models = self._fetch_available_models()
					shed_model = self._pick_load_shed_model(current_model, available_models)
					if shed_model and shed_model != current_model:
						logger.warning(
							"Model '%s' overloaded on provider '%s'; fallback to '%s'.",
							current_model,
							self.provider,
							shed_model,
						)
						payload["model"] = shed_model
						self.config["model"] = shed_model
						load_shed_retried = True
						continue

				detail = f"HTTP {status} for {url}"
				if body:
					detail += f" | body: {body}"

				# 4xx (except 429 handled above) is usually non-retryable.
				retryable = status >= 500
				max_retries_for_error = transient_max_retries if transient_overload else retries
				if (not retryable) or attempt >= max_retries_for_error:
					raise RuntimeError(detail) from exc

				backoff_max = float(self.config.get("backoff_max", 60.0))
				backoff = float(self.config.get("backoff", 2.0))
				wait = min(backoff_max, backoff ** attempt)
				if transient_overload:
					wait = max(wait, transient_min_wait)
				time.sleep(wait)
				attempt += 1
			except requests.RequestException:
				if attempt >= retries:
					raise
				backoff_max = float(self.config.get("backoff_max", 60.0))
				backoff = float(self.config.get("backoff", 2.0))
				time.sleep(min(backoff_max, backoff ** attempt))
				attempt += 1

		raise RuntimeError(f"HTTP request failed after retries for {url}")

	def _throttle(self) -> None:
		now = time.time()
		elapsed = now - self._last_call_ts
		wait = float(self.config.get("min_interval_sec", 1.0)) - elapsed
		if wait > 0:
			time.sleep(wait)
		self._last_call_ts = time.time()

	def _compute_wait(self, resp: requests.Response, attempt: int) -> float:
		retry_after = resp.headers.get("Retry-After")
		if retry_after:
			try:
				return min(float(self.config.get("backoff_max", 60.0)), float(retry_after))
			except ValueError:
				pass

		backoff_max = float(self.config.get("backoff_max", 60.0))
		backoff = float(self.config.get("backoff", 2.0))
		jitter_factor = float(self.config.get("jitter", 0.3))
		base = min(backoff_max, backoff ** max(1, attempt))
		jitter = base * jitter_factor * random.random()
		return base + jitter

@dataclass
class SkillInfo:
	name: str
	description: str
	skill_dir: str
	skill_md: str

@dataclass
class Candidate:
	turn: int
	parent_turn: int
	stage: str
	variant_kind: str
	score: float
	verification_pass: bool
	metrics: Dict[str, float]
	notes: str


class FatalRunError(RuntimeError):
	def __init__(
		self,
		message: str,
		*,
		status_reason: str = "fatal_error",
		failing_stage: str = "",
		resumable: bool = False,
	) -> None:
		super().__init__(message)
		self.status_reason = status_reason
		self.failing_stage = failing_stage
		self.resumable = resumable

##TODO: Extend scratchpad to store command results
@dataclass
class Scratchpad:
	goal: str
	fpga: str
	target_frequency_mhz: str
	top_function_name: str
	run_dir: str
	stage_artifacts: Dict[str, str]
	analysis: Dict[int, str]
	command: Dict[int, List[str]]
	reference: Dict[int, List[str]]
	json_artifact: Dict[int, Dict[str, Any]]
	skill: Dict[int, str]
	optimized_code: Dict[int, str]
	optimized_code_file: Dict[int, str]
	input_mode: str = "plain_c"

## TODO: Differentiate our agent framework with other agent frameworks
class Agent:

	"""Agent that discovers skills, selects relevant ones, and runs the LLM workflow."""

	def __init__(
		self,
		config: Dict[str, Any],
		system_prompt: str,
		code_path: str,
		req_path: str,
		top_function_name: str | None = None,
		input_mode: str = "plain_c",
		skills_dir: str | None = None,
		out_dir: str | None = None,
	) -> None:
		self.config = config
		self.client = LLMClient(config)
		self.system_prompt = system_prompt
		self.code_path = os.path.abspath(code_path)
		self.req_path = os.path.abspath(req_path)
		self.top_function_name = top_function_name or ""
		self.input_mode = "hls_native" if str(input_mode).strip().lower() == "hls_native" else "plain_c"
		base_dir = os.path.dirname(os.path.abspath(__file__))
		self.skills_dir = skills_dir or os.path.join(base_dir, "skills")
		self.skills_path = self.skills_dir
		self.out_dir = out_dir or os.path.join(base_dir, "runs")
		self.analysis_turn : int = 0
		self.command_turn : int = 0
		self.reference_turn : int = 0
		self.skill_turn : int = 0
		self.optimized_code_turn : int = 0
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
		self.candidates: List[Candidate] = []
		self.conversation_history: List[Dict[str, str]] = []

	def _chat_with_history(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
		"""Call LLM with conversation history context injected."""
		context_summary = self._build_context_summary()
		augmented = []
		if messages and messages[0]["role"] == "system":
			augmented.append(messages[0])
			if context_summary:
				augmented.append({"role": "user", "content": f"<ConversationContext>\n{context_summary}\n</ConversationContext>"})
				augmented.append({"role": "assistant", "content": "Understood, I have the prior context."})
			augmented.extend(messages[1:])
		else:
			if context_summary:
				augmented.append({"role": "user", "content": f"<ConversationContext>\n{context_summary}\n</ConversationContext>"})
				augmented.append({"role": "assistant", "content": "Understood, I have the prior context."})
			augmented.extend(messages)
		result = self.client.chat(augmented)
		# Record this exchange in history
		user_parts = [m["content"] for m in messages if m["role"] == "user"]
		self.conversation_history.append({
			"role": "user",
			"content": user_parts[0][:2000] if user_parts else "",
		})
		self.conversation_history.append({
			"role": "assistant",
			"content": result["content"][:2000],
		})
		return result

	def _build_context_summary(self) -> str:
		"""Build a summary from the last 5 conversation rounds, each truncated to 2000 chars."""
		if not self.conversation_history:
			return ""
		recent = self.conversation_history[-10:]  # last 5 rounds = 10 messages
		parts = []
		for msg in recent:
			role = msg["role"]
			content = msg["content"][:2000]
			parts.append(f"[{role}]: {content}")
		return "\n---\n".join(parts)

	def run_CoT(self) -> None:
		"""Legacy entrypoint removed in HLSClaw; use LangGraphAgent.run_langgraph()."""
		raise RuntimeError(
			"Agent.run_CoT() is deprecated in HLSClaw. "
			"Use LangGraphAgent.run_langgraph() or `python -m HLSClaw.run` instead."
		)

	def discover_skills(self) -> List[SkillInfo]:
		"""Discover skills under skills_dir, validate them, and read properties."""
		skills: List[SkillInfo] = []
		if not os.path.isdir(self.skills_dir):
			return skills

		for entry in sorted(os.listdir(self.skills_dir)):
			skill_dir = Path(self.skills_dir) / entry
			if not skill_dir.is_dir():
				continue
			
			problems = validate(skill_dir)
			if problems:
				raise ValueError(
					"[ERROR] Invalid skill format: "
					+ str(skill_dir)
					+ "\n"
					+ "\n".join(problems)
				)

			props = read_properties(skill_dir)
			skill_md = skill_dir / "SKILL.md"
			skills.append(
				SkillInfo(
					name=getattr(props, "name", "") or entry,
					description=getattr(props, "description", "") or "",
					skill_dir=str(skill_dir),
					skill_md=str(skill_md),
				)
			)
		return skills

	def skills_to_prompt(self, skills: List[SkillInfo]) -> str:
		"""Convert a list of SkillInfo to a prompt string."""
		if not skills:
			return ""
		prompt_parts = []
		for skill in skills:
			skill_dir = Path(skill.skill_dir)
			prompt_parts.append(to_prompt([skill_dir]).strip())
		return "\n\n".join(prompt_parts)
	
	def _resolve_skill_name(self, skills: List[SkillInfo], name: str) -> str:
		"""Three-level skill name matching: exact -> case-insensitive -> substring."""
		if not name or not skills:
			return ""
		by_name = {skill.name: skill.name for skill in skills}
		# Exact match
		if name in by_name:
			return name
		# Case-insensitive match
		name_lower = name.lower().strip()
		for skill in skills:
			if skill.name.lower() == name_lower:
				return skill.name
		# Substring match
		for skill in skills:
			if name_lower in skill.name.lower() or skill.name.lower() in name_lower:
				return skill.name
		return ""

	def _load_skill_docs(self, skills: List[SkillInfo], name: str) -> str:
		"""Load a specific SKILL.md and return its raw content."""
		if not skills or not name:
			return ""
		resolved = self._resolve_skill_name(skills, name)
		if not resolved:
			return ""
		by_name = {skill.name: skill for skill in skills}
		skill_md = Path(by_name[resolved].skill_md)
		if not skill_md.is_file():
			return ""
		return self._read_text(str(skill_md))

	def _rewrite_skill_references_dir(self) -> str:
		return os.path.join(self.skills_dir, "rewrite", "references")

	def _rewrite_auto_lessons_md_path(self) -> str:
		return os.path.join(self._rewrite_skill_references_dir(), "auto_learned_lessons.md")

	def _rewrite_dataflow_circuit_structural_strategy_md_path(self) -> str:
		return os.path.join(self._rewrite_skill_references_dir(), "dataflow_circuit_structural_strategy.md")

	def _rewrite_dataflow_circuit_legality_checks_md_path(self) -> str:
		return os.path.join(self._rewrite_skill_references_dir(), "dataflow_circuit_legality_checks.md")

	def _kg_rag_skill_references_dir(self) -> str:
		return os.path.join(self.skills_dir, "kg-rag", "references")

	def _kg_rag_query_templates_md_path(self) -> str:
		return os.path.join(self._kg_rag_skill_references_dir(), "query_templates.md")

	def _load_rewrite_auto_lessons_prompt(self) -> str:
		path = self._rewrite_auto_lessons_md_path()
		if not os.path.exists(path):
			return ""
		text = self._read_text(path).strip()
		if not text:
			return ""
		lines = text.splitlines()
		if len(lines) > 160:
			blocks: List[str] = []
			intro = "\n".join(lines[:18]).strip()
			if intro:
				blocks.append(intro)
			for spec in self._rewrite_lessons_section_specs():
				start, end = self._find_rewrite_section_span(text, spec["heading"])
				if start < 0 or end < 0:
					continue
				section_block = text[start:end].strip()
				if not section_block:
					continue
				section_lines = section_block.splitlines()
				if len(section_lines) > 18:
					section_lines = section_lines[:2] + ["..."] + section_lines[-15:]
				blocks.append("\n".join(section_lines).strip())
			text = "\n\n".join(block for block in blocks if block).rstrip()
		return text

	def _load_trimmed_markdown_prompt(self, path: str, *, max_lines: int = 120) -> str:
		if not os.path.exists(path):
			return ""
		text = self._read_text(path).strip()
		if not text:
			return ""
		lines = text.splitlines()
		if len(lines) <= max_lines:
			return text
		head = lines[: max_lines - 12]
		tail = lines[-10:]
		return "\n".join(head + ["...", "..."] + tail).rstrip()

	def _load_rewrite_dataflow_circuit_prompt(self) -> str:
		parts: List[str] = []
		for title, path in (
			("Dataflow Circuit Structural Strategy", self._rewrite_dataflow_circuit_structural_strategy_md_path()),
			("Dataflow Circuit Legality Checks", self._rewrite_dataflow_circuit_legality_checks_md_path()),
		):
			text = self._load_trimmed_markdown_prompt(path, max_lines=90)
			if text:
				parts.append(f"## {title}\n{text}")
		return "\n\n".join(parts).rstrip()

	def _load_kg_rag_query_templates_prompt(self) -> str:
		path = self._kg_rag_query_templates_md_path()
		if not os.path.exists(path):
			return ""
		return self._read_text(path).strip()

	def select_skills(
		self,
		available_skills_xml: str,
		user_prompt: str,
		log_file: str
	) -> str:
		"""Ask LLM to select a single skill; return empty string on failure."""
		message = [
		{"role": "system", "content": "Read all the skills and the user-provided prompt, then choose the single best skill to perform next. If the user-provided prompt explicitly recommends a skill, you MUST select that skill. Otherwise, choose the best fit. Output skill name wrapped in <skill></skill>, brief rationale in <analysis></analysis>, and a structured object in <json>{\"selected_skill\":\"...\",\"reason\":\"...\"}</json>."},
		{"role": "user", "content": available_skills_xml + "\n" + user_prompt},
		]
		result = self.client.chat(message)
		raw = result["content"].strip()
		self._log_block(log_file, "Skill selection raw output", raw, max_lines=18, max_chars=1200)
		self._parse_artifacts(raw, log_file)
		if self.skill_turn not in self.scratchpad.skill:
			raise ValueError("[ERROR] Skill selector did not return a valid <skill> tag.")
		_log_skill(f"Selected: {_c(self.scratchpad.skill[self.skill_turn].strip(), '1;37')}")
		selected_skill = self.scratchpad.skill[self.skill_turn].lower().replace("\n", " ")
		return selected_skill

	def build_skill_prompt(self, available_skills: List[SkillInfo], skill_name: str) -> List[Dict[str, str]]:
		"""Build skill prompt from skill document."""
		# D2: Fuzzy resolve skill name before building paths
		resolved = self._resolve_skill_name(available_skills, skill_name)
		if resolved:
			skill_name = resolved
		basic_info : str = self._build_info_prompt_from_scratchpad()
		skill_scripth_path = os.path.join(self.skills_dir, skill_name, "scripts")
		skill_reference_path = os.path.join(self.skills_dir, skill_name, "references")
		skill_assets_path = os.path.join(self.skills_dir, skill_name, "assets")
		skill_path_info = (
        "<SkillPaths>\n"
        f"  <Skill_Script_absolute_Path>{skill_scripth_path}</Skill_Script_absolute_Path>\n"
        f"  <Skill_Reference_absolute_Path>{skill_reference_path}</Skill_Reference_absolute_Path>\n"
        f"  <Skill_Assets_absolute_Path>{skill_assets_path}</Skill_Assets_absolute_Path>\n"
		"</SkillPaths>"
    	)
		skill_prompt = self._load_skill_docs(available_skills, skill_name)
		if "rewrite" in skill_name.lower():
			auto_lessons = self._load_rewrite_auto_lessons_prompt()
			if auto_lessons:
				skill_prompt += (
					"\n\n## Auto-Learned Rewrite Lessons (Loaded Automatically)\n"
					+ auto_lessons
					+ "\n"
				)
			dataflow_circuit_guidance = self._load_rewrite_dataflow_circuit_prompt()
			if dataflow_circuit_guidance:
				skill_prompt += (
					"\n\n## Dataflow Circuit Structural Guidance (Loaded Automatically)\n"
					+ dataflow_circuit_guidance
					+ "\n"
				)
		if "kg-rag" in skill_name.lower() or skill_name.lower() == "rag":
			query_templates = self._load_kg_rag_query_templates_prompt()
			if query_templates:
				skill_prompt += (
					"\n\n## KG-RAG Query Templates (Loaded Automatically)\n"
					+ query_templates
					+ "\n"
				)
		message = [
			{"role": "system", "content": "Read the specified skill and decide what action to take. Basic information about design is provided. Mark analysis content with <analysis></analysis>. If you need to execute commands, output one command per <command>...</command> block. Each command must start with python or other compiler name based on what you learned from the SKILL.md and include the absolute path to the script file. Do not combine commands with &&, \%\ or any other separators or symbols. If you need to cite rules, references, or supporting material, output the reference file name per <reference>...</reference> block. Also output one structured plan object in <json>{\"skill\":\"...\",\"commands\":[...],\"references\":[...],\"expected_outputs\":[...]}</json>. Not all skills include commands or references; if a section is not applicable, use empty arrays."},
	        {"role": "user", "content": basic_info + "\n" + skill_path_info + "\n" + skill_prompt + "\n"},
	    	]
		return message

	def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
		return [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

	def _generate_llm_prompt(self, output: Dict[str, Any]) -> str:
		LLM_prompt : str = ""
		if "analysis" in output:
			LLM_prompt += "Previous Results:\n"
			LLM_prompt += output["analysis"] + "\n"
		if "command_result" in output:
			LLM_prompt += "Executed Command Results:\n"
			LLM_prompt += output["command_result"] + "\n"
		if "reference_result" in output:
			LLM_prompt += "Reference Materials:\n"
			LLM_prompt += output["reference_result"] + "\n"
		if "json_artifacts" in output:
			LLM_prompt += "Structured Artifacts:\n"
			LLM_prompt += json.dumps(output["json_artifacts"], indent=2) + "\n"
		return LLM_prompt

	def _extract_xml_tag(self, text: str, tag: str) -> List[str]:
		"""Extract all contents between XML tags. Returns empty list if tag not found."""
		import re
		pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
		matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
		return [match.strip() for match in matches if match and match.strip()]

	def _preview_text_for_log(self, text: str, max_lines: int = 20, max_chars: int = 1600) -> str:
		content = str(text or "").strip()
		if not content:
			return ""
		if _VERBOSE:
			return content
		lines = content.splitlines()
		total_lines = len(lines)
		if total_lines > max_lines:
			head_count = max(1, max_lines - 4)
			lines = lines[:head_count] + ["...", f"[{total_lines - head_count} more lines omitted]"]
		preview = "\n".join(lines)
		if len(preview) > max_chars:
			preview = preview[: max_chars - 16].rstrip() + "\n...[truncated]"
		return preview

	def _log_block(self, log_file: str, header: str, content: str, max_lines: int = 20, max_chars: int = 1600) -> None:
		preview = self._preview_text_for_log(content, max_lines=max_lines, max_chars=max_chars)
		if preview:
			self._append_text(log_file, f"[DEBUG] {header}\n{preview}\n\n")
		else:
			self._append_text(log_file, f"[DEBUG] {header}\n<empty>\n\n")

	def _summarize_json_for_log(self, data: Any) -> str:
		if isinstance(data, dict):
			keys = sorted(str(key) for key in data.keys())
			preview_keys = ", ".join(keys[:8])
			if len(keys) > 8:
				preview_keys += ", ..."
			return f"object keys=[{preview_keys}]"
		if isinstance(data, list):
			return f"array len={len(data)}"
		return type(data).__name__

	def _parse_artifacts(self, raw: str, log_file: str, allow_optimized_code: bool = True) -> Dict[str, Any]:
		"""Parse XML output tags and store them in scratchpad."""
		# Capture current skill name before any mutations
		_current_skill_name = self.scratchpad.skill.get(self.skill_turn, "")

		analysis_blocks = self._extract_xml_tag(raw, "analysis")
		command_blocks = self._extract_xml_tag(raw, "command")
		skill_blocks = self._extract_xml_tag(raw, "skill")
		optimized_blocks = self._extract_xml_tag(raw, "optimized_code")
		reference_blocks = self._extract_xml_tag(raw, "reference")
		json_blocks = self._extract_xml_tag(raw, "json")

		output : Dict[str, Any] = {}

		if analysis_blocks:
			self.analysis_turn += 1
			for content in analysis_blocks:
				turn = deepcopy(self.analysis_turn)
				self.scratchpad.analysis[turn] = content
				preview = self._preview_text_for_log(content, max_lines=16, max_chars=1200)
				self._append_text(log_file, f"[DEBUG] analysis turn {turn}: {preview}\n")
			output["analysis"] = self.scratchpad.analysis[self.analysis_turn]

		if command_blocks:
			self.command_turn += 1
			command_result : str = ""
			self._append_text(log_file, f"[DEBUG] Executing {len(command_blocks)} command block(s)\n")
			for content in command_blocks:
				turn = deepcopy(self.command_turn)
				if turn not in self.scratchpad.command:
					self.scratchpad.command[turn] = []
				self.scratchpad.command[turn].append(content)
				self._append_text(log_file, f"[INFO] Executing command: {content}\n")
				_log_exec(content)
				result = self._run_command(content)
				command_result += f"\n[COMMAND]: {content}\n[OUTPUT]:\n{result}\n"
				command_preview = self._preview_text_for_log(result, max_lines=14, max_chars=1200)
				self._append_text(log_file, f"[DEBUG] command turn {turn}: {content}\n{command_preview}\n")
			output["command_result"] = command_result
			self._append_text(log_file, "[DEBUG] Command execution completed\n\n")
			_log_debug("Command execution completed")

		if skill_blocks:
			self.skill_turn += 1
			for content in skill_blocks:
				turn = deepcopy(self.skill_turn)
				self.scratchpad.skill[turn] = content
				self._append_text(log_file, f"[DEBUG] skill turn {turn}: {content}\n")
		
		if reference_blocks:
			self.reference_turn += 1
			reference_result : str = ""
			# Use skill name captured at entry, or fall back to latest
			ref_skill_name = _current_skill_name or self.scratchpad.skill.get(self.skill_turn, "")
			for content in reference_blocks:
				turn = deepcopy(self.reference_turn)
				if turn not in self.scratchpad.reference:
					self.scratchpad.reference[turn] = []
				self.scratchpad.reference[turn].append(content)
				self._append_text(log_file, f"[INFO] Loading reference file: {content}\n")
				_log_debug(f"Loading reference: {content}")
				try:
					result = self._read_text(os.path.join(self.skills_dir, ref_skill_name, "references", content))
				except FileNotFoundError:
					result = f"[WARN] Reference file not found: {content}"
					self._append_text(log_file, f"[WARN] Reference file not found: {content}\n")
				reference_result += f"\n[REFERENCE FILE]: {content}\n[CONTENT]:\n{result}\n"
				self._append_text(log_file, f"[DEBUG] reference turn {turn}: loaded {content}\n")
			output["reference_result"] = reference_result
			self._append_text(log_file, "[DEBUG] Reference loading completed\n\n")
			_log_debug("Reference loading completed")

		if json_blocks:
			json_results: List[Dict[str, Any]] = []
			for content in json_blocks:
				try:
					obj = json.loads(content)
					json_results.append(obj)
					turn = len(self.scratchpad.json_artifact) + 1
					self.scratchpad.json_artifact[turn] = obj
					json_preview = self._preview_text_for_log(json.dumps(obj, indent=2), max_lines=18, max_chars=1200)
					self._append_text(log_file, f"[DEBUG] json artifact turn {turn}: {self._summarize_json_for_log(obj)}\n{json_preview}\n")
				except json.JSONDecodeError:
					self._append_text(log_file, f"[WARN] Invalid JSON artifact ignored:\n{self._preview_text_for_log(content, max_lines=12, max_chars=800)}\n")
			if json_results:
				output["json_artifacts"] = json_results
				
		if optimized_blocks and allow_optimized_code:
			optimized_turns: List[int] = []
			# Derive turn numbers from scratchpad state, not self.optimized_code_turn
			next_turn = (max(self.scratchpad.optimized_code.keys()) + 1) if self.scratchpad.optimized_code else 1
			for content in optimized_blocks:
				turn = next_turn
				next_turn += 1
				self.scratchpad.optimized_code[turn] = content
				code_lines = len(content.splitlines())
				self._append_text(log_file, f"[DEBUG] optimized_code turn {turn}: {code_lines} lines captured\n")
				# Temporarily set for _create_code_file
				self.optimized_code_turn = turn
				self._create_code_file(
					run_dir=os.path.abspath(self.scratchpad.run_dir),
					design_name=self._derive_design_name(self.code_path, os.path.dirname(self.code_path)),
					log_file=log_file,
				)
				optimized_turns.append(turn)
			# Set to last generated turn
			self.optimized_code_turn = optimized_turns[-1]
			output["optimized_turns"] = optimized_turns
		elif optimized_blocks and not allow_optimized_code:
			self._append_text(log_file, f"[WARN] Ignored {len(optimized_blocks)} <optimized_code> block(s) in this phase.\n")
		return output

	def _patch_tcl_add_files(self, tcl_text: str, design_file: str) -> str:
		import re

		lines = tcl_text.splitlines()
		patched = []
		replaced = False
		for line in lines:
			if re.match(r"\s*add_files\b", line):
				patched.append(f"add_files {design_file}")
				replaced = True
			else:
				patched.append(line)
		if not replaced:
			patched.insert(0, f"add_files {design_file}")
		return "\n".join(patched)

	def load_requirement_to_scratchpad(self, req_json_path: str) -> None:
		"""
		Load requirement.json and assign values to scratchpad fields:
		- target_device -> fpga
		- clock_frequency_mhz -> target_frequency_mhz
		- target_optimization -> goal
		"""
		with open(req_json_path, 'r', encoding='utf-8') as f:
			req = json.load(f)
		self.scratchpad.fpga = req.get('target_device', '')
		self.scratchpad.target_frequency_mhz = str(req.get('clock_frequency_mhz', ''))
		self.scratchpad.goal = req.get('target_optimization', '')
		if not self.scratchpad.top_function_name:
			self.scratchpad.top_function_name = req.get('top_function', req.get('top_function_name', ''))

	def _append_text(self, file_path: str, text: str) -> None:
		with open(file_path, "a", encoding="utf-8") as f:
			f.write(text)

	def _read_text(self, path: str) -> str:
		with open(path, "r", encoding="utf-8") as f:
			return f.read()

	def _write_text(self, path: str, content: str) -> None:
		with open(path, "w", encoding="utf-8") as f:
			f.write(content)

	def _next_json_artifact_turn(self) -> int:
		return len(self.scratchpad.json_artifact) + 1

	def _store_json_artifact(self, label: str, data: Dict[str, Any], path: str, log_file: str) -> Dict[str, Any]:
		turn = self._next_json_artifact_turn()
		self.scratchpad.json_artifact[turn] = data
		self.scratchpad.stage_artifacts[label] = path
		preview = self._preview_text_for_log(json.dumps(data, indent=2), max_lines=18, max_chars=1200)
		self._append_text(
			log_file,
			f"[DEBUG] stored json artifact '{label}' at turn {turn}: {path} ({self._summarize_json_for_log(data)})\n{preview}\n",
		)
		return {"json_artifacts": [data]}

	def _run_python_script(
		self,
		script_path: str,
		args: List[str],
		log_file: str,
		stage_label: str,
		cwd: str | None = None,
		timeout: int | None = None,
	) -> tuple[int, str]:
		argv = [sys.executable, script_path, *args]
		run_dir = cwd or self.scratchpad.run_dir or os.path.abspath(self.out_dir)
		self._append_text(log_file, f"[INFO] [{stage_label}] Executing: {' '.join(argv)}\n")
		try:
			effective_timeout = timeout
			if effective_timeout is None:
				effective_timeout = int(self.config.get("command_timeout", 1800))
			if effective_timeout is not None and effective_timeout <= 0:
				effective_timeout = None
			result = subprocess.run(
				argv,
				shell=False,
				check=False,
				text=True,
				cwd=run_dir,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				timeout=effective_timeout,
			)
			output = result.stdout
			output_preview = self._preview_text_for_log(output, max_lines=24, max_chars=1600)
			self._append_text(log_file, f"[DEBUG] [{stage_label}] return_code={result.returncode}\n{output_preview}\n")
			return result.returncode, output
		except subprocess.TimeoutExpired:
			output = f"[ERROR] Script timed out: {' '.join(argv)}"
			self._append_text(log_file, f"[ERROR] [{stage_label}] {output}\n")
			return -1, output

	def _load_json_file(self, path: str) -> Dict[str, Any]:
		return json.loads(self._read_text(path))

	def _resolve_pragma_dse_jobs(self) -> int:
		raw = self.config.get("pragma_dse_jobs", 4)
		try:
			value = int(raw)
		except (TypeError, ValueError):
			value = 4
		if value > 0:
			return value
		return 4

	def _resolve_positive_int_config(self, key: str, default: int) -> int:
		raw = self.config.get(key, default)
		try:
			value = int(raw)
		except (TypeError, ValueError):
			value = default
		if value > 0:
			return value
		return default

	def _resolve_non_negative_int_config(self, key: str, default: int) -> int:
		raw = self.config.get(key, default)
		try:
			value = int(raw)
		except (TypeError, ValueError):
			value = default
		if value >= 0:
			return value
		return default

	def _resolve_pragma_dse_search_strategy(self) -> str:
		raw = str(self.config.get("pragma_dse_search_strategy", "progressive") or "").strip().lower()
		if raw in {"singles", "progressive"}:
			return raw
		return "progressive"
	
	def _clock_period_from_target_freq(self) -> str:
		target = float(self.scratchpad.target_frequency_mhz)
		return f"{1000.0 / target:.3f}"

	def _find_csynth_report(self, root_dir: str) -> str:
		matches = sorted(Path(root_dir).glob("**/syn/report/csynth.xml"))
		return str(matches[0]) if matches else ""

	def _run_command(self, command: str) -> str:
		"""Run a shell command in run_dir and return combined stdout/stderr output."""
		forbidden = ["&&", "||", ";", "|", "`", "$(", ">", "<"]
		if any(token in command for token in forbidden):
			return f"[BLOCKED] Unsafe shell token found in command: {command}"

		try:
			argv = shlex.split(command)
		except ValueError as exc:
			return f"[BLOCKED] Failed to parse command: {exc}"
		if not argv:
			return "[BLOCKED] Empty command"

		allowed_bins = {"python", "python3", "clang++", "g++", "vitis-run"}
		exe = os.path.basename(argv[0])
		if exe not in allowed_bins:
			return f"[BLOCKED] Command not allowed: {exe}"
		if exe in {"python", "python3"}:
			argv[0] = sys.executable

		run_dir = self.scratchpad.run_dir or os.path.abspath(self.out_dir)
		timeout = int(self.config.get("command_timeout", 600))  # 10 min default
		try:
			result = subprocess.run(
				argv,
				shell=False,
				check=False,
				text=True,
				cwd=run_dir,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				timeout=timeout,
			)
			return f"[RETURN_CODE]={result.returncode}\n{result.stdout}"
		except subprocess.TimeoutExpired:
			return f"[RETURN_CODE]=-1\n[ERROR] Command timed out after {timeout}s: {command}"
	
	#TODO: Currently only extract python commands, need to extend to other shell commands if needed.
	def _extract_python_commands(self, script: str) -> list[str]:
		"""
		Given a multi-line shell script string, extract valid python commands (one per line).
		Only lines starting with 'python ' and not commented out are considered.
		Returns a list of command strings suitable for subprocess execution.
		"""
		commands = []
		for line in script.splitlines():
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			# Remove inline comments
			if '#' in line:
				line = line.split('#', 1)[0].strip()
			if line.startswith('python '):
				commands.append(line)
		return commands

	def _derive_design_name(self, code_file: str, code_root: str) -> str:
		if os.path.isdir(code_root):
			rel = os.path.relpath(code_file, code_root)
			first = rel.split(os.sep)[0]
			return os.path.splitext(first)[0]
		return os.path.splitext(os.path.basename(code_file))[0]

	def _sanitize_path_component(self, text: str, default: str = "unknown") -> str:
		value = str(text or "").strip()
		value = re.sub(r"[\\/]+", "_", value)
		value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
		value = re.sub(r"_+", "_", value).strip("._-")
		return value or default

	def _current_model_dir_name(self) -> str:
		return self._sanitize_path_component(str(self.config.get("model", "")).strip(), default="unknown_model")

	def _run_dir_name(self, design_name: str, run_id: str) -> str:
		return f"{design_name}_run_{run_id}"

	def _run_relative_path(self, run_dir: str) -> str:
		if not run_dir:
			return ""
		try:
			rel = os.path.relpath(run_dir, os.path.abspath(self.out_dir))
		except ValueError:
			rel = ""
		if rel and rel != "." and not rel.startswith(".."):
			return rel
		return os.path.basename(os.path.normpath(run_dir))
		
	def _create_code_file(self, run_dir: str, design_name: str, log_file : str) -> None:
		design_file = design_name + "_v" + str(self.optimized_code_turn) + "_opt.cc"
		design_path = os.path.join(run_dir, design_file)
		self.scratchpad.optimized_code_file[self.optimized_code_turn] = design_path
		code = self.scratchpad.optimized_code[self.optimized_code_turn]
		self._write_text(design_path, code)
		self._append_text(log_file, f"[INFO] Created code file: {design_path}\n")
		_log_file(f"Created: {_c(design_path, '1;37')}")
	
	# TODO: Be careful about using the restore method since currently I don't tell the difference between restore code version and current optimized code version.
	def _restore_code_from_scratchpad(self, run_dir:str, design_name: str, restore_version: int, log_file: str) ->	None:
		restore_design_file = design_name + "_restore_v" + str(restore_version) + "_opt.cc"
		design_path = os.path.join(run_dir, restore_design_file)
		self.scratchpad.optimized_code_turn = restore_version
		self.scratchpad.optimized_code_file[restore_version] = design_path
		code = self.scratchpad.optimized_code[restore_version]
		self._write_text(design_path, code)
		self._append_text(log_file, f"[INFO] Restored code file: {design_path}\n")
		_log_file(f"Restored: {_c(design_path, '1;37')}")

	def _max_hardware_opt_rounds(self) -> int:
		return max(1, int(self.config.get("max_hardware_opt_rounds", 10)))

	def _find_matching_brace(self, text: str, open_index: int) -> int:
		depth = 0
		for idx in range(open_index, len(text)):
			char = text[idx]
			if char == "{":
				depth += 1
			elif char == "}":
				depth -= 1
				if depth == 0:
					return idx
		return -1

	def _extract_function_body(self, code: str, function_name: str) -> str:
		pattern = re.compile(rf"\b{re.escape(function_name)}\s*\([^{{;]*\)\s*\{{", re.MULTILINE)
		match = pattern.search(code)
		if not match:
			return ""
		open_index = code.find("{", match.start())
		if open_index < 0:
			return ""
		close_index = self._find_matching_brace(code, open_index)
		if close_index < 0:
			return ""
		return code[open_index + 1 : close_index]

	def _split_c_args(self, args_text: str) -> List[str]:
		args: List[str] = []
		current: List[str] = []
		depth = 0
		for char in args_text:
			if char == "," and depth == 0:
				arg = "".join(current).strip()
				if arg:
					args.append(arg)
				current = []
				continue
			current.append(char)
			if char in "([{<":
				depth += 1
			elif char in ")]}>":
				depth = max(0, depth - 1)
		trailing = "".join(current).strip()
		if trailing:
			args.append(trailing)
		return args

	def _validate_hardware_dataflow(self, code: str) -> List[str]:
		top_body = self._extract_function_body(code, self.scratchpad.top_function_name)
		if not top_body:
			return []
		if not re.search(r"#\s*pragma\s+HLS\s+dataflow\b", top_body, re.IGNORECASE):
			return []

		m_axi_ports = self._extract_m_axi_interface_ports(code)
		if not m_axi_ports:
			return []

		port_usage: Dict[str, set[str]] = {port: set() for port in m_axi_ports}
		call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\((.*?)\)\s*;", re.DOTALL)
		for match in call_re.finditer(top_body):
			callee = match.group(1)
			if callee in {"if", "for", "while", "switch", "return"}:
				continue
			for arg in self._split_c_args(match.group(2)):
				token_match = re.match(r"([A-Za-z_]\w*)", arg.strip())
				if not token_match:
					continue
				token = token_match.group(1)
				if token in port_usage:
					port_usage[token].add(callee)

		issues: List[str] = []
		for port, bundle in m_axi_ports.items():
			callees = sorted(port_usage.get(port, set()))
			if len(callees) > 1:
				issues.append(
					f"Invalid DATAFLOW pattern: m_axi port '{port}' on bundle '{bundle}' is consumed by multiple processes {callees}. "
					"Vitis rejects shared external-memory reads across DATAFLOW processes; stage the data into local buffers/streams or remove DATAFLOW."
					)
		return issues

	def _validate_function_scope_pragmas(self, code: str) -> List[str]:
		issues: List[str] = []
		brace_depth = 0
		in_block_comment = False
		inline_re = re.compile(r"#\s*pragma\s+HLS\s+inline\b", re.IGNORECASE)

		for idx, raw_line in enumerate(code.splitlines(), start=1):
			line = raw_line
			cleaned_parts: List[str] = []
			cursor = 0
			while cursor < len(line):
				if in_block_comment:
					end = line.find("*/", cursor)
					if end < 0:
						cursor = len(line)
						break
					in_block_comment = False
					cursor = end + 2
					continue
				start = line.find("/*", cursor)
				line_comment = line.find("//", cursor)
				if line_comment >= 0 and (start < 0 or line_comment < start):
					cleaned_parts.append(line[cursor:line_comment])
					cursor = len(line)
					break
				if start < 0:
					cleaned_parts.append(line[cursor:])
					cursor = len(line)
					break
				cleaned_parts.append(line[cursor:start])
				cursor = start + 2
				in_block_comment = True
			cleaned = "".join(cleaned_parts)
			stripped = cleaned.strip()
			if stripped and inline_re.search(stripped) and brace_depth == 0:
				issues.append(
					f"Invalid HLS pragma placement at line {idx}: '#pragma HLS inline' is outside any function body. "
					"Place INLINE inside the function body it applies to, not immediately before the function definition at file scope."
				)
			brace_depth += cleaned.count("{")
			brace_depth -= cleaned.count("}")
			brace_depth = max(0, brace_depth)

		return issues

	def _extract_m_axi_interface_ports(self, code: str) -> Dict[str, str]:
		top_body = self._extract_function_body(code, self.scratchpad.top_function_name)
		if not top_body:
			return {}
		interface_re = re.compile(
			r"#\s*pragma\s+HLS\s+interface\s+(?:mode\s*=\s*)?m_axi\b[^\n]*\bport\s*=\s*([A-Za-z_]\w*)\b(?:[^\n]*\bbundle\s*=\s*([A-Za-z_]\w*))?",
			re.IGNORECASE,
		)
		m_axi_ports: Dict[str, str] = {}
		for match in interface_re.finditer(top_body):
			port = match.group(1)
			bundle = match.group(2) or port
			m_axi_ports[port] = bundle
		return m_axi_ports

	def _collect_array_layout_pragmas(self, code: str) -> List[Dict[str, Any]]:
		layout_re = re.compile(
			r"#\s*pragma\s+HLS\s+(array_partition|array_reshape)\b[^\n]*\bvariable\s*=\s*([A-Za-z_]\w*)",
			re.IGNORECASE,
		)
		sites: List[Dict[str, Any]] = []
		for idx, line in enumerate(code.splitlines(), start=1):
			match = layout_re.search(line)
			if not match:
				continue
			sites.append(
				{
					"line": idx,
					"kind": match.group(1).lower(),
					"variable": match.group(2),
					"text": line.rstrip(),
					}
				)
		return sites

	def _collect_local_arrays(self, code: str) -> List[Dict[str, Any]]:
		array_re = re.compile(
			r"^\s*(?:static\s+)?(?:const\s+)?(?:[\w:<>]+\s+)*([A-Za-z_]\w*)\s*\[(\d+)\]\s*(?:\[[^\]]+\]\s*)*;",
			re.MULTILINE,
		)
		arrays: List[Dict[str, Any]] = []
		for idx, line in enumerate(code.splitlines(), start=1):
			stripped = line.strip()
			if not stripped or stripped.startswith("//") or stripped.startswith("#pragma"):
				continue
			match = array_re.match(line)
			if not match:
				continue
			arrays.append(
				{
					"line": idx,
					"variable": match.group(1),
					"text": line.rstrip(),
				}
			)
		return arrays

	def _filter_hardware_tuning_targets(
		self,
		code: str,
		targets: List[Dict[str, Any]],
		log_file: str,
	) -> tuple[List[Dict[str, Any]], List[str]]:
		if not targets:
			return [], []
		m_axi_ports = self._extract_m_axi_interface_ports(code)
		layout_sites = self._collect_array_layout_pragmas(code)
		filtered: List[Dict[str, Any]] = []
		warnings: List[str] = []
		for target in targets:
			kind = str(target.get("kind", "")).strip().lower()
			if kind not in {"array_partition", "array_reshape"}:
				filtered.append(target)
				continue
			target_line = 0
			try:
				target_line = int(target.get("line") or 0)
			except (TypeError, ValueError):
				target_line = 0
			matching_sites = [site for site in layout_sites if site["kind"] == kind]
			if not matching_sites:
				warnings.append(
					f"Dropped tuning target {kind} near line {target_line}: no matching pragma site exists in the hardware rewrite output. "
					"If this knob should align with an UNROLL factor on external m_axi data, first stage a local buffer/window and partition that local array instead."
				)
				continue
			site = min(
				matching_sites,
				key=lambda item: (
					abs(int(item["line"]) - target_line),
					0 if int(item["line"]) >= target_line else 1,
					int(item["line"]),
				),
			)
			if site["variable"] in m_axi_ports:
				warnings.append(
					f"Dropped tuning target {kind} for variable '{site['variable']}' near line {site['line']}: "
					"ARRAY_PARTITION/RESHAPE on a top-level m_axi interface is not a valid throughput knob. "
					"If UNROLL requires aligned memory parallelism, rewrite the design to load a local on-chip buffer/window and partition that local buffer."
				)
				continue
			filtered.append(target)
		if warnings:
			self._append_text(
				log_file,
				"[WARN] Hardware rewrite tuning plan adjusted:\n"
				+ "\n".join(f"  - {item}" for item in warnings)
					+ "\n",
				)
		return filtered, warnings

	def _filter_interface_tuning_targets(
		self,
		code: str,
		targets: List[Dict[str, Any]],
		log_file: str,
	) -> tuple[List[Dict[str, Any]], List[str]]:
		if not targets:
			return [], []
		m_axi_ports = self._extract_m_axi_interface_ports(code)
		if not m_axi_ports:
			warnings = ["Dropped interface tuning targets: the hardware rewrite output does not declare any explicit m_axi interface pragmas."]
			self._append_text(log_file, "[WARN] Hardware rewrite interface tuning targets adjusted:\n  - " + warnings[0] + "\n")
			return [], warnings
		per_port_options = {"latency", "num_read_outstanding", "num_write_outstanding", "max_read_burst_length", "max_write_burst_length"}
		global_options = {"m_axi_auto_max_ports", "m_axi_alignment_byte_size", "m_axi_max_widen_bitwidth", "m_axi_conservative_mode"}
		filtered: List[Dict[str, Any]] = []
		warnings: List[str] = []
		for target in targets:
			option = str(target.get("option", "")).strip().lower().lstrip("-")
			port = str(target.get("port", "")).strip()
			if port:
				if port not in m_axi_ports:
					warnings.append(f"Dropped interface target for port '{port}': no matching explicit m_axi interface exists in the hardware rewrite output.")
					continue
				if option not in per_port_options:
					warnings.append(f"Dropped interface target for port '{port}': option '{option}' is not supported by the current interface DSE layer.")
					continue
			else:
				if option not in global_options:
					warnings.append(f"Dropped interface target with no port: option '{option}' is not supported by the current interface DSE layer.")
					continue
			filtered.append(target)
		if warnings:
			self._append_text(
				log_file,
				"[WARN] Hardware rewrite interface tuning targets adjusted:\n"
				+ "\n".join(f"  - {item}" for item in warnings)
				+ "\n",
			)
		return filtered, warnings

	def _filter_storage_tuning_targets(
		self,
		code: str,
		targets: List[Dict[str, Any]],
		log_file: str,
	) -> tuple[List[Dict[str, Any]], List[str]]:
		if not targets:
			return [], []
		local_arrays = {item["variable"] for item in self._collect_local_arrays(code)}
		m_axi_ports = set(self._extract_m_axi_interface_ports(code))
		filtered: List[Dict[str, Any]] = []
		warnings: List[str] = []
		for target in targets:
			variable = str(target.get("variable", "")).strip()
			if not variable:
				warnings.append("Dropped storage target with no variable name.")
				continue
			if variable in m_axi_ports:
				warnings.append(f"Dropped storage target for '{variable}': top-level m_axi interfaces are not valid bind_storage targets.")
				continue
			if variable not in local_arrays:
				warnings.append(f"Dropped storage target for '{variable}': no matching local array declaration was found in the hardware rewrite output.")
				continue
			filtered.append(target)
		if warnings:
			self._append_text(
				log_file,
				"[WARN] Hardware rewrite storage tuning targets adjusted:\n"
				+ "\n".join(f"  - {item}" for item in warnings)
				+ "\n",
			)
		return filtered, warnings

	def _filter_op_tuning_targets(
		self,
		code: str,
		targets: List[Dict[str, Any]],
		log_file: str,
	) -> tuple[List[Dict[str, Any]], List[str]]:
		if not targets:
			return [], []
		supported_ops = {"mul", "add", "sub", "fmul", "fadd", "fsub", "dmul", "dadd", "dsub"}
		filtered: List[Dict[str, Any]] = []
		warnings: List[str] = []
		for target in targets:
			op = str(target.get("op", "")).strip().lower()
			if op not in supported_ops:
				warnings.append(f"Dropped operator target '{op}': op is not supported by the current config_op-based DSE layer.")
				continue
			filtered.append(target)
		if warnings:
			self._append_text(
				log_file,
				"[WARN] Hardware rewrite operator tuning targets adjusted:\n"
				+ "\n".join(f"  - {item}" for item in warnings)
				+ "\n",
			)
		return filtered, warnings

	def _filter_dataflow_tuning_targets(
		self,
		code: str,
		targets: List[Dict[str, Any]],
		log_file: str,
	) -> tuple[List[Dict[str, Any]], List[str]]:
		if not targets:
			return [], []
		if re.search(r"#\s*pragma\s+HLS\s+dataflow\b", code, re.IGNORECASE) is None:
			warnings = ["Dropped dataflow tuning targets: the hardware rewrite output does not contain an explicit HLS DATAFLOW pragma."]
			self._append_text(log_file, "[WARN] Hardware rewrite dataflow tuning targets adjusted:\n  - " + warnings[0] + "\n")
			return [], warnings
		supported_options = {"default_channel", "fifo_depth", "start_fifo_depth", "scalar_fifo_depth", "task_level_fifo_depth"}
		filtered: List[Dict[str, Any]] = []
		warnings: List[str] = []
		for target in targets:
			option = str(target.get("option", "")).strip().lower().lstrip("-")
			if option not in supported_options:
				warnings.append(f"Dropped dataflow target '{option}': option is not supported by the current DSE layer.")
				continue
			filtered.append(target)
		if warnings:
			self._append_text(
				log_file,
				"[WARN] Hardware rewrite dataflow tuning targets adjusted:\n"
				+ "\n".join(f"  - {item}" for item in warnings)
				+ "\n",
			)
		return filtered, warnings

	def _validate_hardware_candidate(self, turn: int, log_file: str) -> List[str]:
		code = self.scratchpad.optimized_code.get(turn, "")
		issues = self._validate_hardware_dataflow(code)
		issues.extend(self._validate_function_scope_pragmas(code))
		if issues:
			self._append_text(
				log_file,
				"[WARN] Hardware rewrite candidate rejected before pragma stages:\n"
				+ "\n".join(f"  - {issue}" for issue in issues)
				+ "\n",
			)
		return issues

	def _sanitize_vitis_message(self, message: str) -> str:
		cleaned = message.strip()
		cleaned = re.sub(r"\s*\([^)]*:[0-9:]+\)", "", cleaned)
		return cleaned

	def _extract_vitis_log_issues(self, run_dir: str, limit: int = 6) -> List[str]:
		issues: List[str] = []
		seen: set[str] = set()
		ignored_patterns = (
			"config_rtl -enable_maxiConservative",
			"Ignore array_reshape applied to 'A' which is a m_axi interface",
		)
		issue_markers = ("error:", "warning:", "[timeout]", "timed out")
		for rel_path in ("vitis_run.log", "vitis_hls.log", os.path.join("logs", "hls_run_tcl.log")):
			log_path = os.path.join(run_dir, rel_path)
			if not os.path.exists(log_path):
				continue
			for line in self._read_text(log_path).splitlines():
				lower_line = line.lower()
				if not any(marker in lower_line for marker in issue_markers):
					continue
				message = self._sanitize_vitis_message(line)
				if any(pattern in message for pattern in ignored_patterns):
					continue
				if message in seen:
					continue
				seen.add(message)
				issues.append(message)
				if len(issues) >= limit:
					return issues
		return issues

	def _collect_recent_vitis_diagnostics(self, max_items: int = 8) -> List[str]:
		diagnostics: List[str] = []
		seen: set[str] = set()

		def _add(message: str) -> None:
			text = str(message).strip()
			if not text or text in seen:
				return
			seen.add(text)
			diagnostics.append(text)

		report_path = self.scratchpad.stage_artifacts.get("pragma_dse_report", "")
		if report_path and os.path.exists(report_path):
			try:
				report = self._load_json_file(report_path)
			except Exception:
				report = {}
			for item in list(report.get("results", []) or [])[: max(max_items, 8)]:
				_add(item.get("error", ""))
				for warning in list(item.get("warnings", []) or [])[:3]:
					_add(warning)
					if len(diagnostics) >= max_items:
						return diagnostics[:max_items]
				run_dir = str(item.get("run_dir", "")).strip()
				if run_dir:
					for issue in self._extract_vitis_log_issues(run_dir, limit=3):
						_add(issue)
						if len(diagnostics) >= max_items:
							return diagnostics[:max_items]

		profiling_path = self.scratchpad.stage_artifacts.get("profiling_report", "")
		if profiling_path and os.path.exists(profiling_path):
			try:
				report = self._load_json_file(profiling_path)
			except Exception:
				report = {}
			for key in ("warnings", "issues", "critical_warnings"):
				for message in list(report.get(key, []) or []):
					_add(message)
					if len(diagnostics) >= max_items:
						return diagnostics[:max_items]

		return diagnostics[:max_items]

	def _summarize_pragma_tuning_dead_end(self, candidate_payload: Dict[str, Any]) -> str:
		candidate_count = int(candidate_payload.get("candidate_count", 0) or 0)
		if candidate_count > 0:
			return ""
		lines = [
			"Previous hardware rewrite left no valid tuning candidates for the current DSE layer.",
			"Revise the hardware rewrite so that it exposes useful knobs, such as PIPELINE II, UNROLL factor, ARRAY_PARTITION/RESHAPE factor, STREAM depth, explicit m_axi interfaces, local arrays suitable for bind_storage, or legal DATAFLOW regions.",
		]
		for skipped in list(candidate_payload.get("skipped_pragmas", candidate_payload.get("skipped_loops", [])))[:6]:
			line_no = skipped.get("line", "?")
			kind = skipped.get("kind", "pragma")
			reason = skipped.get("reason", "site was skipped")
			lines.append(f"- {kind} pragma around line {line_no}: {reason}")
		return "\n".join(lines)

	def _summarize_pragma_dse_dead_end(self, report: Dict[str, Any]) -> str:
		results = list(report.get("results", []) or [])
		successful = [item for item in results if item.get("success")]
		if successful:
			return ""
		evaluated_count = int(report.get("evaluated_count", len(results)) or 0)
		issue_counter: Counter[str] = Counter()
		for item in results[: min(8, len(results))]:
			error = str(item.get("error", "")).strip()
			if error:
				issue_counter[self._sanitize_vitis_message(error)] += 1
			for warning in list(item.get("warnings", []) or [])[:3]:
				warning_text = str(warning).strip()
				if warning_text:
					issue_counter[self._sanitize_vitis_message(warning_text)] += 1
			for issue in self._extract_vitis_log_issues(str(item.get("run_dir", ""))):
				issue_counter[issue] += 1

		lines = [
			f"Previous pragma DSE attempt produced no acceptable candidates across {evaluated_count} evaluations.",
		]
		for message, count in issue_counter.most_common(5):
			lines.append(f"- {message} (seen in {count} candidate logs)")
		if any("timed out" in message.lower() for message in issue_counter):
			lines.append(
				"Timeout recovery rule: interpret Vitis HLS timeout as evidence that the current hardware optimization strategy is too aggressive, not just that the timeout budget is too small."
			)
			lines.append(
				"Revise the next hardware rewrite conservatively: reduce UNROLL factors, relax PIPELINE targets by increasing II (for example II=2 or II=4 instead of II=1), and simplify or remove aggressive DATAFLOW/INLINE combinations."
			)
			lines.append(
				"If STREAM/FIFO channels are present in a DATAFLOW design, increase STREAM depth/FIFO depth to reduce scheduling pressure before trying more parallelism again."
			)
			lines.append(
				"Narrow the next pragma-tuning/DSE space around conservative settings first; do not keep exploring large UNROLL factors or many aggressive combinations after timeout."
			)
		if any("Bundled bus interface" in message and "dataflow" in message.lower() for message in issue_counter):
			lines.append(
				"Avoid DATAFLOW processes that directly share the same m_axi port/bundle. "
				"Buffer external arrays locally, stream staged data between processes, or remove DATAFLOW."
			)
		if any("can only be applied inside loop body" in message for message in issue_counter):
			lines.append(
				"Keep pragma placement structurally legal for Vitis HLS. Attach tunable parameters to the pragma sites created by hardware rewrite instead of synthesizing invalid top-level replacements."
			)
		return "\n".join(lines)

	def _derive_rewrite_retrospective(
		self,
		design_name: str,
		run_dir: str,
		final_summary: Dict[str, Any],
	) -> Dict[str, Any]:
		run_id = os.path.basename(run_dir.rstrip(os.sep))
		stage_results = final_summary.get("stage_results", {}) or {}
		software_candidates = [
			candidate
			for candidate in self.candidates
			if candidate.variant_kind == "software" and candidate.stage.startswith("attempt_")
		]
		software_failures: List[Dict[str, Any]] = []
		for candidate in software_candidates:
			if candidate.verification_pass:
				continue
			software_failures.append(
				{
					"turn": candidate.turn,
					"stage": candidate.stage,
					"notes": candidate.notes,
				}
			)

		candidate_payload: Dict[str, Any] = {}
		candidates_path = self.scratchpad.stage_artifacts.get("pragma_candidates", "")
		if candidates_path and os.path.exists(candidates_path):
			candidate_payload = self._load_json_file(candidates_path)

		report: Dict[str, Any] = {}
		report_path = self.scratchpad.stage_artifacts.get("pragma_dse_report", "")
		if report_path and os.path.exists(report_path):
			report = self._load_json_file(report_path)

		hardware_issues: List[str] = []
		for stage_name in ("hardware_rewrite", "pragma_tuning", "pragma_dse"):
			stage_info = stage_results.get(stage_name, {}) or {}
			reason = str(stage_info.get("reason", "")).strip()
			error = str(stage_info.get("error", "")).strip()
			if reason:
				hardware_issues.append(reason)
			if error:
				hardware_issues.append(error)

		for item in report.get("results", []) or []:
			for issue in self._extract_vitis_log_issues(str(item.get("run_dir", "")), limit=8):
				hardware_issues.append(issue)

		hardware_attempts = int(final_summary.get("hardware_attempt", 0) or 0)
		if hardware_attempts <= 0:
			stage_attempt = int(stage_results.get("hardware_rewrite", {}).get("attempt", 0) or 0)
			hardware_attempts = stage_attempt or len(
				[candidate for candidate in self.candidates if candidate.variant_kind == "hardware"]
			)
		if hardware_attempts <= 0 and final_summary.get("hardware_turn"):
			hardware_attempts = 1

		return {
			"run_id": run_id,
			"design_name": design_name,
			"software_loop": {
				"attempt_count": len(software_candidates),
				"failure_count": len(software_failures),
				"failures": software_failures,
			},
			"hardware_loop": {
				"hardware_attempts": hardware_attempts,
				"pragma_candidate_count": int(candidate_payload.get("candidate_count", 0) or 0),
				"pragma_dse_successful_candidate_count": int(report.get("successful_candidate_count", 0) or 0),
				"issues": hardware_issues[:12],
			},
		}

	def _rewrite_lessons_section_specs(self) -> List[Dict[str, str]]:
		return [
			{
				"category": "software_rewrite_guardrails",
				"heading": "## 1. Software Rewrite Guardrails",
				"prefix": "1",
				"label": "Software Rewrite Guardrails",
			},
			{
				"category": "hardware_rewrite_guardrails",
				"heading": "## 2. Hardware Rewrite Guardrails",
				"prefix": "2",
				"label": "Hardware Rewrite Guardrails",
			},
			{
				"category": "memory_and_axi_bottlenecks",
				"heading": "## 3. Memory and AXI Bottlenecks",
				"prefix": "3",
				"label": "Memory and AXI Bottlenecks",
			},
			{
				"category": "pragma_placement_and_combination_rules",
				"heading": "## 4. Pragma Placement and Combination Rules",
				"prefix": "4",
				"label": "Pragma Placement and Combination Rules",
			},
			{
				"category": "timing_and_arithmetic_lessons",
				"heading": "## 5. Timing and Arithmetic Lessons",
				"prefix": "5",
				"label": "Timing and Arithmetic Lessons",
			},
			{
				"category": "pragma_tuning_and_dse_lessons",
				"heading": "## 6. Pragma-Tuning and DSE Lessons",
				"prefix": "6",
				"label": "Pragma-Tuning and DSE Lessons",
			},
			{
				"category": "tooling_and_flow_compatibility",
				"heading": "## 7. Tooling and Flow Compatibility",
				"prefix": "7",
				"label": "Tooling and Flow Compatibility",
			},
			{
				"category": "high_priority_rewrite_heuristics",
				"heading": "## 8. High-Priority Rewrite Heuristics",
				"prefix": "8",
				"label": "High-Priority Rewrite Heuristics",
			},
		]

	def _normalize_rewrite_lesson_category(self, category: str) -> str:
		key = re.sub(r"[^a-z0-9]+", "_", str(category or "").strip().lower()).strip("_")
		if not key:
			return "tooling_and_flow_compatibility"
		aliases = {
			"software": "software_rewrite_guardrails",
			"software_rewrite": "software_rewrite_guardrails",
			"software_rewrite_guardrails": "software_rewrite_guardrails",
			"software_guardrails": "software_rewrite_guardrails",
			"hardware": "hardware_rewrite_guardrails",
			"hardware_rewrite": "hardware_rewrite_guardrails",
			"hardware_rewrite_guardrails": "hardware_rewrite_guardrails",
			"hardware_guardrails": "hardware_rewrite_guardrails",
			"memory": "memory_and_axi_bottlenecks",
			"memory_and_axi": "memory_and_axi_bottlenecks",
			"memory_and_axi_bottlenecks": "memory_and_axi_bottlenecks",
			"axi": "memory_and_axi_bottlenecks",
			"pragma": "pragma_placement_and_combination_rules",
			"pragma_placement": "pragma_placement_and_combination_rules",
			"pragma_placement_and_combination_rules": "pragma_placement_and_combination_rules",
			"timing": "timing_and_arithmetic_lessons",
			"timing_and_arithmetic": "timing_and_arithmetic_lessons",
			"timing_and_arithmetic_lessons": "timing_and_arithmetic_lessons",
			"pragma_tuning": "pragma_tuning_and_dse_lessons",
			"dse": "pragma_tuning_and_dse_lessons",
			"pragma_tuning_and_dse_lessons": "pragma_tuning_and_dse_lessons",
			"tooling": "tooling_and_flow_compatibility",
			"tooling_and_flow": "tooling_and_flow_compatibility",
			"tooling_and_flow_compatibility": "tooling_and_flow_compatibility",
			"heuristics": "high_priority_rewrite_heuristics",
			"high_priority_rewrite_heuristics": "high_priority_rewrite_heuristics",
		}
		return aliases.get(key, "tooling_and_flow_compatibility")

	def _sanitize_rewrite_lesson_entry(
		self,
		lesson: Dict[str, Any],
		default_design: str,
	) -> Dict[str, Any] | None:
		category = self._normalize_rewrite_lesson_category(str(lesson.get("category", "")))
		title = str(lesson.get("title", "")).strip()
		rule = str(lesson.get("rule", "")).strip()
		why = str(lesson.get("why", "")).strip()
		symptom = str(lesson.get("symptom", "")).strip()
		notes_raw = lesson.get("notes", [])
		notes: List[str] = []
		if isinstance(notes_raw, list):
			for item in notes_raw:
				text = str(item).strip()
				if text:
					notes.append(text)
		else:
			text = str(notes_raw).strip()
			if text:
				notes.append(text)
		seen_in_raw = lesson.get("seen_in", [])
		seen_in: List[str] = []
		if isinstance(seen_in_raw, list):
			for item in seen_in_raw:
				text = str(item).strip()
				if text:
					seen_in.append(text)
		else:
			text = str(seen_in_raw).strip()
			if text:
				seen_in.append(text)
		if default_design:
			seen_in.append(default_design)
		seen_in = sorted({item for item in seen_in if item})
		if not title or not rule or not why:
			return None
		return {
			"category": category,
			"title": title[:120],
			"rule": rule[:400],
			"why": why[:400],
			"symptom": symptom[:240],
			"notes": self._dedupe_compact_strings(notes, max_items=4, max_len=240),
			"seen_in": seen_in,
		}

	def _dedupe_compact_strings(
		self,
		items: List[str],
		*,
		max_items: int | None = None,
		max_len: int = 240,
	) -> List[str]:
		deduped: List[str] = []
		seen_keys = set()
		for item in items:
			text = re.sub(r"\s+", " ", str(item or "").strip())
			if not text:
				continue
			if len(text) > max_len:
				text = text[: max_len - 3].rstrip() + "..."
			key = self._normalize_text_for_compare(text)
			if not key or key in seen_keys:
				continue
			seen_keys.add(key)
			deduped.append(text)
			if max_items is not None and len(deduped) >= max_items:
				break
		return deduped

	def _rewrite_lesson_category_rank(self, category: str) -> int:
		for idx, spec in enumerate(self._rewrite_lessons_section_specs()):
			if spec["category"] == category:
				return idx
		return len(self._rewrite_lessons_section_specs())

	def _merge_rewrite_lessons_locally(self, lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
		for lesson in lessons:
			sanitized = self._sanitize_rewrite_lesson_entry(lesson, "")
			if not sanitized:
				continue
			key = (
				self._normalize_text_for_compare(sanitized.get("category", "")),
				self._normalize_text_for_compare(sanitized.get("title", "")),
				self._normalize_text_for_compare(sanitized.get("rule", "")),
			)
			existing = merged.get(key)
			if existing is None:
				merged[key] = deepcopy(sanitized)
				continue
			if len(str(sanitized.get("why", "")).strip()) > len(str(existing.get("why", "")).strip()):
				existing["why"] = sanitized.get("why", "")
			if not str(existing.get("symptom", "")).strip() and str(sanitized.get("symptom", "")).strip():
				existing["symptom"] = sanitized.get("symptom", "")
			existing["seen_in"] = sorted(
				{
					str(item).strip()
					for item in list(existing.get("seen_in", []) or []) + list(sanitized.get("seen_in", []) or [])
					if str(item).strip()
				}
			)
			existing["notes"] = self._dedupe_compact_strings(
				list(existing.get("notes", []) or []) + list(sanitized.get("notes", []) or []),
				max_items=4,
				max_len=240,
			)
		return list(merged.values())

	def _sort_rewrite_lessons(self, lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		return sorted(
			lessons,
			key=lambda lesson: (
				self._rewrite_lesson_category_rank(str(lesson.get("category", ""))),
				-len(lesson.get("seen_in", []) or []),
				self._normalize_text_for_compare(lesson.get("title", "")),
			),
		)

	def _limit_rewrite_lessons_per_category(
		self,
		lessons: List[Dict[str, Any]],
		*,
		max_per_category: int = 4,
	) -> List[Dict[str, Any]]:
		counts: Counter[str] = Counter()
		limited: List[Dict[str, Any]] = []
		for lesson in self._sort_rewrite_lessons(lessons):
			category = str(lesson.get("category", "")).strip()
			if counts[category] >= max_per_category:
				continue
			limited.append(lesson)
			counts[category] += 1
		return limited

	def _sanitize_rewrite_summary_points(self, summary_raw: Any, *, max_items: int = 6) -> List[str]:
		items: List[str] = []
		if isinstance(summary_raw, list):
			for item in summary_raw:
				text = str(item).strip()
				if text:
					items.append(text)
		elif isinstance(summary_raw, str):
			for line in summary_raw.splitlines():
				text = line.strip()
				if text:
					items.append(text)
		cleaned: List[str] = []
		for item in items:
			text = str(item).strip()
			if text.startswith("- "):
				text = text[2:].strip()
			elif text.startswith("-"):
				text = text[1:].strip()
			if text:
				cleaned.append(text)
		return self._dedupe_compact_strings(cleaned, max_items=max_items, max_len=160)

	def _fallback_rewrite_auto_summary(self, lessons: List[Dict[str, Any]]) -> List[str]:
		prioritized = sorted(
			self._sort_rewrite_lessons(lessons),
			key=lambda lesson: (
				-len(lesson.get("seen_in", []) or []),
				self._rewrite_lesson_category_rank(str(lesson.get("category", ""))),
				self._normalize_text_for_compare(lesson.get("title", "")),
			),
		)
		summary = [str(lesson.get("rule", "")).strip() for lesson in prioritized if str(lesson.get("rule", "")).strip()]
		return self._dedupe_compact_strings(summary, max_items=6, max_len=160)

	def _strip_markdown_backticks(self, text: str) -> str:
		value = str(text or "").strip()
		while len(value) >= 2 and value.startswith("`") and value.endswith("`"):
			value = value[1:-1].strip()
		return value

	def _parse_markdown_code_list(self, text: str) -> List[str]:
		value = str(text or "").strip()
		if not value:
			return []
		matches = re.findall(r"`([^`]+)`", value)
		if matches:
			return self._dedupe_compact_strings(matches, max_len=120)
		parts = [self._strip_markdown_backticks(item.strip()) for item in value.split(",")]
		return self._dedupe_compact_strings(parts, max_len=120)

	def _extract_markdown_bullets_from_section(self, text: str, heading: str) -> List[str]:
		start, end = self._find_rewrite_section_span(text, heading)
		if start < 0 or end < 0:
			return []
		section_text = text[start:end]
		if "### " in section_text:
			return []
		bullets: List[str] = []
		for line in section_text.splitlines():
			stripped = line.strip()
			if stripped.startswith("- "):
				bullets.append(stripped[2:].strip())
		return bullets

	def _parse_rewrite_auto_lessons_markdown(self, text: str) -> Dict[str, Any]:
		prefix_to_category = {spec["prefix"]: spec["category"] for spec in self._rewrite_lessons_section_specs()}
		summary_points = self._sanitize_rewrite_summary_points(
			self._extract_markdown_bullets_from_section(text, "## Summary")
			+ self._extract_markdown_bullets_from_section(text, "## Quick Heuristics")
		)
		lessons: List[Dict[str, Any]] = []
		entry_matches = list(re.finditer(r"^### (\d+)\.(\d+)\s+(.+)$", text, re.MULTILINE))
		for idx, match in enumerate(entry_matches):
			category = prefix_to_category.get(match.group(1), "")
			if not category:
				continue
			block_end = entry_matches[idx + 1].start() if idx + 1 < len(entry_matches) else len(text)
			block = text[match.end():block_end]
			lesson: Dict[str, Any] = {
				"category": category,
				"title": match.group(3).strip(),
				"rule": "",
				"why": "",
				"symptom": "",
				"seen_in": [],
				"notes": [],
			}
			pending_field = ""
			for raw_line in block.splitlines():
				if raw_line.startswith("## "):
					break
				stripped = raw_line.strip()
				if not stripped:
					pending_field = ""
					continue
				if stripped.startswith("- "):
					body = stripped[2:].strip()
					label, sep, value = body.partition(":")
					if not sep:
						pending_field = ""
						continue
					label_key = self._normalize_text_for_compare(label)
					value = value.strip()
					if label_key == "rule":
						lesson["rule"] = value
						pending_field = "rule"
					elif label_key == "why":
						lesson["why"] = value
						pending_field = "why"
					elif label_key == "typical symptom":
						lesson["symptom"] = self._strip_markdown_backticks(value)
						pending_field = "symptom"
					elif label_key == "seen in":
						lesson["seen_in"] = list(lesson.get("seen_in", [])) + self._parse_markdown_code_list(value)
						pending_field = "seen_in"
					else:
						lesson["notes"] = list(lesson.get("notes", []))
						lesson["notes"].append(body)
						pending_field = "notes"
					continue
				if not (raw_line.startswith("  ") or raw_line.startswith("\t")):
					pending_field = ""
					continue
				if pending_field == "symptom":
					extra = self._strip_markdown_backticks(stripped)
					lesson["symptom"] = (str(lesson.get("symptom", "")).strip() + " " + extra).strip()
				elif pending_field == "seen_in":
					lesson["seen_in"] = list(lesson.get("seen_in", [])) + self._parse_markdown_code_list(stripped)
				elif pending_field in {"rule", "why"}:
					lesson[pending_field] = (str(lesson.get(pending_field, "")).strip() + " " + stripped).strip()
				elif pending_field == "notes" and lesson.get("notes"):
					notes = list(lesson.get("notes", []))
					notes[-1] = (str(notes[-1]).strip() + " " + self._strip_markdown_backticks(stripped)).strip()
					lesson["notes"] = notes
			sanitized = self._sanitize_rewrite_lesson_entry(lesson, "")
			if sanitized:
				lessons.append(sanitized)
		return {
			"summary": summary_points,
			"lessons": self._sort_rewrite_lessons(self._merge_rewrite_lessons_locally(lessons)),
		}

	def _fallback_rewrite_lessons_entries(self, retrospective: Dict[str, Any]) -> List[Dict[str, Any]]:
		design_name = str(retrospective.get("design_name", "")).strip()
		issues = [str(issue).strip() for issue in retrospective.get("hardware_loop", {}).get("issues", []) if str(issue).strip()]
		failures = retrospective.get("software_loop", {}).get("failures", []) or []
		lessons: List[Dict[str, Any]] = []
		if failures:
			note = str(failures[0].get("notes", "")).strip()
			lesson = self._sanitize_rewrite_lesson_entry(
				{
					"category": "software_rewrite_guardrails",
					"title": "Preserve software equivalence before hardware optimization",
					"rule": "Keep software rewrites conservative enough to preserve observable behavior before starting hardware rewrite.",
					"why": "A software-invalid candidate cannot serve as a stable baseline for downstream hardware optimization.",
					"symptom": note,
					"seen_in": [design_name],
				},
				design_name,
			)
			if lesson:
				lessons.append(lesson)
		for issue in issues:
			lower = issue.lower()
			candidate_lesson: Dict[str, Any] | None = None
			if "only allowed in function scope" in lower:
				candidate_lesson = {
					"category": "hardware_rewrite_guardrails",
					"title": "Keep HLS pragmas inside legal function scope",
					"rule": "Emit #pragma HLS directives only in legal Vitis function or loop scope, never at file scope.",
					"why": "Illegal pragma placement invalidates the hardware rewrite and every downstream DSE candidate built from it.",
					"symptom": issue,
					"seen_in": [design_name],
				}
			elif "multiple bus read operation" in lower or "m_axi" in lower:
				candidate_lesson = {
					"category": "memory_and_axi_bottlenecks",
					"title": "Reduce repeated external-memory traffic before adding more parallelism",
					"rule": "Introduce local buffering or reuse when a pipelined loop repeatedly reads the same external-memory interface.",
					"why": "AXI contention usually imposes an II floor that pragma changes alone do not remove.",
					"symptom": issue,
					"seen_in": [design_name],
				}
			elif "timed out" in lower:
				candidate_lesson = {
					"category": "pragma_tuning_and_dse_lessons",
					"title": "Narrow the hardware search space after Vitis timeout",
					"rule": "After Vitis timeout, simplify the hardware strategy before exploring more aggressive pragma combinations.",
					"why": "Timeout is usually evidence that the current search space or rewrite is structurally too aggressive.",
					"symptom": issue,
					"seen_in": [design_name],
				}
			elif "dataflow" in lower:
				candidate_lesson = {
					"category": "hardware_rewrite_guardrails",
					"title": "Stage shared external memory before DATAFLOW partitioning",
					"rule": "Do not place multiple DATAFLOW processes directly on the same external-memory interface; buffer or stream data first.",
					"why": "Shared m_axi traffic inside DATAFLOW often produces invalid or unschedulable hardware structures.",
					"symptom": issue,
					"seen_in": [design_name],
				}
			if not candidate_lesson:
				continue
			sanitized = self._sanitize_rewrite_lesson_entry(candidate_lesson, design_name)
			if sanitized:
				lessons.append(sanitized)
		if not lessons and issues:
			sanitized = self._sanitize_rewrite_lesson_entry(
				{
					"category": "tooling_and_flow_compatibility",
					"title": "Separate tool failures from optimization conclusions",
					"rule": "Verify that the expected HLS artifacts were produced before treating a run outcome as a design lesson.",
					"why": "Missing reports or orchestration failures can look like optimization failures even when no valid HLS result exists.",
					"symptom": issues[0],
					"seen_in": [design_name],
				},
				design_name,
			)
			if sanitized:
				lessons.append(sanitized)
		deduped: List[Dict[str, Any]] = []
		seen_keys = set()
		for lesson in lessons:
			key = (
				self._normalize_text_for_compare(lesson.get("category", "")),
				self._normalize_text_for_compare(lesson.get("title", "")),
				self._normalize_text_for_compare(lesson.get("rule", "")),
			)
			if key in seen_keys:
				continue
			seen_keys.add(key)
			deduped.append(lesson)
		return deduped[:4]

	def _extract_json_object_from_text(self, text: str) -> Dict[str, Any]:
		raw = str(text or "").strip()
		raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
		raw = re.sub(r"\n?```$", "", raw).strip()
		try:
			loaded = json.loads(raw)
			return loaded if isinstance(loaded, dict) else {}
		except Exception:
			match = re.search(r"\{.*\}", raw, re.DOTALL)
			if not match:
				return {}
			try:
				loaded = json.loads(match.group(0))
				return loaded if isinstance(loaded, dict) else {}
			except Exception:
				return {}

	def _generate_rewrite_lessons_entries(
		self,
		retrospective: Dict[str, Any],
		log_file: str,
	) -> List[Dict[str, Any]]:
		section_names = ", ".join(spec["category"] for spec in self._rewrite_lessons_section_specs())
		messages = [
			{
				"role": "system",
				"content": (
					"Summarize reusable rewrite lessons from one HLS optimization run. "
					"Return JSON only with schema {\"lessons\":[...]} and no markdown fences. "
					"Each lesson must contain category, title, rule, why, symptom, and seen_in. "
					f"Choose category only from: {section_names}. "
					"Prefer at most 4 lessons. "
					"Use stable topical lessons, not chronological notes. "
					"Do not include timestamps, run ids, or headings. "
					"If there is no stable lesson, return {\"lessons\":[]}."
				),
			},
			{
				"role": "user",
				"content": "Run retrospective:\n" + json.dumps(retrospective, indent=2),
			},
		]
		try:
			raw = self.client.chat(messages)["content"].strip()
			self._log_block(log_file, "Rewrite lessons summary output", raw, max_lines=18, max_chars=1200)
			payload = self._extract_json_object_from_text(raw)
			lessons_raw = payload.get("lessons", [])
			if isinstance(lessons_raw, list):
				design_name = str(retrospective.get("design_name", "")).strip()
				lessons: List[Dict[str, Any]] = []
				for item in lessons_raw:
					if not isinstance(item, dict):
						continue
					sanitized = self._sanitize_rewrite_lesson_entry(item, design_name)
					if sanitized:
						lessons.append(sanitized)
				if lessons:
					deduped: List[Dict[str, Any]] = []
					seen_keys = set()
					for lesson in lessons:
						key = (
							self._normalize_text_for_compare(lesson.get("category", "")),
							self._normalize_text_for_compare(lesson.get("title", "")),
							self._normalize_text_for_compare(lesson.get("rule", "")),
						)
						if key in seen_keys:
							continue
						seen_keys.add(key)
						deduped.append(lesson)
					return deduped[:4]
		except Exception as exc:
			self._append_text(log_file, f"[WARN] Failed to summarize rewrite lessons with LLM: {exc}\n")
		return self._fallback_rewrite_lessons_entries(retrospective)

	def _render_rewrite_lessons_markdown(self, lessons: List[Dict[str, Any]]) -> str:
		if not lessons:
			return ""
		specs = {spec["category"]: spec for spec in self._rewrite_lessons_section_specs()}
		lines: List[str] = []
		for lesson in lessons:
			spec = specs.get(str(lesson.get("category", "")), {})
			label = spec.get("label", str(lesson.get("category", "")).replace("_", " ").title())
			lines.extend(
				[
					f"### {label}: {lesson.get('title', '')}",
					"",
					f"- Rule: {lesson.get('rule', '')}",
					f"- Why: {lesson.get('why', '')}",
				]
			)
			for note in lesson.get("notes", []) or []:
				lines.append(f"- {note}")
			symptom = str(lesson.get("symptom", "")).strip()
			if symptom:
				lines.extend(["- Typical symptom:", f"  `{symptom}`"])
			seen_in = lesson.get("seen_in", []) or []
			if seen_in:
				lines.extend(["- Seen in:", "  " + ", ".join(f"`{item}`" for item in seen_in if str(item).strip())])
			lines.extend(["", ""])
		return "\n".join(lines).strip()

	def _render_rewrite_retrospective_markdown(self, retrospective: Dict[str, Any]) -> str:
		lines = [
			"# Rewrite Retrospective",
			"",
			f"- Run: `{retrospective.get('run_id', '')}`",
			f"- Design: `{retrospective.get('design_name', '')}`",
			"",
			"## Software Loop",
			f"- Attempts: {retrospective.get('software_loop', {}).get('attempt_count', 0)}",
			f"- Failures: {retrospective.get('software_loop', {}).get('failure_count', 0)}",
			"",
			"## Hardware Loop",
			f"- Hardware rewrite attempts: {retrospective.get('hardware_loop', {}).get('hardware_attempts', 0)}",
			f"- Pragma candidates: {retrospective.get('hardware_loop', {}).get('pragma_candidate_count', 0)}",
			f"- Successful pragma-dse candidates: {retrospective.get('hardware_loop', {}).get('pragma_dse_successful_candidate_count', 0)}",
			"",
			"## Hardware Issues",
		]
		hardware_issues = retrospective.get("hardware_loop", {}).get("issues", [])
		if not hardware_issues:
			lines.append("- No hardware issues recorded.")
		else:
			for issue in hardware_issues[:8]:
				lines.append(f"- {issue}")
		lines.extend([
			"",
			"## Learned Lessons",
		])
		lessons_markdown = str(retrospective.get("lessons_markdown", "")).strip()
		if not lessons_markdown:
			lines.append("- No reusable lessons were extracted from this run.")
		else:
			lines.extend(lessons_markdown.splitlines())
		lines.append("")
		return "\n".join(lines)

	def _compact_rewrite_auto_lessons(
		self,
		existing_summary: List[str],
		existing_lessons: List[Dict[str, Any]],
		new_lessons: List[Dict[str, Any]],
		log_file: str,
	) -> tuple[List[str], List[Dict[str, Any]]]:
		combined = self._sort_rewrite_lessons(
			self._merge_rewrite_lessons_locally(list(existing_lessons) + list(new_lessons))
		)
		if not combined:
			return [], []
		section_names = ", ".join(spec["category"] for spec in self._rewrite_lessons_section_specs())
		messages = [
			{
				"role": "system",
				"content": (
					"Consolidate a self-updating library of HLS rewrite lessons. "
					"Return JSON only with schema {\"summary\":[...],\"lessons\":[...]}. "
					"Each lesson must contain category, title, rule, why, symptom, seen_in, and optional notes. "
					f"Choose category only from: {section_names}. "
					"Merge semantically overlapping lessons, keep only stable reusable rules, and union seen_in evidence. "
					"Do not keep near-duplicate wording when one compact lesson can represent the same guidance. "
					"Preserve genuinely new lessons if they add distinct guidance. "
					"Prefer at most 4 lessons per category and at most 6 summary bullets. "
					"Do not include markdown fences, timestamps, run ids, or chronology."
				),
			},
			{
				"role": "user",
				"content": json.dumps(
					{
						"existing_summary": existing_summary,
						"lessons": combined,
					},
					indent=2,
				),
			},
		]
		try:
			raw = self.client.chat(messages)["content"].strip()
			self._log_block(log_file, "Rewrite auto-lessons compaction output", raw, max_lines=18, max_chars=1200)
			payload = self._extract_json_object_from_text(raw)
			lessons_raw = payload.get("lessons", [])
			compacted_lessons: List[Dict[str, Any]] = []
			if isinstance(lessons_raw, list):
				for item in lessons_raw:
					if not isinstance(item, dict):
						continue
					sanitized = self._sanitize_rewrite_lesson_entry(item, "")
					if sanitized:
						compacted_lessons.append(sanitized)
			compacted_lessons = self._limit_rewrite_lessons_per_category(
				self._merge_rewrite_lessons_locally(compacted_lessons),
				max_per_category=4,
			)
			if compacted_lessons:
				summary_points = self._sanitize_rewrite_summary_points(payload.get("summary", []))
				if not summary_points:
					summary_points = self._fallback_rewrite_auto_summary(compacted_lessons)
				return summary_points, compacted_lessons
		except Exception as exc:
			self._append_text(log_file, f"[WARN] Failed to compact rewrite auto-lessons with LLM: {exc}\n")
		fallback_lessons = self._sort_rewrite_lessons(self._merge_rewrite_lessons_locally(combined))
		return self._fallback_rewrite_auto_summary(fallback_lessons), fallback_lessons

	def _render_rewrite_auto_lessons_markdown(
		self,
		summary_points: List[str],
		lessons: List[Dict[str, Any]],
	) -> str:
		lines = [
			"# Auto-Learned Rewrite Lessons",
			"",
			"This file is self-updated from completed HLSClaw runs.",
			"It keeps reusable rewrite lessons compact by merging redundant content over time.",
			"",
			"## Summary",
			"",
		]
		if summary_points:
			for point in summary_points:
				lines.append(f"- {point}")
		else:
			lines.append("- No stable lessons have been consolidated yet.")
		grouped: Dict[str, List[Dict[str, Any]]] = {}
		for lesson in self._sort_rewrite_lessons(lessons):
			category = str(lesson.get("category", "")).strip()
			grouped.setdefault(category, []).append(lesson)
		for spec in self._rewrite_lessons_section_specs():
			lines.extend(["", spec["heading"], ""])
			for index, lesson in enumerate(grouped.get(spec["category"], []), start=1):
				lines.extend(self._render_rewrite_auto_lesson_entry(spec["prefix"], index, lesson).splitlines())
		return "\n".join(lines).rstrip() + "\n"

	def _ensure_rewrite_auto_lessons_file(self) -> str:
		path = self._rewrite_auto_lessons_md_path()
		if os.path.exists(path) and self._read_text(path).strip():
			return path
		self._write_text(path, self._render_rewrite_auto_lessons_markdown([], []))
		return path

	def _normalize_text_for_compare(self, text: Any) -> str:
		return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

	def _find_rewrite_section_span(self, text: str, heading: str) -> tuple[int, int]:
		start = text.find(heading)
		if start < 0:
			return -1, -1
		search_from = start + len(heading)
		next_match = re.search(r"^## ", text[search_from:], re.MULTILINE)
		end = search_from + next_match.start() if next_match else len(text)
		return start, end

	def _render_rewrite_auto_lesson_entry(self, prefix: str, index: int, lesson: Dict[str, Any]) -> str:
		lines = [
			f"### {prefix}.{index} {lesson.get('title', '')}",
			"",
			f"- Rule: {lesson.get('rule', '')}",
			f"- Why: {lesson.get('why', '')}",
		]
		for note in lesson.get("notes", []) or []:
			lines.append(f"- {note}")
		symptom = str(lesson.get("symptom", "")).strip()
		if symptom:
			lines.extend(["- Typical symptom:", f"  `{symptom}`"])
		seen_in = lesson.get("seen_in", []) or []
		if seen_in:
			lines.extend(["- Seen in:", "  " + ", ".join(f"`{item}`" for item in seen_in if str(item).strip())])
		lines.extend(["", ""])
		return "\n".join(lines)

	def _append_rewrite_auto_lessons_entries(
		self,
		lessons: List[Dict[str, Any]],
		log_file: str,
	) -> str:
		path = self._ensure_rewrite_auto_lessons_file()
		existing = self._parse_rewrite_auto_lessons_markdown(self._read_text(path))
		summary_points, merged_lessons = self._compact_rewrite_auto_lessons(
			list(existing.get("summary", []) or []),
			list(existing.get("lessons", []) or []),
			lessons,
			log_file,
		)
		self._write_text(path, self._render_rewrite_auto_lessons_markdown(summary_points, merged_lessons))
		return path

	def _update_rewrite_skill_from_run(
		self,
		design_name: str,
		run_dir: str,
		final_summary: Dict[str, Any],
		log_file: str,
	) -> Dict[str, str]:
		references_dir = self._rewrite_skill_references_dir()
		os.makedirs(references_dir, exist_ok=True)
		retrospective = self._derive_rewrite_retrospective(design_name, run_dir, final_summary)
		lessons = self._generate_rewrite_lessons_entries(retrospective, log_file)
		lessons_markdown = self._render_rewrite_lessons_markdown(lessons)
		retrospective["lessons"] = lessons
		retrospective["lessons_markdown"] = lessons_markdown

		retrospective_md_path = os.path.join(run_dir, "rewrite_retrospective.md")
		self._write_text(retrospective_md_path, self._render_rewrite_retrospective_markdown(retrospective))
		auto_lessons_md_path = self._append_rewrite_auto_lessons_entries(lessons, log_file)
		self._append_text(
			log_file,
			f"[INFO] Rewrite retrospective saved: {retrospective_md_path}\n"
			f"[INFO] Rewrite skill lessons appended: {auto_lessons_md_path}\n",
		)
		return {
			"rewrite_retrospective_md": retrospective_md_path,
			"rewrite_auto_lessons_md": auto_lessons_md_path,
		}

	def _find_skill_name(self, skills: List[SkillInfo], keyword: str) -> str:
		kw = keyword.lower()
		for skill in skills:
			if kw in skill.name.lower():
				return skill.name
		return ""

	def _check_vitis_available(self) -> bool:
		"""Check if vitis-run is available on PATH."""
		if shutil.which("vitis-run"):
			return True
		logger.warning("vitis-run not found on PATH")
		return False

	def _run_direct_profiling(self, log_file: str) -> Dict[str, Any]:
		script_dir = os.path.join(self.skills_dir, "profiling", "scripts")
		stage_dir = os.path.join(self.scratchpad.run_dir, "profiling")
		os.makedirs(stage_dir, exist_ok=True)
		tcl_path = os.path.join(stage_dir, "gen.tcl")
		report_path = os.path.join(stage_dir, "profiling_report.json")
		code_path = self.scratchpad.optimized_code_file[self.optimized_code_turn]
		clock_period = self._clock_period_from_target_freq()

		self._run_python_script(
			os.path.join(script_dir, "gen_tcl.py"),
			[code_path, self.scratchpad.top_function_name, self.scratchpad.fpga, clock_period, tcl_path],
			log_file,
			"profiling-gen-tcl",
			cwd=stage_dir,
		)
		self._run_python_script(
			os.path.join(script_dir, "run_vitis.py"),
			[tcl_path],
			log_file,
			"profiling-vitis",
			cwd=stage_dir,
			timeout=0,
		)
		csynth_path = self._find_csynth_report(stage_dir)
		if not csynth_path:
			raise ValueError("Profiling failed: csynth.xml was not generated")
		rc, output = self._run_python_script(
			os.path.join(script_dir, "parse_csynth.py"),
			[csynth_path, "-o", report_path],
			log_file,
			"profile-parse",
			cwd=stage_dir,
		)
		if rc != 0 or not os.path.exists(report_path):
			raise ValueError("Profiling failed: parse_csynth did not produce JSON output")
		data = self._load_json_file(report_path)
		out = self._store_json_artifact("profiling_report", data, report_path, log_file)
		out["analysis"] = "Profiling completed from real Vitis HLS csynth results."
		out["command_result"] = output
		return out

	def _run_direct_csim_verification(self, turn: int, log_file: str, variant_kind: str) -> Dict[str, Any]:
		script_path = os.path.join(
			self.skills_dir,
			"csim-verification",
			"scripts",
			"run_csim_equiv.py",
		)
		run_dir = self.scratchpad.run_dir
		version = f"v{turn}"
		verification_dir = os.path.join(run_dir, "csim_verification", version)
		report_path = os.path.join(verification_dir, "csim_equiv_report.json")
		trials = max(1, int(self.config.get("csim_equiv_trials", 8)))
		buf_size = max(1, int(self.config.get("csim_equiv_buf_size", 4096)))
		seed = int(self.config.get("csim_equiv_seed", 1))
		atol = float(self.config.get("csim_equiv_atol", 1e-5))
		rtol = float(self.config.get("csim_equiv_rtol", 1e-5))
		per_run_timeout = max(1, int(self.config.get("csim_equiv_timeout_sec", 300)))
		if os.path.exists(report_path):
			os.remove(report_path)
		rc, command_result = self._run_python_script(
			script_path,
			[
				self.scratchpad.optimized_code_file[turn],
				self.scratchpad.optimized_code_file[0],
				self.scratchpad.top_function_name,
				"--part",
				self.scratchpad.fpga,
				"--clock-period",
				self._clock_period_from_target_freq(),
				"--work-dir",
				verification_dir,
				"--report",
				report_path,
				"--trials",
				str(trials),
				"--seed",
				str(seed),
				"--buf-size",
				str(buf_size),
				"--atol",
				str(atol),
				"--rtol",
				str(rtol),
				"--timeout",
				str(per_run_timeout),
			],
			log_file,
			f"csim-equiv-{version}",
			cwd=run_dir,
			timeout=per_run_timeout * 2 + 120,
		)
		report = self._load_json_file(report_path) if os.path.exists(report_path) else {}
		status = str(report.get("status", "")).strip().upper() or self._extract_verification_status(
			{"command_result": command_result}
		)
		if status not in {"PASS", "FAIL", "ERROR", "TIMEOUT"} or (rc != 0 and status == "PASS"):
			status = "ERROR"
		diagnostics = self._extract_verification_diagnostics(
			{"command_result": command_result, "json_artifacts": [report] if report else []}
		)
		output = {
			"analysis": (
				f"AMD Vitis C-sim equivalence verification completed for {version} "
				f"({variant_kind}). Result: {status}. Only PASS is accepted."
				+ ("\nKey diagnostics:\n" + "\n".join(f"- {item}" for item in diagnostics) if diagnostics else "")
			),
			"command_result": f"[csim_command_rc]={rc}\n{command_result}",
			"status": status,
			"failure_diagnostics": diagnostics,
			"json_artifacts": [
				{
					"skill": "csim-verification",
					"version": version,
					"variant_kind": variant_kind,
					"status": status,
					"success": status == "PASS",
					"reason": str(report.get("reason", "")).strip(),
					"mismatch": report.get("mismatch"),
					"report_path": report_path if os.path.exists(report_path) else "",
					"diagnostics": diagnostics,
				}
			],
		}
		artifact_key = f"{variant_kind}_csim_verification_report"
		if os.path.exists(report_path):
			self.scratchpad.stage_artifacts[artifact_key] = report_path
		return output

	def _run_direct_pragma_tuning(self, log_file: str) -> Dict[str, Any]:
		script_dir = os.path.join(self.skills_dir, "pragma-tuning", "scripts")
		candidates_path = os.path.join(self.scratchpad.run_dir, "pragma_candidates.json")
		plan_path = self.scratchpad.stage_artifacts.get("pragma_tuning_plan", "")
		hardware_turn = self.scratchpad.stage_artifacts.get("hardware_rewrite_turn")
		if hardware_turn is None:
			raise ValueError("Pragma tuning requires a hardware rewrite result; no hardware_rewrite_turn was recorded")
		hardware_turn = int(hardware_turn)
		code_path = self.scratchpad.optimized_code_file[hardware_turn]
		args = [
			"--code",
			code_path,
			"--output",
			candidates_path,
			"--variant-kind",
			"hardware",
			"--source-turn",
			str(hardware_turn),
		]
		if plan_path and os.path.exists(plan_path):
			args.extend(["--plan", plan_path])
		rc, output = self._run_python_script(
			os.path.join(script_dir, "gen_candidates.py"),
			args,
			log_file,
			"pragma-tuning-generate",
		)
		if rc != 0 or not os.path.exists(candidates_path):
			raise ValueError("Pragma tuning failed: pragma_candidates.json was not generated")
		self.scratchpad.stage_artifacts["pragma_candidates"] = candidates_path
		data = self._load_json_file(candidates_path)
		out = self._store_json_artifact("pragma_candidates", data, candidates_path, log_file)
		out["analysis"] = "Pragma tuning completed with parameter-tuning candidate generation only."
		out["command_result"] = output
		return out

	def _run_direct_pragma_dse(self, log_file: str, hardware_attempt: int = 0) -> Dict[str, Any]:
		script_dir = os.path.join(self.skills_dir, "pragma-dse", "scripts")
		candidates_path = self.scratchpad.stage_artifacts.get("pragma_candidates", "")
		output_path = os.path.join(self.scratchpad.run_dir, "pragma_dse_report.json")
		iteration_idx = max(1, int(hardware_attempt or 0))
		work_dir = os.path.join(self.scratchpad.run_dir, "pragma_dse_runs", f"hw_iter_{iteration_idx}")
		code_path = ""
		if candidates_path and os.path.exists(candidates_path):
			candidate_payload = self._load_json_file(candidates_path)
			code_path = candidate_payload.get("source_code_path", "")
			if not code_path or not os.path.exists(code_path):
				raise ValueError("Pragma DSE requires a valid hardware rewrite source_code_path in pragma_candidates.json")
		else:
			code_path = self.scratchpad.stage_artifacts.get("hardware_rewrite_code", "")
			if not code_path or not os.path.exists(code_path):
				raise ValueError(
					"Pragma DSE requires either pragma_candidates.json from pragma-tuning or a valid hardware rewrite source for fallback candidate generation"
				)
		args = [
			"--code",
			code_path,
			"--top-func",
			self.scratchpad.top_function_name,
			"--part",
			self.scratchpad.fpga,
			"--target-freq-mhz",
			self.scratchpad.target_frequency_mhz,
			"--goal",
			self.scratchpad.goal,
			"--work-dir",
			work_dir,
			"--output",
			output_path,
			"--search-strategy",
			self._resolve_pragma_dse_search_strategy(),
			"--candidate-timeout-sec",
			str(self._resolve_non_negative_int_config("pragma_dse_candidate_timeout_sec", 0)),
		]
		if candidates_path and os.path.exists(candidates_path):
			args.extend(["--candidates", candidates_path])
		if self.config.get("pragma_dse_max_candidates") is not None:
			args.extend(["--max-candidates", str(self._resolve_positive_int_config("pragma_dse_max_candidates", 8))])
		search_strategy = self._resolve_pragma_dse_search_strategy()
		if search_strategy == "progressive":
			args.extend(
				[
					"--max-combos",
					str(self._resolve_non_negative_int_config("pragma_dse_max_combos", 4)),
					"--top-per-site",
					str(self._resolve_positive_int_config("pragma_dse_top_per_site", 1)),
					"--beam-width",
					str(self._resolve_positive_int_config("pragma_dse_beam_width", 2)),
					"--beam-temperature",
					str(float(self.config.get("pragma_dse_beam_temperature", 50.0))),
					"--beam-cooling",
					str(float(self.config.get("pragma_dse_beam_cooling", 0.85))),
					"--random-combo-fraction",
					str(float(self.config.get("pragma_dse_random_combo_fraction", 0.3))),
				]
			)
		else:
			args.extend(["--max-combos", "0"])
		pragma_dse_jobs = self._resolve_pragma_dse_jobs()
		args.extend(["--jobs", str(pragma_dse_jobs)])
		self._append_text(
			log_file,
			f"[INFO] [pragma-dse] Hardware iteration {iteration_idx} uses work dir: {work_dir}\n",
		)
		rc, output = self._run_python_script(
			os.path.join(script_dir, "run_pragma_dse.py"),
			args,
			log_file,
			"pragma-dse",
			timeout=0,
		)
		if rc != 0 and not os.path.exists(output_path):
			raise ValueError("Pragma DSE failed: pragma_dse_report.json was not generated")
		self.scratchpad.stage_artifacts["pragma_dse_report"] = output_path
		data = self._load_json_file(output_path)
		out = self._store_json_artifact("pragma_dse_report", data, output_path, log_file)
		out["analysis"] = "Pragma DSE completed with real Vitis HLS QoR evaluation."
		out["command_result"] = output
		return out

	def _clear_hardware_optimization_artifacts(self) -> None:
		for key in (
			"hardware_rewrite_turn",
			"hardware_rewrite_code",
			"pragma_tuning_plan",
			"pragma_candidates",
			"pragma_dse_report",
		):
			self.scratchpad.stage_artifacts.pop(key, None)

	def _run_optional_skill(
		self,
		skills: List[SkillInfo],
		available_skills_xml: str,
		preferred_skill: str,
		user_prompt: str,
		log_file: str,
		stage_label: str,
	) -> Dict[str, Any]:
		selected = preferred_skill or self.select_skills(available_skills_xml, user_prompt, log_file)
		if not selected:
			return {}
		messages = self.build_skill_prompt(skills, selected)
		if user_prompt.strip():
			messages.append({"role": "user", "content": user_prompt.strip()})
		raw = self.client.chat(messages)["content"]
		self._log_block(log_file, f"{stage_label} raw output", raw, max_lines=20, max_chars=1400)
		return self._parse_artifacts(raw, log_file)

	def _extract_metrics_from_json_artifacts(self, artifacts: List[Dict[str, Any]]) -> Dict[str, float]:
		metrics: Dict[str, float] = {}
		keys = ["estimated_latency", "estimated_timing_violation", "resource_pressure", "ii", "latency", "timing_violation"]
		for art in artifacts:
			for k in keys:
				if k in art:
					try:
						metrics[k] = float(art[k])
					except (TypeError, ValueError):
						continue
		return metrics

	def _normalize_pragma_plan_targets(self, raw_targets: Any) -> List[Dict[str, Any]]:
		if not isinstance(raw_targets, list):
			return []
		targets: List[Dict[str, Any]] = []
		for item in raw_targets:
			if not isinstance(item, dict):
				continue
			kind = str(item.get("kind", "")).strip().lower()
			if not kind:
				continue
			target: Dict[str, Any] = {"kind": kind}
			line = item.get("line")
			try:
				if line is not None:
					target["line"] = int(line)
			except (TypeError, ValueError):
				pass
			parameter = str(item.get("parameter") or item.get("parameter_kind") or "").strip().lower()
			if parameter:
				target["parameter"] = parameter
			try_values: List[int] = []
			for value in item.get("try_values", []) or []:
				try:
					int_value = int(value)
				except (TypeError, ValueError):
					continue
				if int_value <= 0 or int_value in try_values:
					continue
				try_values.append(int_value)
			if try_values:
				target["try_values"] = try_values
			priority = str(item.get("priority", "")).strip().lower()
			if priority:
				target["priority"] = priority
			reason = str(item.get("reason", "")).strip()
			if reason:
				target["reason"] = reason
			targets.append(target)
		return targets

	def _normalize_interface_plan_targets(self, raw_targets: Any) -> List[Dict[str, Any]]:
		if not isinstance(raw_targets, list):
			return []
		targets: List[Dict[str, Any]] = []
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
			try_values: List[int] = []
			for value in item.get("try_values", []) or []:
				try:
					int_value = int(value)
				except (TypeError, ValueError):
					continue
				if int_value <= 0 or int_value in try_values:
					continue
				try_values.append(int_value)
			if try_values:
				target["try_values"] = try_values
			priority = str(item.get("priority", "")).strip().lower()
			if priority:
				target["priority"] = priority
			reason = str(item.get("reason", "")).strip()
			if reason:
				target["reason"] = reason
			targets.append(target)
		return targets

	def _normalize_storage_plan_targets(self, raw_targets: Any) -> List[Dict[str, Any]]:
		if not isinstance(raw_targets, list):
			return []
		targets: List[Dict[str, Any]] = []
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
				entry: Dict[str, Any] = {}
				storage_type = str(binding.get("type", "")).strip().lower()
				impl = str(binding.get("impl", "")).strip().lower()
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
			priority = str(item.get("priority", "")).strip().lower()
			if priority:
				target["priority"] = priority
			reason = str(item.get("reason", "")).strip()
			if reason:
				target["reason"] = reason
			targets.append(target)
		return targets

	def _normalize_op_plan_targets(self, raw_targets: Any) -> List[Dict[str, Any]]:
		if not isinstance(raw_targets, list):
			return []
		targets: List[Dict[str, Any]] = []
		for item in raw_targets:
			if not isinstance(item, dict):
				continue
			op = str(item.get("op", "")).strip().lower()
			if not op:
				continue
			target: Dict[str, Any] = {"op": op}
			try_impls: List[str] = []
			for impl in item.get("try_impls", []) or []:
				text = str(impl).strip().lower()
				if text and text not in try_impls:
					try_impls.append(text)
			if try_impls:
				target["try_impls"] = try_impls
			latency_values: List[int] = []
			for value in item.get("latency_values", []) or []:
				try:
					int_value = int(value)
				except (TypeError, ValueError):
					continue
				if int_value < 0 or int_value in latency_values:
					continue
				latency_values.append(int_value)
			if latency_values:
				target["latency_values"] = latency_values
			priority = str(item.get("priority", "")).strip().lower()
			if priority:
				target["priority"] = priority
			reason = str(item.get("reason", "")).strip()
			if reason:
				target["reason"] = reason
			targets.append(target)
		return targets

	def _normalize_dataflow_plan_targets(self, raw_targets: Any) -> List[Dict[str, Any]]:
		if not isinstance(raw_targets, list):
			return []
		targets: List[Dict[str, Any]] = []
		for item in raw_targets:
			if not isinstance(item, dict):
				continue
			option = str(item.get("option", "")).strip().lower().lstrip("-")
			if not option:
				continue
			target: Dict[str, Any] = {"option": option}
			if option == "default_channel":
				try_values = []
				for value in item.get("try_values", []) or []:
					text = str(value).strip().lower()
					if text and text not in try_values:
						try_values.append(text)
				if try_values:
					target["try_values"] = try_values
			else:
				try_values_int: List[int] = []
				for value in item.get("try_values", []) or []:
					try:
						int_value = int(value)
					except (TypeError, ValueError):
						continue
					if int_value <= 0 or int_value in try_values_int:
						continue
					try_values_int.append(int_value)
				if try_values_int:
					target["try_values"] = try_values_int
			priority = str(item.get("priority", "")).strip().lower()
			if priority:
				target["priority"] = priority
			reason = str(item.get("reason", "")).strip()
			if reason:
				target["reason"] = reason
			targets.append(target)
		return targets

	def _extract_tuning_plan_from_json_artifacts(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
		for art in artifacts:
			if not isinstance(art, dict):
				continue
			if not any(
				key in art
				for key in ("tuning_targets", "pragma_targets", "interface_targets", "storage_targets", "op_targets", "dataflow_targets")
			):
				continue
			pragma_targets = self._normalize_pragma_plan_targets(art.get("pragma_targets", art.get("tuning_targets", [])))
			interface_targets = self._normalize_interface_plan_targets(art.get("interface_targets", []))
			storage_targets = self._normalize_storage_plan_targets(art.get("storage_targets", []))
			op_targets = self._normalize_op_plan_targets(art.get("op_targets", []))
			dataflow_targets = self._normalize_dataflow_plan_targets(art.get("dataflow_targets", []))
			plan: Dict[str, Any] = {
				"pragma_targets": pragma_targets,
				"tuning_targets": list(pragma_targets),
				"interface_targets": interface_targets,
				"storage_targets": storage_targets,
				"op_targets": op_targets,
				"dataflow_targets": dataflow_targets,
			}
			notes = str(art.get("notes", "")).strip()
			if notes:
				plan["notes"] = notes
			return plan
		return {}

	def _persist_pragma_tuning_plan(self, hardware_turn: int, hardware_output: Dict[str, Any], log_file: str) -> str:
		artifacts = list(hardware_output.get("json_artifacts", []) or [])
		plan = self._extract_tuning_plan_from_json_artifacts(artifacts)
		if not plan:
			self.scratchpad.stage_artifacts.pop("pragma_tuning_plan", None)
			return ""
		hardware_code = self.scratchpad.optimized_code.get(hardware_turn, "")
		filtered_pragma_targets, pragma_warnings = self._filter_hardware_tuning_targets(
			hardware_code,
			list(plan.get("pragma_targets", plan.get("tuning_targets", [])) or []),
			log_file,
		)
		filtered_interface_targets, interface_warnings = self._filter_interface_tuning_targets(
			hardware_code,
			list(plan.get("interface_targets", []) or []),
			log_file,
		)
		filtered_storage_targets, storage_warnings = self._filter_storage_tuning_targets(
			hardware_code,
			list(plan.get("storage_targets", []) or []),
			log_file,
		)
		filtered_op_targets, op_warnings = self._filter_op_tuning_targets(
			hardware_code,
			list(plan.get("op_targets", []) or []),
			log_file,
		)
		filtered_dataflow_targets, dataflow_warnings = self._filter_dataflow_tuning_targets(
			hardware_code,
			list(plan.get("dataflow_targets", []) or []),
			log_file,
		)
		validation_warnings = pragma_warnings + interface_warnings + storage_warnings + op_warnings + dataflow_warnings
		plan_payload = {
			"generator": "hardware-rewrite",
			"source_code_path": self.scratchpad.optimized_code_file.get(hardware_turn, ""),
			"source_turn": hardware_turn,
			"based_on_hardware_rewrite": True,
			"pragma_targets": filtered_pragma_targets,
			"tuning_targets": list(filtered_pragma_targets),
			"interface_targets": filtered_interface_targets,
			"storage_targets": filtered_storage_targets,
			"op_targets": filtered_op_targets,
			"dataflow_targets": filtered_dataflow_targets,
		}
		if plan.get("notes"):
			plan_payload["notes"] = plan["notes"]
		if validation_warnings:
			plan_payload["validation_warnings"] = validation_warnings
		plan_path = os.path.join(self.scratchpad.run_dir, "pragma_tuning_plan.json")
		self._write_text(plan_path, json.dumps(plan_payload, indent=2) + "\n")
		self.scratchpad.stage_artifacts["pragma_tuning_plan"] = plan_path
		self._append_text(
			log_file,
			f"[INFO] Hardware rewrite tuning plan saved: {plan_path} "
			f"(pragma={len(plan_payload['pragma_targets'])}, interface={len(plan_payload['interface_targets'])}, "
			f"storage={len(plan_payload['storage_targets'])}, op={len(plan_payload['op_targets'])}, "
			f"dataflow={len(plan_payload['dataflow_targets'])})\n",
		)
		return plan_path

	def _extract_metrics_from_output_text(self, text: str) -> Dict[str, float]:
		import re

		metrics: Dict[str, float] = {}
		if not text:
			return metrics
		lat = re.search(r"Latency:\s*([0-9]+)", text, re.IGNORECASE)
		ii = re.search(r"\bII\b[^0-9]*([0-9]+)", text, re.IGNORECASE)
		timing_bad = re.search(r"Timing violation", text, re.IGNORECASE)
		if lat:
			metrics["latency"] = float(lat.group(1))
		if ii:
			metrics["ii"] = float(ii.group(1))
		if timing_bad:
			metrics["timing_violation"] = 1.0
		return metrics

	def _is_plain_cpp_design(self, code: str) -> bool:
		hls_only_patterns = (
			r"#\s*pragma\s+HLS\b",
			r"\bhls::stream\b",
			r"\bap_(?:u?int|fixed|ufixed)\b",
			r"#\s*include\s*<\s*ap_(?:int|fixed)\.h\s*>",
			r"#\s*include\s*<\s*hls_stream\.h\s*>",
			r"\bm_axi\b",
			r"\bs_axilite\b",
			r"\baxis\b",
			r"\bap_ctrl_(?:none|hs|chain)\b",
		)
		return not any(re.search(pattern, code) for pattern in hls_only_patterns)

	def _select_best_candidate(self, variant_kind: str | None = None) -> Candidate | None:
		candidates = self.candidates
		if variant_kind is not None:
			candidates = [cand for cand in candidates if cand.variant_kind == variant_kind]
		if not candidates:
			return None
		return max(candidates, key=lambda cand: cand.score)

	def _run_hardware_rewrite(
		self,
		log_file: str,
		profiling_analysis: str,
		rag_analysis: str,
		rewrite_analysis: str,
		base_turn: int,
		feedback: str = "",
		attempt_idx: int = 1,
	) -> Dict[str, Any]:
		self._clear_hardware_optimization_artifacts()
		base_code = self.scratchpad.optimized_code[base_turn]
		_log_phase(f"Phase: Hardware Rewrite Attempt {attempt_idx}/{self._max_hardware_opt_rounds()}")
		raw = self._chat_with_history([
			{
				"role": "system",
					"content": (
					"Convert the selected software-validated design into exactly one hardware-oriented HLS variant. "
					"Preserve the top-function name, return type, parameter order, and parameter base types so the result remains interface-compatible with Original C. "
					"You may use HLS-specific constructs such as interface pragmas, array partition/reshape pragmas, DATAFLOW, hls::stream, "
					"ap_int/ap_uint, and interface annotations when justified. Emit the baseline HLS pragma combination directly in this "
					"hardware rewrite, including PIPELINE, UNROLL, DATAFLOW, STREAM, and ARRAY_PARTITION/RESHAPE when justified. "
					"Pick one primary optimization hypothesis for this round based on the current bottleneck class instead of applying every advanced pragma at once. "
					"If the profile is dominated by AXI or bus-read/write contention, prioritize local buffering, memory reuse, and array partitioning before adding more task-level parallelism. "
					"If the profile is dominated by II/resource limitation, prioritize matching pipeline structure, local memory bandwidth, and tunable partition/unroll factors to the true access pattern. "
					"If the profile is dominated by timing-critical arithmetic or reductions, prioritize shortening the critical path and simplifying the schedule before adding more parallelism. "
						"When the code is still array/loop dominated and has a clear producer/consumer structure, you may choose a structural dataflow circuit optimization hypothesis instead of another pragma-only round. "
						"In that case, change stage boundaries, local staging, banked stream structure, loop tiling, and aligned local array partitioning together as one coherent rewrite. "
						"Do not introduce banked streams unless you can also explain the matching tile factors and local array partitioning that support them. "
						"Use DATAFLOW only when you can clearly decompose the kernel into legal producer/consumer stages with isolated external-memory access; otherwise prefer a single pipelined structure with local buffers. "
						"The later pragma-tuning/pragma-dse stages will primarily adjust existing pragmas, and can also explore a small number of explicit interface, local-storage, operator-binding, and dataflow Tcl knobs when those structures exist in the emitted code. "
						"If an inner loop is unrolled and needs multiple reads from the same external m_axi port, do not treat ARRAY_PARTITION on the top-level m_axi array as the knob. "
						"Instead, rewrite the kernel to stage a local on-chip buffer/window, then place ARRAY_PARTITION/RESHAPE on that local buffer with a factor aligned to the UNROLL factor or concurrent-read count. "
						"Also return one tuning-plan JSON artifact of the form "
						"{\"pragma_targets\":[{\"line\": int, \"kind\": str, \"parameter\": \"ii|factor|depth\", \"try_values\": [int, ...], \"priority\": \"high|medium|low\", \"reason\": str}], "
						"\"interface_targets\":[{\"port\": str, \"option\": str, \"try_values\": [int, ...], \"priority\": \"high|medium|low\", \"reason\": str}], "
						"\"storage_targets\":[{\"variable\": str, \"bindings\": [{\"type\": str, \"impl\": str, \"latency\": int}], \"priority\": \"high|medium|low\", \"reason\": str}], "
						"\"op_targets\":[{\"op\": str, \"try_impls\": [str, ...], \"latency_values\": [int, ...], \"priority\": \"high|medium|low\", \"reason\": str}], "
						"\"dataflow_targets\":[{\"option\": str, \"try_values\": [int_or_str, ...], \"priority\": \"high|medium|low\", \"reason\": str}], "
						"\"notes\": str}. "
						"List only the 2-4 most important tuning targets across all families for later exploration. "
						"Only list ARRAY_PARTITION/RESHAPE tuning targets for pragma sites that actually exist in the emitted code, and never point those targets at top-level m_axi interface arrays. "
						"Only list interface targets when explicit m_axi interfaces exist in the emitted code, only list storage targets for real local arrays, and only list dataflow targets when the emitted code uses DATAFLOW. "
						"If you use DATAFLOW, keep the design structurally legal for Vitis HLS: do not let multiple DATAFLOW processes directly "
						"read the same external m_axi port/bundle. Stage shared external data locally or remove DATAFLOW when necessary. "
						"Keep producer/consumer token counts and bank shapes consistent across any structural dataflow rewrite. "
					"Keep pragma placement structurally legal for Vitis HLS. In particular, '#pragma HLS inline' must appear inside the function body it applies to, not immediately before a function definition at file scope. "
					"If the previous hardware round ended baseline-tied after pragma-dse, treat that as evidence that the current code shape exposes weak tuning knobs; change the memory structure, stage boundaries, or loop structure instead of only adding more pragmas of the same kind. "
					"If the prior rounds already used DATAFLOW but remained baseline-tied, prefer changing stage topology, bank factors, broadcast/transpose placement, or reduction structure before adding more DATAFLOW-related pragmas. "
					"If the latest hardware-loop feedback mentions Vitis timeout, treat that as a sign that the prior optimization strategy was too aggressive. "
					"In the next rewrite, respond conservatively by reducing UNROLL factors, relaxing PIPELINE II targets, increasing STREAM/FIFO depth when channels are used, "
					"and simplifying DATAFLOW/INLINE structure instead of increasing parallelism further. "
					"Return exactly one <optimized_code></optimized_code> block, one <analysis></analysis> block, one metrics "
					"<json>{\"estimated_latency\": number, \"estimated_timing_violation\": 0_or_1, \"resource_pressure\": number}</json>, "
					"and one tuning-plan <json>{...}</json>. "
					"Do not emit multiple variants."
				),
			},
			{
				"role": "user",
				"content": (
					profiling_analysis
					+ rag_analysis
					+ rewrite_analysis
					+ ("\n Hardware Rewrite Feedback \n" + feedback if feedback else "")
					+ "\n Selected Software Rewrite Result\n"
					+ base_code
				),
			},
		])["content"].strip()
		self._log_block(log_file, "Hardware rewrite output", raw, max_lines=24, max_chars=1600)
		hardware_output = self._parse_artifacts(raw, log_file, allow_optimized_code=True)
		hardware_turns = hardware_output.get("optimized_turns", [])
		if not hardware_turns:
			self._append_text(log_file, "[WARN] Hardware rewrite produced no optimized code. Continuing with software candidate.\n")
			self.optimized_code_turn = base_turn
			return {
				"hardware_turn": None,
				"feedback": "Hardware rewrite produced no optimized code. Return exactly one hardware-oriented variant.",
				"metrics": {},
			}
		hardware_turn = hardware_turns[-1]
		validation_issues = self._validate_hardware_candidate(hardware_turn, log_file)
		if validation_issues:
			self.optimized_code_turn = base_turn
			return {
				"hardware_turn": None,
				"feedback": "\n".join(validation_issues),
				"metrics": {},
				"validation_issues": validation_issues,
			}
		verification_output = self._run_direct_csim_verification(hardware_turn, log_file, "hardware")
		verification_status = self._extract_verification_status(verification_output)
		if verification_status != "PASS":
			self.optimized_code_turn = base_turn
			verification_summary = self._summarize_verification_failure_brief(verification_output)
			return {
				"hardware_turn": None,
				"feedback": (
					"Hardware rewrite did not pass AMD Vitis C-sim equivalence verification against Original C. "
					"Revise the hardware design before pragma tuning/DSE. "
					+ verification_summary
				),
				"metrics": {},
				"validation_issues": list(verification_output.get("failure_diagnostics", []) or []),
			}
		metrics = self._extract_metrics_from_output_text(self._generate_llm_prompt(hardware_output))
		if hardware_output.get("json_artifacts"):
			metrics.update(self._extract_metrics_from_json_artifacts(hardware_output["json_artifacts"]))
		plan_path = self._persist_pragma_tuning_plan(hardware_turn, hardware_output, log_file)
		self._register_candidate(
			turn=hardware_turn,
			parent_turn=base_turn,
			stage="hardware_rewrite",
			variant_kind="hardware",
			verification_pass=True,
			metrics=metrics,
			notes="Hardware-oriented rewrite for HLS-only optimization",
		)
		self.scratchpad.stage_artifacts["hardware_rewrite_turn"] = hardware_turn
		self.scratchpad.stage_artifacts["hardware_rewrite_code"] = self.scratchpad.optimized_code_file[hardware_turn]
		self._append_text(log_file, f"[INFO] Hardware candidate selected for downstream HLS stages: v{hardware_turn}\n")
		_log_info(f"Hardware candidate: {_c(f'v{hardware_turn}', '1;37')}")
		return {
			"hardware_turn": hardware_turn,
			"feedback": "",
			"metrics": metrics,
			"pragma_tuning_plan": plan_path,
		}

	def _run_software_rewrite_loop(
		self,
		log_file: str,
		profiling_analysis: str,
		rag_analysis: str,
		rewrite_analysis: str,
		verification_skill_name: str,
	) -> int:
		max_attempts = max(1, int(self.config.get("max_opt_rounds", 5)))
		baseline_turn = self.optimized_code_turn
		if not any(
			candidate.turn == baseline_turn and candidate.variant_kind == "software" and candidate.stage == "seed"
			for candidate in self.candidates
		):
			self._register_candidate(
				turn=baseline_turn,
				parent_turn=baseline_turn,
				stage="seed",
				variant_kind="software",
				verification_pass=True,
				metrics={},
				notes="Baseline software design (assumed correct)",
			)

		parent_turn = baseline_turn
		last_feedback = ""
		last_failed_turn: int | None = None
		for attempt_idx in range(1, max_attempts + 1):
			_log_phase(f"Software Rewrite Attempt {attempt_idx}/{max_attempts}")
			self._append_text(log_file, f"[INFO] === Software Rewrite Attempt {attempt_idx}/{max_attempts} ===\n")
			self.optimized_code_turn = parent_turn
			current_code = self.scratchpad.optimized_code[parent_turn]
			raw = self._chat_with_history([
				{
					"role": "system",
					"content": (
						"Based on the analysis and references, generate exactly one optimized software C/C++ variant "
						"for AMD Vitis C-sim equivalence checking. The candidate must remain plain C/C++ and must not use any HLS-only constructs "
						"such as HLS pragmas, hls::stream, ap_int/ap_uint/ap_fixed, AXI/interface annotations, or "
						"vendor-specific HLS headers/types. Return exactly one <optimized_code></optimized_code>, one "
						"<analysis></analysis>, and optionally one <json>{\"estimated_latency\": number, "
						"\"estimated_timing_violation\": 0_or_1, \"resource_pressure\": number}</json>. "
						"If validation feedback is provided, revise the design to fix that failure."
					),
				},
				{
					"role": "user",
					"content": (
						profiling_analysis
						+ rag_analysis
						+ rewrite_analysis
						+ last_feedback
						+ "\n Current Design \n"
						+ current_code
					),
				},
			])["content"].strip()
			self._log_block(log_file, "Software rewrite candidate output", raw, max_lines=24, max_chars=1600)
			rewrite_output = self._parse_artifacts(raw, log_file)
			candidate_turns = rewrite_output.get("optimized_turns", [])
			if not candidate_turns:
				self._append_text(log_file, "[WARN] No optimized software candidate generated in this attempt.\n")
				_log_warn("No software candidate generated in this attempt.")
				break
			if len(candidate_turns) > 1:
				self._append_text(
					log_file,
					f"[WARN] Software rewrite emitted {len(candidate_turns)} candidates; using the last one only.\n",
				)
				_log_warn("Software rewrite emitted multiple candidates; using the last one only.")

			candidate_turn = candidate_turns[-1]
			self.optimized_code_turn = candidate_turn
			last_failed_turn = candidate_turn
			verification_pass = False
			verification_status = "UNKNOWN"
			verification_output: Dict[str, Any] = {}
			candidate_code = self.scratchpad.optimized_code[candidate_turn]
			if not self._is_plain_cpp_design(candidate_code):
				verification_notes = "Rejected: software rewrite contains HLS-only constructs"
				last_feedback = (
					"\n Validation Feedback \n"
						"Candidate was rejected before Vitis C-sim equivalence checking because the software stage introduced HLS-only constructs. "
					"Keep the next rewrite strictly plain C/C++."
				)
				_log_warn(f"Rejecting software candidate v{candidate_turn}: HLS-only constructs are not allowed in this stage")
			elif verification_skill_name:
				verification_notes = "ERROR"
				_log_skill(f"Vitis C-sim verification for software candidate v{candidate_turn}")
				verification_output = self._run_direct_csim_verification(candidate_turn, log_file, "software")
				verification_status = self._extract_verification_status(verification_output)
				verification_pass = verification_status == "PASS"
				verification_notes = (
					"PASS"
					if verification_pass
					else self._summarize_verification_failure_brief(verification_output)
				)
				_log_info(
					f"C-sim v{candidate_turn}: "
					f"{_c(verification_notes, '1;32' if verification_pass else '1;31')}"
				)
				if verification_pass:
					last_feedback = ""
				else:
					feedback_intro = "Previous software rewrite did not pass AMD Vitis C-sim equivalence validation. Fix the specific mismatch or C-sim error below before proposing the next rewrite.\n"
					last_feedback = (
						"\n Validation Feedback \n"
						+ feedback_intro
						+ self._generate_llm_prompt(verification_output)
					)
			else:
				verification_notes = "C-sim verification skill unavailable"
				last_feedback = "\n Validation Feedback \nC-sim verification is required but unavailable.\n"

			metrics = self._extract_metrics_from_output_text(
				self._generate_llm_prompt(verification_output if verification_skill_name else {})
			)
			if verification_output.get("json_artifacts"):
				metrics.update(self._extract_metrics_from_json_artifacts(verification_output["json_artifacts"]))
			self._register_candidate(
				turn=candidate_turn,
				parent_turn=parent_turn,
				stage=f"attempt_{attempt_idx}",
				variant_kind="software",
				verification_pass=verification_pass,
				metrics=metrics,
				notes=verification_notes,
			)

			if verification_pass:
				self._append_text(
					log_file,
					f"[INFO] Selected software candidate: v{candidate_turn} csim_equivalence=PASS\n",
				)
				_log_info(
					f"Selected software candidate: {_c(f'v{candidate_turn}', '1;37')} "
					f"csim={verification_status.lower()}"
				)
				return candidate_turn

			parent_turn = candidate_turn

		fallback_turn = baseline_turn
		if last_failed_turn is not None:
			self._append_text(
				log_file,
				f"[WARN] No C-sim-equivalent software rewrite found after {max_attempts} attempts. "
				f"Falling back to baseline v{baseline_turn}; latest failed candidate was v{last_failed_turn}.\n",
			)
		else:
			self._append_text(
				log_file,
				f"[WARN] Software rewrite produced no usable candidate. Falling back to baseline v{baseline_turn}.\n",
			)
		_log_warn(f"Falling back to baseline software candidate v{baseline_turn}")
		return fallback_turn

	def _score_candidate(self, metrics: Dict[str, float], verification_pass: bool, variant_kind: str = "software") -> float:
		# TODO: Replace this simple scalar score with a calibrated QoR model that
		# balances latency, II, timing slack, and resource utilization explicitly.
		score = 0.0
		if variant_kind == "software":
			score += 100.0 if verification_pass else -80.0
		score -= metrics.get("timing_violation", 0.0) * 30.0
		score -= metrics.get("estimated_timing_violation", 0.0) * 25.0
		score -= metrics.get("ii", 0.0) * 5.0
		score -= metrics.get("latency", metrics.get("estimated_latency", 0.0)) * 0.001
		score -= metrics.get("resource_pressure", 0.0) * 2.0
		return score

	def _register_candidate(
		self,
		turn: int,
		parent_turn: int,
		stage: str,
		variant_kind: str,
		verification_pass: bool,
		metrics: Dict[str, float],
		notes: str,
	) -> None:
		score = self._score_candidate(metrics, verification_pass, variant_kind=variant_kind)
		self.candidates.append(
			Candidate(
				turn=turn,
				parent_turn=parent_turn,
				stage=stage,
				variant_kind=variant_kind,
				score=score,
				verification_pass=verification_pass,
				metrics=metrics,
				notes=notes,
			)
		)

	def _extract_verification_diagnostics(self, output: Dict[str, Any], limit: int = 8) -> List[str]:
		text = str(output.get("command_result", "") or "")
		diagnostics: List[str] = []
		seen: set[str] = set()

		def add(item: str) -> None:
			message = item.strip()
			if not message or message in seen:
				return
			seen.add(message)
			diagnostics.append(message)

		for artifact in list(output.get("json_artifacts", []) or []):
			if not isinstance(artifact, dict):
				continue
			reason = str(artifact.get("reason", "")).strip()
			if reason:
				add(reason)
			mismatch = artifact.get("mismatch")
			if mismatch:
				add(f"Mismatch: {json.dumps(mismatch, separators=(',', ':'))}")

		reason_match = re.search(r"\[csim_reason\]=(.*)", text)
		if reason_match:
			add(reason_match.group(1).strip())
		mismatch_match = re.search(r"\[csim_mismatch\]=(.*)", text)
		if mismatch_match:
			add(f"Mismatch: {mismatch_match.group(1).strip()}")

		markers = (
			"output mismatch",
			"trace length mismatch",
			"trace schema mismatch",
			"c-sim failed",
			"csim failed",
			"timed out",
			"fatal error:",
			"error:",
		)
		for raw_line in text.splitlines():
			line = raw_line.strip()
			if not line:
				continue
			lower = line.lower()
			if lower.startswith("[command]:") or lower.startswith("[output]:"):
				continue
			if any(marker in lower for marker in markers):
				add(line)
				if len(diagnostics) >= limit:
					break

		return diagnostics[:limit]

	def _extract_verification_status(self, output: Dict[str, Any]) -> str:
		for artifact in list(output.get("json_artifacts", []) or []):
			status = str(artifact.get("status", "")).strip().upper()
			if status in {"PASS", "FAIL", "ERROR", "TIMEOUT"}:
				return status

		text = str(output.get("command_result", "") or "")
		match = re.search(r"\[csim_status\]=([A-Z_]+)", text)
		if match:
			status = match.group(1).strip().upper()
			if status in {"PASS", "FAIL", "ERROR", "TIMEOUT"}:
				return status
		return "UNKNOWN"

	def _summarize_verification_failure_brief(self, output: Dict[str, Any], limit: int = 2) -> str:
		status = self._extract_verification_status(output)
		diagnostics = list(output.get("failure_diagnostics", []) or [])
		if not diagnostics:
			diagnostics = self._extract_verification_diagnostics(output, limit=max(limit, 4))
		if not diagnostics:
			return status if status in {"FAIL", "ERROR", "TIMEOUT"} else "ERROR"
		summary = "; ".join(diagnostics[:limit])
		if len(summary) > 220:
			summary = summary[:217] + "..."
		prefix = status if status in {"FAIL", "ERROR", "TIMEOUT"} else "ERROR"
		return f"{prefix}: {summary}"

	def _build_info_prompt_from_scratchpad(self) -> str:
		current_code = ""
		if self.scratchpad.optimized_code:
			current_code = self.scratchpad.optimized_code.get(
				self.optimized_code_turn,
				self.scratchpad.optimized_code.get(max(self.scratchpad.optimized_code.keys()), ""),
			)

		current_code_path = self.scratchpad.optimized_code_file.get(self.optimized_code_turn, "")
		original_code_path = self.scratchpad.optimized_code_file.get(0, "")

		return (
			"<ScratchpadInfo>\n"
			f"  <Goal>{html.escape(self.scratchpad.goal)}</Goal>\n"
			f"  <FPGA_Board>{html.escape(self.scratchpad.fpga)}</FPGA_Board>\n"
			f"  <Target_Frequency_MHz>{html.escape(self.scratchpad.target_frequency_mhz)}</Target_Frequency_MHz>\n"
			f"  <Top_Function_Name>{html.escape(self.scratchpad.top_function_name)}</Top_Function_Name>\n"
			f"  <Run_Directory>{html.escape(self.scratchpad.run_dir)}</Run_Directory>\n"
			f"  <Current_Code_Path>{html.escape(current_code_path)}</Current_Code_Path>\n"
			f"  <Original_Code_Path>{html.escape(original_code_path)}</Original_Code_Path>\n"
			f"  <Current_Code>\n{current_code}\n  </Current_Code>\n"
			"</ScratchpadInfo>"
		)

		
