import os
import re

_PRAGMA_DSE_JOBS = int(os.getenv("PRAGMA_DSE_JOBS", "4"))
_PRAGMA_DSE_MAX_CANDIDATES = int(os.getenv("PRAGMA_DSE_MAX_CANDIDATES", "8"))
_PRAGMA_DSE_SEARCH_STRATEGY = os.getenv("PRAGMA_DSE_SEARCH_STRATEGY", "progressive")
_PRAGMA_DSE_MAX_COMBOS = int(os.getenv("PRAGMA_DSE_MAX_COMBOS", "4"))
_PRAGMA_DSE_TOP_PER_SITE = int(os.getenv("PRAGMA_DSE_TOP_PER_SITE", "1"))
_PRAGMA_DSE_BEAM_WIDTH = int(os.getenv("PRAGMA_DSE_BEAM_WIDTH", "2"))
_PRAGMA_DSE_BEAM_TEMPERATURE = float(os.getenv("PRAGMA_DSE_BEAM_TEMPERATURE", "50.0"))
_PRAGMA_DSE_BEAM_COOLING = float(os.getenv("PRAGMA_DSE_BEAM_COOLING", "0.85"))
_PRAGMA_DSE_RANDOM_COMBO_FRACTION = float(os.getenv("PRAGMA_DSE_RANDOM_COMBO_FRACTION", "0.3"))
_PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC = int(os.getenv("PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC", "0"))
_MAX_HARDWARE_OPT_ROUNDS = int(os.getenv("MAX_HARDWARE_OPT_ROUNDS", "10"))
_CSIM_VERIFICATION_CONFIG = {
    "csim_equiv_trials": int(os.getenv("CSIM_EQUIV_TRIALS", "8")),
    "csim_equiv_buf_size": int(os.getenv("CSIM_EQUIV_BUF_SIZE", "4096")),
    "csim_equiv_seed": int(os.getenv("CSIM_EQUIV_SEED", "1")),
    "csim_equiv_atol": float(os.getenv("CSIM_EQUIV_ATOL", "1e-5")),
    "csim_equiv_rtol": float(os.getenv("CSIM_EQUIV_RTOL", "1e-5")),
    "csim_equiv_timeout_sec": int(os.getenv("CSIM_EQUIV_TIMEOUT_SEC", "300")),
}

SYSTEM_PROMPT = (
    "You are an expert HLS assistant."
    " Follow a staged HLS optimization workflow driven by the available SKILLS."
    " Provide a detailed, phase-by-phase plan and outputs, but do not reveal hidden chain-of-thought."
    "\n\n"
    "Phase 0 — Intake & Constraints: restate goals, target device, QoR constraints (freq/latency/resources),"
    " and I/O requirements; identify missing info."
    "\n"
    "Phase 1 — Code Understanding: summarize kernels, dataflow, memory accesses, loop structure,"
    " and potential parallelism."
    "\n"
    "Phase 2 — Profiling (profiling): run/parse Vitis profiling to locate hotspots, II/latency bottlenecks,"
    " and constraint conflicts."
    "\n"
    "Phase 3 — Software Rewrite (rewrite): produce one plain C/C++ optimization candidate per attempt that remains compatible"
    " with AMD Vitis HLS C simulation; keep this stage free of HLS-only constructs."
    "\n"
    "Phase 4 — Correctness Validation Loop (csim-verification): simulate Original C and rewritten C independently"
    " with identical deterministic inputs in AMD Vitis HLS, then compare return values and array side effects."
    " Only explicit PASS may advance the pipeline; FAIL, ERROR, or TIMEOUT must return to software rewrite."
    "\n"
    "Phase 5 — Strategy Retrieval (kg-rag): query knowledge graph after the software/C-sim loop using profiling summaries"
    " and the validated software design to obtain hardware-oriented optimization strategies with rationale/risks."
    "\n"
    "Phase 6 — Hardware Rewrite (rewrite): transform the validated software rewrite result into an HLS-oriented design;"
    " HLS-only constructs are allowed here. Run the same Original-C-versus-rewrite Vitis C-sim check before pragma tuning;"
    " this stage should emit the baseline HLS pragma combination directly and identify the most important pragma sites for later parameter tuning."
    " If hardware validation, pragma-tuning, or pragma-dse fails, return to kg-rag and iterate the hardware loop for a bounded number of rounds."
    "\n"
    "Phase 7 — Pragma Tuning (pragma-tuning): generate parameter-tuning candidates by adjusting existing pragmas in the hardware-oriented code variant."
    "\n"
    "Phase 8 — Pragma DSE (pragma-dse): evaluate pragma-tuning candidates under constraints and select best QoR with Vitis HLS."
    "\n"
    "Phase 9 — Finalization: deliver the best code+pragmas, QoR summary, and remaining risks/next steps."
)

def _normalize_api_base(provider: str, api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if not base:
        return base
    if provider in ("anthropic", "openai", "deepseek", "custom") and not base.endswith("/v1"):
        return base + "/v1"
    return base


def _resolve_custom_api_base() -> str:
    direct = os.getenv("CUSTOM_API_BASE", "").strip()
    if direct:
        return _normalize_api_base("custom", direct)
    return _normalize_api_base("custom", "")


def _resolve_custom_api_key() -> str:
    direct = os.getenv("CUSTOM_API_KEY", "").strip()
    if direct:
        return direct
    return ""


def _resolve_custom_model() -> str:
    direct = os.getenv("CUSTOM_MODEL", "").strip()
    if direct:
        return direct
    return "kimi-k2p5"


def _resolve_gemini_api_key() -> str:
    direct = os.getenv("GEMINI_API_KEY", "").strip()
    if direct:
        return direct
    return os.getenv("GOOGLE_API_KEY", "").strip()


OPENROUTER_CONFIG: dict[str, any] = {
    "provider": "openrouter",
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "api_base": os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
    "model": os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free"),
    "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("OPENROUTER_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("OPENROUTER_TIMEOUT", "300")),
    "max_retries": int(os.getenv("OPENROUTER_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("OPENROUTER_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("OPENROUTER_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "6")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "3.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}

OPENAI_CONFIG: dict[str, any] = {
    "provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "api_base": _normalize_api_base(
        "openai",
        os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    ),
    "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("OPENAI_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("OPENAI_TIMEOUT", "300")),
    "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("OPENAI_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("OPENAI_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "6")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "3.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}

DEEPSEEK_CONFIG: dict[str, any] = {
    "provider": "deepseek",
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "api_base": _normalize_api_base(
        "deepseek",
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    ),
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("DEEPSEEK_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("DEEPSEEK_TIMEOUT", "300")),
    "max_retries": int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("DEEPSEEK_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("DEEPSEEK_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "6")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "3.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}

CUSTOM_CONFIG: dict[str, any] = {
    "provider": "custom",
    "api_key": _resolve_custom_api_key(),
    "api_base": _resolve_custom_api_base(),
    "model": _resolve_custom_model(),
    "temperature": float(os.getenv("CUSTOM_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("CUSTOM_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("CUSTOM_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("CUSTOM_TIMEOUT", "300")),
    "max_retries": int(os.getenv("CUSTOM_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("CUSTOM_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("CUSTOM_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "6")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "3.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}

GEMINI_CONFIG: dict[str, any] = {
    "provider": "gemini",
    "api_key": _resolve_gemini_api_key(),
    "api_base": os.getenv("GEMINI_API_BASE", "google-genai"),
    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
    "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("GEMINI_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("GEMINI_TIMEOUT", "300")),
    "max_retries": int(os.getenv("GEMINI_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("GEMINI_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("GEMINI_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "6")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "3.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}


# ---------------------------------------------------------------------------
# GitHub Copilot Authentication & Token Exchange
# ---------------------------------------------------------------------------
# Auth flow:
#   1. run.py handles interactive GitHub OAuth device flow login
#   2. GitHub OAuth token is stored at ~/.config/agentskl/github-auth.json
#   3. On each run, the GitHub token is exchanged for a short-lived Copilot API
#      token via https://api.github.com/copilot_internal/v2/token
#   4. The Copilot API token is cached and auto-refreshed when expired.
#
# No dependency on OpenClaw, gh CLI, or any external credential store.

import json as _json
import pathlib as _pathlib
import time as _time

COPILOT_GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_DEVICE_CODE_URL = "https://github.com/login/device/code"
COPILOT_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_COPILOT_DEFAULT_API_BASE = "https://api.individual.githubcopilot.com"

_CONFIG_DIR = _pathlib.Path.home() / ".config" / "agentskl"
GITHUB_AUTH_PATH = _CONFIG_DIR / "github-auth.json"
_COPILOT_TOKEN_CACHE = _CONFIG_DIR / "copilot-token.json"


def _save_json(path: _pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(data, indent=2) + "\n")
    path.chmod(0o600)


def _load_json(path: _pathlib.Path) -> dict | None:
    try:
        if path.exists():
            return _json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    return None


def save_github_token(token: str) -> None:
    """Save a GitHub OAuth token to disk."""
    _save_json(GITHUB_AUTH_PATH, {
        "github_token": token,
        "created_at": int(_time.time()),
    })


def _load_github_token() -> str:
    """Load the stored GitHub OAuth token from disk."""
    data = _load_json(GITHUB_AUTH_PATH)
    if data and isinstance(data.get("github_token"), str):
        return data["github_token"]
    return ""


# ---- Copilot API Token Exchange (automatic, cached) ----

def _load_copilot_token_cache() -> dict | None:
    cached = _load_json(_COPILOT_TOKEN_CACHE)
    if cached and cached.get("expiresAt", 0) - _time.time() * 1000 > 300_000:
        return cached
    return None


def _save_copilot_token_cache(token: str, expires_at: int) -> None:
    _save_json(_COPILOT_TOKEN_CACHE, {
        "token": token,
        "expiresAt": expires_at,
        "updatedAt": int(_time.time() * 1000),
    })


def _derive_copilot_base_url(token: str) -> str:
    """Extract the API base URL from the Copilot token's proxy-ep field."""
    m = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", token)
    if m:
        host = re.sub(r"^https?://", "", m.group(1).strip())
        host = re.sub(r"^proxy\.", "api.", host)
        if host:
            return f"https://{host}"
    return _COPILOT_DEFAULT_API_BASE


def _exchange_copilot_token(github_token: str) -> tuple[str, str]:
    """
    Exchange a GitHub OAuth token for a short-lived Copilot API token.
    Returns (copilot_api_token, api_base_url).
    """
    import requests as _requests

    # Use cache if still valid
    cached = _load_copilot_token_cache()
    if cached:
        tok = cached["token"]
        return tok, _derive_copilot_base_url(tok)

    if not github_token:
        return "", _COPILOT_DEFAULT_API_BASE

    resp = _requests.get(
        _COPILOT_TOKEN_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_token}",
            "Editor-Version": "vscode/1.96.2",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "User-Agent": "GitHubCopilotChat/0.26.7",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    token = data.get("token", "")
    expires_at_raw = data.get("expires_at", 0)
    expires_at = int(expires_at_raw) if isinstance(expires_at_raw, (int, str)) else 0
    if isinstance(expires_at_raw, str):
        expires_at = int(expires_at_raw)
    # Normalize to milliseconds
    if expires_at < 1e10:
        expires_at = expires_at * 1000

    _save_copilot_token_cache(token, expires_at)
    return token, _derive_copilot_base_url(token)


def _resolve_copilot_config() -> tuple[str, str]:
    """Resolve Copilot API key and base URL via token exchange."""
    # Allow direct override via env for CI / testing
    direct_key = os.getenv("COPILOT_API_KEY", "")
    if direct_key:
        return direct_key, os.getenv("COPILOT_API_BASE", _COPILOT_DEFAULT_API_BASE)

    github_token = _load_github_token()
    if not github_token:
        return "", _COPILOT_DEFAULT_API_BASE

    try:
        return _exchange_copilot_token(github_token)
    except Exception:
        return "", _COPILOT_DEFAULT_API_BASE


_copilot_api_key, _copilot_api_base = _resolve_copilot_config()

COPILOT_CONFIG: dict[str, any] = {
    "provider": "copilot",
    "api_key": _copilot_api_key,
    "api_base": _copilot_api_base,
    "model": os.getenv("COPILOT_MODEL", "claude-sonnet-4"),
    "temperature": float(os.getenv("COPILOT_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("COPILOT_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("COPILOT_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("COPILOT_TIMEOUT", "300")),
    "max_retries": int(os.getenv("COPILOT_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("COPILOT_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("COPILOT_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "6")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "3.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}

ANTHROPIC_CONFIG: dict[str, any] = {
    "provider": "anthropic",
    "api_key": os.getenv("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_AUTH_TOKEN", "")),
    "api_base": _normalize_api_base(
        "anthropic",
        os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
    ),
    "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("ANTHROPIC_TOP_P", "1.0")),
    "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000")),
    "timeout": int(os.getenv("ANTHROPIC_TIMEOUT", "300")),
    "max_retries": int(os.getenv("ANTHROPIC_MAX_RETRIES", "3")),
    "backoff": float(os.getenv("ANTHROPIC_BACKOFF", "2.0")),
    "backoff_max": float(os.getenv("ANTHROPIC_BACKOFF_MAX", "60.0")),
    "transient_extra_retries": int(os.getenv("TRANSIENT_EXTRA_RETRIES", "8")),
    "transient_min_wait": float(os.getenv("TRANSIENT_MIN_WAIT", "4.0")),
    "command_timeout": int(os.getenv("COMMAND_TIMEOUT", "1800")),
    "pragma_dse_jobs": _PRAGMA_DSE_JOBS,
    "pragma_dse_max_candidates": _PRAGMA_DSE_MAX_CANDIDATES,
    "pragma_dse_search_strategy": _PRAGMA_DSE_SEARCH_STRATEGY,
    "pragma_dse_max_combos": _PRAGMA_DSE_MAX_COMBOS,
    "pragma_dse_top_per_site": _PRAGMA_DSE_TOP_PER_SITE,
    "pragma_dse_beam_width": _PRAGMA_DSE_BEAM_WIDTH,
    "pragma_dse_beam_temperature": _PRAGMA_DSE_BEAM_TEMPERATURE,
    "pragma_dse_beam_cooling": _PRAGMA_DSE_BEAM_COOLING,
    "pragma_dse_random_combo_fraction": _PRAGMA_DSE_RANDOM_COMBO_FRACTION,
    "pragma_dse_candidate_timeout_sec": _PRAGMA_DSE_CANDIDATE_TIMEOUT_SEC,
    "max_hardware_opt_rounds": _MAX_HARDWARE_OPT_ROUNDS,
    **_CSIM_VERIFICATION_CONFIG,
}


def get_llm_config() -> dict[str, any]:
    """Select LLM provider via LLM_PROVIDER, else by key availability."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider in ("gemini", "google"):
        return GEMINI_CONFIG
    if provider == "anthropic":
        return ANTHROPIC_CONFIG
    if provider == "copilot":
        return COPILOT_CONFIG
    if provider == "openrouter":
        return OPENROUTER_CONFIG
    if provider == "openai":
        return OPENAI_CONFIG
    if provider == "deepseek":
        return DEEPSEEK_CONFIG
    if provider == "custom":
        return CUSTOM_CONFIG

    if _resolve_gemini_api_key():
        return GEMINI_CONFIG
    if os.getenv("CUSTOM_API_KEY"):
        return CUSTOM_CONFIG
    if os.getenv("OPENAI_API_KEY"):
        return OPENAI_CONFIG
    if os.getenv("DEEPSEEK_API_KEY"):
        return DEEPSEEK_CONFIG
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return ANTHROPIC_CONFIG
    if _load_github_token():
        return COPILOT_CONFIG
    return OPENROUTER_CONFIG


LLM_CONFIG = get_llm_config()

SKILLS_DIR = os.getenv("SKILLS_DIR")
OUT_DIR = os.getenv("RUNS_OUT_DIR")
