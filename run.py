import argparse
import importlib
import os
import re
import subprocess
import shutil
import sys
import time
import termios
import tty
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from getpass import getpass
from typing import Optional

import requests

try:
    from . import config as cfg
    from . import agent as agent_runtime
    from .langgraph_agent import LangGraphAgent as Agent
except ImportError:
    import config as cfg
    import agent as agent_runtime
    from langgraph_agent import LangGraphAgent as Agent


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROVIDER_STATE_FILE = os.path.join(_SCRIPT_DIR, ".llm_provider")
_ENV_STORE_FILE = os.path.join(_SCRIPT_DIR, ".llm_env")

_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "copilot": "copilot",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "deepseek": "deepseek",
    "ds": "deepseek",
    "custom": "custom",
    "openrouter": "openrouter",
    "openai": "openai",
    "chatgpt": "openai",
}

# Fallback model list when API is unreachable. Updated 2026-03-03.
_MODEL_PRESETS = {
    "copilot": [
        # Claude
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-sonnet-4.6",
        "claude-opus-4.5",
        "claude-opus-4.6",
        "claude-haiku-4.5",
        # GPT
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        # Gemini
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ],
    "anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-7-sonnet-latest"],
    "openrouter": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-001",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "custom": [
        "deepseek-v3.2",
        "glm47",
        "minimax-m25",
        "gpt-5.2-codex",
        "kimi-k2p5",
        "minimax-m21",
    ],

}

_MODEL_ENV = {
    "copilot": "COPILOT_MODEL",
    "gemini": "GEMINI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "openrouter": "OPENROUTER_MODEL",
    "openai": "OPENAI_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
    "custom": "CUSTOM_MODEL",
}

_INPUT_MODE_ALIASES = {
    "plain_c": "plain_c",
    "plainc": "plain_c",
    "plain-c": "plain_c",
    "hls_native": "hls_native",
    "hlsnative": "hls_native",
    "hls-native": "hls_native",
}

_INTERRUPTABLE_NODES = [
    "profiling",
    "kg_rag",
    "rewrite_guidance",
    "software_rewrite",
    "hardware_rewrite",
    "pragma_tuning",
    "pragma_dse",
]

_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "custom": "CUSTOM_API_KEY",
}

_COLOR_ENABLED = os.isatty(1) and os.getenv("NO_COLOR") is None


class _WorkingIndicator:
    def __init__(self, message: str) -> None:
        self.message = message
        self.enabled = os.isatty(1)
        self._frames = ["(^_^)", "(o_o)", "(-_-)", "(^.^)"]
        self._index = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def _render_line(self) -> str:
        frame = self._frames[self._index % len(self._frames)]
        return _color(f"{frame} {self.message}", "35")

    def _write_line(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\033[2K\r" + self._render_line())
        sys.stdout.flush()

    def _clear_line(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    def _loop(self) -> None:
        while not self._stop.wait(0.12):
            with self._lock:
                if self._stop.is_set():
                    return
                self._index += 1
                self._write_line()

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        print("")
        with self._lock:
            self._write_line()
        self._thread = threading.Thread(target=self._loop, name="hlsclaw-working-indicator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._clear_line()
        self._started = False

    def before_print(self) -> None:
        if not self.enabled or not self._started:
            return
        self._lock.acquire()
        self._clear_line()

    def after_print(self) -> None:
        if not self.enabled or not self._started:
            return
        try:
            self._write_line()
        finally:
            self._lock.release()


def _color(text: str, code: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def _label(icon: str, level: str) -> str:
    palette = {"INFO": "36", "WARN": "33", "ERROR": "31", "STEP": "35"}
    return _color(f"{icon} [{level}]", palette.get(level, "0"))


def _info(message: str) -> None:
    print(f"{_label('ℹ', 'INFO')} {message}")


def _warn(message: str) -> None:
    print(f"{_label('⚠', 'WARN')} {message}")


def _exit_message() -> None:
    print("")
    print(_color("✋ Exit", "33"))


def _step(message: str) -> None:
    print(f"{_label('➤', 'STEP')} {message}")


def _title() -> None:
    banner = [
        "==============================",
        "          HLSClaw",
        "==============================",
    ]
    for line in banner:
        print(_color(line, "36"))


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = sys.stdin.read(1)
            if nxt == "[":
                code = sys.stdin.read(1)
                if code == "A":
                    return "UP"
                if code == "B":
                    return "DOWN"
            return "ESC"
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _interactive_select(
    title: str,
    options: list[tuple[str, str]],
    default_index: int = 0,
    page_size: int = 12,
) -> str:
    if not options:
        raise ValueError("No selectable options.")
    if not os.isatty(0):
        return options[max(0, min(default_index, len(options) - 1))][1]

    selected = max(0, min(default_index, len(options) - 1))
    visible_rows = min(page_size, len(options))
    redraw_lines = visible_rows + 1

    _step(title)
    _info("Use ↑/↓ to move, Enter to confirm.")
    print(_color("  ctrl+c to exit", "90"))

    def _draw() -> None:
        half = visible_rows // 2
        start = max(0, selected - half)
        end = min(len(options), start + visible_rows)
        start = max(0, end - visible_rows)
        window = options[start:end]

        for row in range(visible_rows):
            if row < len(window):
                label, _ = window[row]
                absolute = start + row
                if absolute == selected:
                    line = f"  {_color('❯', '32')} {_color(label, '1;37')}"
                else:
                    line = f"    {label}"
            else:
                line = ""
            sys.stdout.write("\033[2K\r" + line + "\n")

        status = f"{selected + 1}/{len(options)}"
        sys.stdout.write("\033[2K\r" + _color(f"  [{status}]", "90") + "\n")
        sys.stdout.flush()

    _draw()
    while True:
        key = _read_key()
        if key == "UP":
            selected = (selected - 1) % len(options)
        elif key == "DOWN":
            selected = (selected + 1) % len(options)
        elif key == "ENTER":
            print("")
            return options[selected][1]
        else:
            continue
        sys.stdout.write(f"\033[{redraw_lines}A")
        sys.stdout.flush()
        _draw()


def _normalize_provider(provider: Optional[str]) -> str:
    key = (provider or "").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider: {provider}")
    return _PROVIDER_ALIASES[key]


def _read_env_store() -> dict[str, str]:
    if not os.path.isfile(_ENV_STORE_FILE):
        return {}
    values: dict[str, str] = {}
    try:
        with open(_ENV_STORE_FILE, "r", encoding="utf-8") as f:
            for raw in f:
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
    except OSError:
        return {}
    return values


def _write_env_store(values: dict[str, str]) -> None:
    try:
        with open(_ENV_STORE_FILE, "w", encoding="utf-8") as f:
            for key in sorted(values.keys()):
                f.write(f"{key}={values[key]}\n")
        os.chmod(_ENV_STORE_FILE, 0o600)
    except OSError:
        pass


def _persist_env_var(key: str, value: str) -> None:
    values = _read_env_store()
    if value:
        values[key] = value
    elif key in values:
        del values[key]
    _write_env_store(values)


def _load_local_env() -> None:
    values = _read_env_store()
    for key, val in values.items():
        if not os.getenv(key):
            os.environ[key] = val


def _load_saved_provider() -> Optional[str]:
    if not os.path.isfile(_PROVIDER_STATE_FILE):
        return None
    try:
        saved = open(_PROVIDER_STATE_FILE, "r", encoding="utf-8").read().strip()
    except OSError:
        return None
    if not saved:
        return None
    try:
        return _normalize_provider(saved)
    except ValueError:
        return None


def _save_provider(provider: str) -> None:
    try:
        with open(_PROVIDER_STATE_FILE, "w", encoding="utf-8") as f:
            f.write(provider + "\n")
    except OSError:
        pass


def _fallback_provider_from_env() -> str:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.getenv("COPILOT_API_KEY") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"):
        return "copilot"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("CUSTOM_API_KEY"):
        return "custom"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "anthropic"


def _prompt_provider(default_provider: str) -> str:
    options = [
        ("🐙 copilot", "copilot"),
        ("💠 gemini (google)", "gemini"),
        ("🤖 chatgpt (openai)", "openai"),
        ("🟦 deepseek", "deepseek"),
        ("🧪 custom", "custom"),
        ("🧠 claude (anthropic)", "anthropic"),
        ("🌐 openrouter", "openrouter"),
    ]
    default_index = 0
    for i, (_, provider) in enumerate(options):
        if provider == default_provider:
            default_index = i
            break
    return _interactive_select("Select LLM provider", options, default_index=default_index, page_size=6)


def _normalize_input_mode(value: Optional[str]) -> str:
    key = (value or "").strip().lower()
    if key not in _INPUT_MODE_ALIASES:
        raise ValueError(f"Unsupported input mode: {value}")
    return _INPUT_MODE_ALIASES[key]


def _prompt_input_mode(default_mode: str) -> str:
    options = [
        ("🧪 plain_c optimization pipeline starts", "plain_c"),
        ("🧩 hls_native optimization pipeline starts", "hls_native"),
    ]
    default_index = 0
    for i, (_, mode) in enumerate(options):
        if mode == default_mode:
            default_index = i
            break
    return _interactive_select("Select optimization mode", options, default_index=default_index, page_size=4)


def _resolve_input_mode(cli_mode: Optional[str]) -> str:
    if cli_mode:
        return _normalize_input_mode(cli_mode)
    if not os.isatty(0):
        return "plain_c"
    return _prompt_input_mode("plain_c")


def _resolve_provider(cli_provider: Optional[str]) -> str:
    if cli_provider:
        return _normalize_provider(cli_provider)
    if not os.isatty(0):
        if os.getenv("LLM_PROVIDER"):
            return _normalize_provider(os.getenv("LLM_PROVIDER"))
        return _load_saved_provider() or _fallback_provider_from_env()

    default_provider = None
    if os.getenv("LLM_PROVIDER"):
        default_provider = _normalize_provider(os.getenv("LLM_PROVIDER"))
    if not default_provider:
        default_provider = _load_saved_provider() or _fallback_provider_from_env()
    return _prompt_provider(default_provider)


def _ensure_copilot_auth() -> None:
    if os.getenv("COPILOT_API_KEY") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"):
        return
    # Check if we already have a stored GitHub OAuth token (from device flow)
    if cfg._load_github_token():
        return
    if not os.isatty(0):
        raise RuntimeError(
            "Copilot selected but no token found and terminal is non-interactive. "
            "Set COPILOT_API_KEY or run in interactive mode first."
        )

    import requests as _requests
    import time as _time

    _info("Copilot selected. Starting GitHub device flow login ...")

    # Step 1: request device code
    resp = _requests.post(
        cfg.COPILOT_DEVICE_CODE_URL,
        data={"client_id": cfg.COPILOT_GITHUB_CLIENT_ID, "scope": "read:user"},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    device = resp.json()

    user_code = device["user_code"]
    verification_uri = device["verification_uri"]
    device_code = device["device_code"]
    expires_in = device.get("expires_in", 900)
    interval = max(device.get("interval", 5), 5)

    print()
    print("=" * 56)
    print("  GitHub Copilot Login")
    print("=" * 56)
    print()
    print(f"  1. Open:  {verification_uri}")
    print(f"  2. Enter: {user_code}")
    print()
    print("  Waiting for authorization...", end="", flush=True)

    # Step 2: poll for access token
    deadline = _time.time() + expires_in
    while _time.time() < deadline:
        _time.sleep(interval)
        r = _requests.post(
            cfg.COPILOT_ACCESS_TOKEN_URL,
            data={
                "client_id": cfg.COPILOT_GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        body = r.json()
        if "access_token" in body:
            cfg.save_github_token(body["access_token"])
            print(" ✅")
            print(f"  Saved to {cfg.GITHUB_AUTH_PATH}")
            print()
            return
        err = body.get("error", "")
        if err == "authorization_pending":
            print(".", end="", flush=True)
            continue
        if err == "slow_down":
            interval += 2
            continue
        if err == "expired_token":
            print(" ❌")
            raise RuntimeError("Device code expired. Please try again.")
        if err == "access_denied":
            print(" ❌")
            raise RuntimeError("Login cancelled by user.")
        print(f" ❌ ({err})")
        raise RuntimeError(f"GitHub device flow error: {err}")

    print(" ❌")
    raise RuntimeError("Device code expired. Please try again.")




def _ensure_openai_auth() -> None:
    env_key = "OPENAI_API_KEY"
    if os.getenv(env_key):
        return
    if not os.isatty(0):
        raise RuntimeError(f"openai selected but {env_key} is not set in non-interactive mode.")

    api_base = (os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").strip().rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    def _verify_key(candidate: str) -> tuple[bool, str]:
        key = (candidate or "").strip()
        if not key:
            return False, f"{env_key} cannot be empty."
        try:
            r = requests.get(
                f"{api_base}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            if r.status_code == 200:
                return True, ""
            preview = (r.text or "").strip().replace("\n", "\\n")
            if len(preview) > 300:
                preview = preview[:300] + "...(truncated)"
            return False, f"HTTP {r.status_code}: {preview}"
        except requests.RequestException as exc:
            return False, f"Request failed: {exc}"

    state: dict[str, str] = {"api_key": "", "error": ""}
    done = threading.Event()

    def _login_page() -> str:
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OpenAI Login</title>
    <style>
      :root {{
        --bg: #0b1020;
        --card: rgba(255, 255, 255, 0.06);
        --text: #e7eefc;
        --muted: rgba(231, 238, 252, 0.72);
        --accent: #10a37f;
        --border: rgba(255, 255, 255, 0.14);
        --danger: #ff5a5f;
      }}
      html, body {{ height: 100%; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        background: radial-gradient(1200px 800px at 10% 0%, rgba(16,163,127,0.14), transparent 60%),
                    radial-gradient(800px 600px at 100% 30%, rgba(99,102,241,0.18), transparent 55%),
                    var(--bg);
        color: var(--text);
        display: grid;
        place-items: center;
        padding: 24px;
      }}
      .card {{
        width: min(720px, 100%);
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
      }}
      h1 {{ margin: 0 0 8px; font-size: 20px; }}
      p {{ margin: 0 0 14px; color: var(--muted); line-height: 1.45; font-size: 14px; }}
      .row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
      input {{
        flex: 1 1 360px;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,0.25);
        color: var(--text);
        font-size: 14px;
        outline: none;
      }}
      button {{
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid rgba(16,163,127,0.45);
        background: rgba(16,163,127,0.18);
        color: var(--text);
        font-weight: 600;
        cursor: pointer;
      }}
      button:hover {{ background: rgba(16,163,127,0.28); }}
      a {{
        color: rgba(231,238,252,0.88);
        text-decoration: none;
        border-bottom: 1px dashed rgba(231,238,252,0.35);
      }}
      a:hover {{ border-bottom-color: rgba(231,238,252,0.75); }}
      .note {{ margin-top: 12px; font-size: 12px; color: rgba(231,238,252,0.55); }}
      code {{ color: rgba(231,238,252,0.85); }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>OpenAI Login</h1>
      <p>Paste your <code>{env_key}</code>, then click Login. It will be verified via <code>{api_base}/models</code> and stored in <code>{_ENV_STORE_FILE}</code> (mode 600).</p>
      <form class="row" method="post" action="/login">
        <input name="api_key" type="password" placeholder="sk-..." autocomplete="off" spellcheck="false" />
        <button type="submit">Login</button>
      </form>
      <p>Need a key? <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer">Open API keys page</a></p>
      <div class="note">This page is served from localhost only. Your key is only used for verification calls.</div>
    </div>
  </body>
</html>
"""

    def _result_page(ok: bool, message: str) -> str:
        title = "Login Successful" if ok else "Login Failed"
        border = "rgba(16,163,127,0.55)" if ok else "rgba(255,90,95,0.55)"
        bg = "rgba(16,163,127,0.12)" if ok else "rgba(255,90,95,0.12)"
        msg = (message or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        background: #0b1020;
        color: #e7eefc;
        display: grid;
        place-items: center;
        padding: 24px;
      }}
      .card {{
        width: min(720px, 100%);
        background: {bg};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 20px;
      }}
      h1 {{ margin: 0 0 10px; font-size: 18px; }}
      pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 13px;
        color: rgba(231,238,252,0.9);
      }}
      a {{
        display: inline-block;
        margin-top: 12px;
        color: rgba(231,238,252,0.88);
        text-decoration: none;
        border-bottom: 1px dashed rgba(231,238,252,0.35);
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>{title}</h1>
      <pre>{msg}</pre>
      <a href="/">Back</a>
    </div>
  </body>
</html>
"""

    def _write_html(handler: BaseHTTPRequestHandler, status: int, body: str) -> None:
        data = body.encode("utf-8", errors="replace")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                _write_html(self, HTTPStatus.OK, _login_page())
                return
            if self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/login":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                clen = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                clen = 0
            if clen <= 0 or clen > 16 * 1024:
                _write_html(self, HTTPStatus.BAD_REQUEST, _result_page(False, "Invalid request body."))
                return
            raw = self.rfile.read(clen).decode("utf-8", errors="replace")
            params = urllib.parse.parse_qs(raw)
            key = (params.get("api_key") or [""])[0].strip()
            ok, msg = _verify_key(key)
            if not ok:
                state["error"] = msg
                _write_html(self, HTTPStatus.UNAUTHORIZED, _result_page(False, msg))
                return
            state["api_key"] = key
            done.set()
            _write_html(self, HTTPStatus.OK, _result_page(True, "✅ Verified and saved. You can close this tab and return to the terminal."))

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/"

    _info(f"OpenAI selected. Open this URL to login: {url}")
    opener = shutil.which("xdg-open")
    if opener:
        try:
            subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    timeout_s = int(os.getenv("OPENAI_LOGIN_TIMEOUT", "600"))
    try:
        if not done.wait(timeout=timeout_s):
            raise RuntimeError(f"OpenAI login timed out after {timeout_s}s.")
    finally:
        server.shutdown()
        server.server_close()

    api_key = state.get("api_key", "").strip()
    if not api_key:
        raise RuntimeError(state.get("error", "OpenAI login failed."))

    os.environ[env_key] = api_key
    _persist_env_var(env_key, api_key)
    _info(f"Saved {env_key} to local env store: {_ENV_STORE_FILE}")


def _fetch_dynamic_models(provider: str) -> list[str]:
    api_base_env = {
        "copilot": ("COPILOT_API_BASE", "https://api.individual.githubcopilot.com"),
        "gemini": ("GEMINI_API_BASE", "google-genai"),
        "anthropic": ("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
        "openrouter": ("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        "openai": ("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "deepseek": ("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        "custom": ("CUSTOM_API_BASE", ""),
    }
    key_env = {
        "copilot": ("COPILOT_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "openrouter": ("OPENROUTER_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "custom": ("CUSTOM_API_KEY",),
    }
    api_base_key, api_base_default = api_base_env.get(provider, ("", ""))
    if not api_base_key:
        return []
    api_base = os.getenv(api_base_key, api_base_default).rstrip("/")
    if provider == "gemini":
        try:
            from google import genai
        except ImportError:
            return []
        token = ""
        for env_name in key_env.get(provider, ()):
            token = os.getenv(env_name, "")
            if token:
                break
        if not token:
            return []
        try:
            client = genai.Client(api_key=token)
            models: list[str] = []
            for item in client.models.list():
                name = str(getattr(item, "name", "") or "").strip()
                actions = getattr(item, "supported_actions", None) or []
                if actions and "generateContent" not in actions:
                    continue
                if name.startswith("models/"):
                    name = name.split("/", 1)[1]
                if name:
                    models.append(name)
            return sorted(set(models))
        except Exception:
            return []
    if provider in ("anthropic", "deepseek") and not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    token = ""
    for env_name in key_env.get(provider, ()):
        token = os.getenv(env_name, "")
        if token:
            break
    if (not token) and provider == "copilot":
        # Use the device flow token from config.py
        try:
            copilot_key, copilot_base = cfg._resolve_copilot_config()
            if copilot_key:
                token = copilot_key
                api_base = copilot_base
        except Exception:
            pass
    if not token:
        return []

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if provider == "anthropic":
        headers.pop("Authorization", None)
        headers["x-api-key"] = token
        headers["anthropic-version"] = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    elif provider == "openrouter":
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
        headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE", "skills-agent")
    elif provider == "copilot":
        headers["Editor-Version"] = "vscode/1.96.2"
        headers["Editor-Plugin-Version"] = "copilot-chat/0.26.7"
        headers["User-Agent"] = "GitHubCopilotChat/0.26.7"
        headers["Copilot-Integration-Id"] = "vscode-chat"

    try:
        resp = requests.get(f"{api_base}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        raw_models = data.get("data", []) if isinstance(data, dict) else []
        models: list[str] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("id", "")).strip()
            if mid:
                models.append(mid)
        return sorted(set(models))
    except Exception:
        return []


def _normalize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_id.strip().lower())


def _pick_supported_model(requested_model: str, available_models: list[str]) -> str:
    if not available_models:
        return requested_model
    if requested_model in available_models:
        return requested_model

    req_norm = _normalize_model_id(requested_model)
    if req_norm:
        for model in available_models:
            if _normalize_model_id(model) == req_norm:
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


def _prompt_model(provider: str, default_model: str) -> str:
    if not os.isatty(0):
        return default_model

    presets = list(_MODEL_PRESETS.get(provider, []))
    dynamic_models = _fetch_dynamic_models(provider)
    if dynamic_models:
        presets = dynamic_models
        _info(f"Loaded {len(presets)} models from {provider} endpoint.")
    elif provider == "copilot":
        _warn("Could not load full Copilot model list from API; using preset shortlist.")

    if not presets:
        presets = [default_model]

    options = [(m, m) for m in presets]
    options.append((_color("✍ custom model id", "33"), "__custom__"))
    default_index = 0
    for i, m in enumerate(presets):
        if m == default_model:
            default_index = i
            break

    icon = {"copilot": "🐙", "openai": "🤖", "deepseek": "🟦", "custom": "🧪", "anthropic": "🧠", "openrouter": "🌐"}.get(provider, "•")
    if provider == "gemini":
        icon = "💠"
    selected = _interactive_select(
        f"Select model for {provider} {icon}",
        options,
        default_index=default_index,
        page_size=12,
    )
    if selected == "__custom__":
        custom = input("Enter custom model id: ").strip()
        if custom and custom.lower() != "custom":
            return custom
        _warn("Invalid custom model id; falling back to default.")
        return default_model
    return selected


def _ensure_api_key(provider: str) -> None:
    env_key = _KEY_ENV[provider]
    if os.getenv(env_key):
        return
    if provider == "gemini":
        google_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if google_key:
            os.environ["GEMINI_API_KEY"] = google_key
            _persist_env_var("GEMINI_API_KEY", google_key)
            return
    if not os.isatty(0):
        raise RuntimeError(
            f"{provider} selected but {env_key} is not set in non-interactive mode."
        )
    api_key = getpass(f"Enter {env_key}: ").strip()
    if not api_key:
        raise RuntimeError(f"{env_key} cannot be empty.")
    os.environ[env_key] = api_key
    _persist_env_var(env_key, api_key)
    _info(f"Saved {env_key} to local env store: {_ENV_STORE_FILE}")


def _configure_provider(provider: str) -> None:
    os.environ["LLM_PROVIDER"] = provider
    _persist_env_var("LLM_PROVIDER", provider)

    if provider == "copilot":
        _ensure_copilot_auth()
    elif provider == "openai":
        _ensure_openai_auth()
    elif provider in ("gemini", "anthropic", "openrouter", "deepseek", "custom"):
        _ensure_api_key(provider)

    model_env_key = _MODEL_ENV[provider]
    stored_model = (os.getenv(model_env_key) or "").strip()
    if (not stored_model) or stored_model.lower() == "custom":
        stored_model = "kimi-k2p5" if provider == "custom" else _MODEL_PRESETS[provider][0]
    default_model = stored_model
    selected_model = _prompt_model(provider, default_model)

    available_models = _fetch_dynamic_models(provider)
    if available_models:
        resolved_model = _pick_supported_model(selected_model, available_models)
        if resolved_model != selected_model:
            _warn(
                f"Model '{selected_model}' is not available on current {provider} endpoint; "
                f"falling back to '{resolved_model}'."
            )
            selected_model = resolved_model

    os.environ[model_env_key] = selected_model
    _persist_env_var(model_env_key, selected_model)


def _prompt_path(label: str, current: Optional[str], must_be_file: bool = False, icon: str = "📂") -> str:
    if current:
        return current
    if not os.isatty(0):
        raise RuntimeError(f"{label} is required in non-interactive mode.")

    while True:
        value = input(f"{_color(f'{icon} {label}', '1;36')}: ").strip()
        if not value:
            continue
        if not os.path.exists(value):
            _warn(f"Path does not exist: {value}")
            continue
        if must_be_file and not os.path.isfile(value):
            _warn(f"Expect file path, got: {value}")
            continue
        return value


def _prompt_top_function(current: Optional[str]) -> str:
    if current:
        return current
    if not os.isatty(0):
        raise RuntimeError("Top function name (--top) is required in non-interactive mode.")
    while True:
        top = input(f"{_color('🔧 Top function name (--top)', '1;36')}: ").strip()
        if top:
            return top


def _build_thread_agent(config_module) -> Agent:
    skills_dir = config_module.SKILLS_DIR or os.path.join(_SCRIPT_DIR, "skills")
    out_dir = config_module.OUT_DIR or os.path.join(_SCRIPT_DIR, "runs")
    return Agent(
        config=config_module.LLM_CONFIG,
        system_prompt=config_module.SYSTEM_PROMPT,
        code_path=os.path.join(_SCRIPT_DIR, "__thread_query__.cc"),
        req_path=os.path.join(_SCRIPT_DIR, "__thread_query__.json"),
        top_function_name="",
        skills_dir=skills_dir,
        out_dir=out_dir,
    )


def _prompt_thread_action() -> str:
    options = [
        ("🆕 start new run", "new_run"),
        ("▶ resume thread", "resume"),
        ("👁 show thread", "show_thread"),
        ("📚 list threads", "list_threads"),
        ("🗑 delete thread", "delete_thread"),
        ("🧹 prune stale threads", "prune_threads"),
    ]
    return _interactive_select("Select HLSClaw action", options, default_index=0, page_size=8)


def _thread_label(record: dict[str, object]) -> str:
    thread_id = str(record.get("thread_id", ""))
    status = str(record.get("execution_status", ""))
    reason = str(record.get("status_reason", ""))
    run_dir = str(record.get("run_dir", ""))
    run_name = ""
    if run_dir:
        normalized = os.path.normpath(run_dir)
        base = os.path.basename(normalized)
        parent = os.path.basename(os.path.dirname(normalized))
        run_name = os.path.join(parent, base) if parent and parent not in {"runs", "results"} else base
    pending = ", ".join(record.get("pending_nodes", []) or [])
    suffix = f" | {run_name}" if run_name else ""
    if reason:
        suffix += f" | {reason}"
    if pending:
        suffix += f" | {pending}"
    return f"{thread_id} | {status}{suffix}"


def _pause_for_menu_return(message: str = "Press Enter to return to the action menu.") -> None:
    if not os.isatty(0):
        return
    input(_color(message, "90") + " ")


def _prompt_thread_record(
    agent: Agent,
    title: str,
    allow_back: bool = False,
    include_completed: bool = False,
    resumable_only: bool = False,
) -> Optional[str]:
    records = agent.list_threads(include_completed=include_completed)
    if resumable_only:
        records = [record for record in records if bool(record.get("resumable", False))]
    if not records:
        if include_completed:
            raise RuntimeError("No persisted threads found.")
        raise RuntimeError("No resumable threads found.")
    options = [(_thread_label(record), str(record.get("thread_id", ""))) for record in records]
    if allow_back:
        options.append(("↩ back", "__back__"))
    default_index = 0
    for i, record in enumerate(records):
        if record.get("execution_status") in {"interrupted", "external_abort", "fatal_error"}:
            default_index = i
            break
    selected = _interactive_select(title, options, default_index=default_index, page_size=10)
    if allow_back and selected == "__back__":
        return None
    return selected


def _handle_thread_action(
    parser: argparse.ArgumentParser,
    config_module,
    action: str,
    interactive: bool = False,
    thread_id_arg: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    agent = _build_thread_agent(config_module)
    try:
        try:
            if action == "list_threads":
                records = agent.list_threads()
                if not records:
                    _info("No resumable threads found.")
                else:
                    for record in records:
                        pending = ", ".join(record.get("pending_nodes", []) or [])
                        resumable = "yes" if record.get("resumable", False) else "no"
                        _info(
                            f"{record.get('thread_id')} | {record.get('execution_status', '')} | "
                            f"reason={record.get('status_reason', '')} | resumable={resumable} | "
                            f"run_dir={record.get('run_dir', '')} | pending={pending}"
                        )
                return True, None

            if action == "show_thread":
                thread_id = thread_id_arg
                if not thread_id or thread_id == "__interactive__":
                    thread_id = _prompt_thread_record(
                        agent,
                        "Select thread to inspect",
                        allow_back=interactive,
                    )
                if thread_id is None:
                    return False, None
                record = agent.get_thread_record(thread_id)
                _info(f"Thread ID: {record.get('thread_id', '')}")
                _info(f"Execution status: {record.get('execution_status', '')}")
                _info(f"Status reason: {record.get('status_reason', '')}")
                if record.get("total_tokens_used") is not None:
                    _info(f"Total tokens used: {record.get('total_tokens_used', 0)}")
                _info(f"Resumable: {record.get('resumable', False)}")
                if record.get("failing_stage"):
                    _info(f"Failing stage: {record.get('failing_stage', '')}")
                _info(f"Pending nodes: {', '.join(record.get('pending_nodes', []) or [])}")
                _info(f"Run dir: {record.get('run_dir', '')}")
                _info(f"State path: {record.get('state_path', '')}")
                _info(f"Log file: {record.get('log_file', '')}")
                _info(f"Checkpoint DB: {record.get('checkpoint_db_path', '')}")
                if record.get("updated_at"):
                    _info(f"Updated at: {record.get('updated_at')}")
                return True, None

            if action == "delete_thread":
                thread_id = thread_id_arg
                if not thread_id or thread_id == "__interactive__":
                    thread_id = _prompt_thread_record(
                        agent,
                        "Select thread to delete",
                        allow_back=interactive,
                    )
                if thread_id is None:
                    return False, None
                result = agent.delete_thread(thread_id)
                _info(f"Deleted thread: {result.get('thread_id', '')}")
                _info(f"Run dir kept: {result.get('run_dir', '')}")
                removed_files = result.get("removed_files", [])
                if removed_files:
                    for path in removed_files:
                        _info(f"Removed file: {path}")
                else:
                    _info("No persisted files were removed.")
                return True, None

            if action == "prune_threads":
                removed = agent.prune_threads()
                if not removed:
                    _info("No stale thread entries found.")
                else:
                    for item in removed:
                        _info(
                            f"Pruned thread: {item.get('thread_id', '')} | "
                            f"run_dir={item.get('run_dir', '')}"
                        )
                return True, None

            if action == "resume":
                thread_id = _prompt_thread_record(
                    agent,
                    "Select thread to resume",
                    allow_back=interactive,
                    resumable_only=True,
                )
                return (thread_id is not None), thread_id

            parser.error(f"[ERROR] Unsupported thread action: {action}")
        except (ValueError, RuntimeError) as exc:
            if interactive:
                _warn(str(exc))
                return False, None
            parser.error(f"[ERROR] {exc}")
    finally:
        agent.close()
    return False, None


def main() -> None:
    parser = argparse.ArgumentParser(description="HLSClaw interactive runner")
    parser.add_argument(
        "--code",
        dest="code_path",
        type=str,
        required=False,
        help="Path to HLS C/C++ source file. If omitted, prompt in terminal.",
    )
    parser.add_argument(
        "--req",
        dest="req_path",
        type=str,
        required=False,
        help="Path to requirement file/folder. If omitted, prompt in terminal.",
    )
    parser.add_argument(
        "--top",
        dest="top_function_name",
        type=str,
        required=False,
        help="Top function name. Required for new runs; if omitted in interactive mode, prompt in terminal.",
    )
    parser.add_argument(
        "--provider",
        dest="provider",
        type=str,
        required=False,
        choices=["copilot", "gemini", "google", "claude", "claude-code", "anthropic", "openrouter", "openai", "chatgpt", "deepseek", "ds", "custom"],
        help="LLM provider override. If omitted, use interactive provider selection.",
    )
    parser.add_argument(
        "--input-mode",
        dest="input_mode",
        type=str,
        required=False,
        choices=["plain_c", "hls_native"],
        help="Optimization pipeline mode. If omitted, use interactive selection for new runs.",
    )
    parser.add_argument(
        "--thread-id",
        dest="thread_id",
        type=str,
        required=False,
        help="Stable LangGraph thread ID used for persistent checkpointing.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume an existing LangGraph thread from its persisted checkpoint.",
    )
    parser.add_argument(
        "--interrupt-after",
        dest="interrupt_after",
        action="append",
        choices=_INTERRUPTABLE_NODES,
        default=[],
        help="Interrupt the LangGraph run after the named node. Can be provided multiple times.",
    )
    parser.add_argument(
        "--pragma-dse-jobs",
        dest="pragma_dse_jobs",
        type=int,
        required=False,
        help="Number of pragma-DSE candidates to evaluate in parallel. Default: 4.",
    )
    parser.add_argument(
        "--pragma-dse-max-candidates",
        dest="pragma_dse_max_candidates",
        type=int,
        required=False,
        help="Maximum number of pragma-DSE single-site candidates to evaluate.",
    )
    parser.add_argument(
        "--pragma-dse-search-strategy",
        dest="pragma_dse_search_strategy",
        choices=["singles", "progressive"],
        required=False,
        help="Pragma-DSE search strategy. Default comes from config and is usually 'progressive'.",
    )
    parser.add_argument(
        "--pragma-dse-max-combos",
        dest="pragma_dse_max_combos",
        type=int,
        required=False,
        help="Maximum number of progressive pragma-DSE combo candidates to evaluate.",
    )
    parser.add_argument(
        "--pragma-dse-top-per-site",
        dest="pragma_dse_top_per_site",
        type=int,
        required=False,
        help="How many winning single-site candidates to keep per pragma site during progressive DSE.",
    )
    parser.add_argument(
        "--pragma-dse-beam-width",
        dest="pragma_dse_beam_width",
        type=int,
        required=False,
        help="Beam width for progressive pragma-DSE combo expansion.",
    )
    parser.add_argument(
        "--pragma-dse-candidate-timeout-sec",
        dest="pragma_dse_candidate_timeout_sec",
        type=int,
        required=False,
        help="Per-candidate Vitis HLS timeout for pragma-DSE. Use 0 to disable the timeout.",
    )
    parser.add_argument(
        "--show-thread",
        dest="show_thread",
        type=str,
        required=False,
        help="Show persisted metadata for a specific LangGraph thread ID and exit.",
    )
    parser.add_argument(
        "--list-threads",
        dest="list_threads",
        action="store_true",
        help="List persisted LangGraph thread metadata and exit.",
    )
    parser.add_argument(
        "--delete-thread",
        dest="delete_thread",
        type=str,
        required=False,
        help="Delete persisted metadata and checkpoint data for a specific thread ID.",
    )
    parser.add_argument(
        "--prune-threads",
        dest="prune_threads",
        action="store_true",
        help="Remove stale thread index entries that no longer have state files or checkpoints.",
    )

    args = parser.parse_args()
    if os.isatty(0):
        _title()

    _load_local_env()

    if os.isatty(0):
        has_explicit_thread_action = any(
            [
                args.resume,
                bool(args.show_thread),
                args.list_threads,
                bool(args.delete_thread),
                args.prune_threads,
            ]
        )
        has_explicit_run_inputs = any(
            [
                bool(args.code_path),
                bool(args.req_path),
                bool(args.top_function_name),
                bool(args.thread_id),
                bool(args.interrupt_after),
            ]
        )
        if not has_explicit_thread_action and not has_explicit_run_inputs:
            config_module = importlib.reload(cfg)
            try:
                while True:
                    selected_action = _prompt_thread_action()
                    if selected_action == "new_run":
                        break
                    if selected_action == "resume":
                        handled, thread_id = _handle_thread_action(parser, config_module, "resume", interactive=True)
                        if handled and thread_id:
                            args.resume = True
                            args.thread_id = thread_id
                            break
                        continue
                    handled, _ = _handle_thread_action(parser, config_module, selected_action, interactive=True)
                    if handled:
                        _pause_for_menu_return()
            except KeyboardInterrupt:
                _exit_message()
                return

    # Re-load config after provider/model/key updates.
    if args.list_threads or args.show_thread or args.delete_thread or args.prune_threads:
        config_module = importlib.reload(cfg)
        if args.list_threads:
            _handle_thread_action(parser, config_module, "list_threads", interactive=False)
        elif args.show_thread:
            _handle_thread_action(parser, config_module, "show_thread", interactive=False, thread_id_arg=args.show_thread)
        elif args.delete_thread:
            _handle_thread_action(parser, config_module, "delete_thread", interactive=False, thread_id_arg=args.delete_thread)
        else:
            _handle_thread_action(parser, config_module, "prune_threads", interactive=False)
        return

    try:
        try:
            provider = _resolve_provider(args.provider)
            _configure_provider(provider)
        except (ValueError, RuntimeError) as exc:
            parser.error(f"[ERROR] {exc}")
    except KeyboardInterrupt:
        _exit_message()
        return

    config_module = importlib.reload(cfg)
    llm_config = config_module.LLM_CONFIG
    if args.pragma_dse_jobs is not None:
        if args.pragma_dse_jobs <= 0:
            parser.error("[ERROR] --pragma-dse-jobs must be > 0")
        llm_config["pragma_dse_jobs"] = args.pragma_dse_jobs
    if args.pragma_dse_max_candidates is not None:
        if args.pragma_dse_max_candidates <= 0:
            parser.error("[ERROR] --pragma-dse-max-candidates must be > 0")
        llm_config["pragma_dse_max_candidates"] = args.pragma_dse_max_candidates
    if args.pragma_dse_search_strategy is not None:
        llm_config["pragma_dse_search_strategy"] = args.pragma_dse_search_strategy
    if args.pragma_dse_max_combos is not None:
        if args.pragma_dse_max_combos < 0:
            parser.error("[ERROR] --pragma-dse-max-combos must be >= 0")
        llm_config["pragma_dse_max_combos"] = args.pragma_dse_max_combos
    if args.pragma_dse_top_per_site is not None:
        if args.pragma_dse_top_per_site <= 0:
            parser.error("[ERROR] --pragma-dse-top-per-site must be > 0")
        llm_config["pragma_dse_top_per_site"] = args.pragma_dse_top_per_site
    if args.pragma_dse_beam_width is not None:
        if args.pragma_dse_beam_width <= 0:
            parser.error("[ERROR] --pragma-dse-beam-width must be > 0")
        llm_config["pragma_dse_beam_width"] = args.pragma_dse_beam_width
    if args.pragma_dse_candidate_timeout_sec is not None:
        if args.pragma_dse_candidate_timeout_sec < 0:
            parser.error("[ERROR] --pragma-dse-candidate-timeout-sec must be >= 0")
        llm_config["pragma_dse_candidate_timeout_sec"] = args.pragma_dse_candidate_timeout_sec
    resolved_pragma_dse_jobs = llm_config.get("pragma_dse_jobs", 0)
    try:
        resolved_pragma_dse_jobs = int(resolved_pragma_dse_jobs)
    except (TypeError, ValueError):
        resolved_pragma_dse_jobs = 4
    if resolved_pragma_dse_jobs <= 0:
        resolved_pragma_dse_jobs = 4
        llm_config["pragma_dse_jobs"] = resolved_pragma_dse_jobs

    if not llm_config.get("api_key"):
        parser.error(
            "[ERROR] API key not provided. Set ANTHROPIC_API_KEY (for Anthropic) "
            "or COPILOT_API_KEY (for Copilot) "
            "or GEMINI_API_KEY / GOOGLE_API_KEY (for Gemini) "
            "or OPENAI_API_KEY (for OpenAI) "
            "or DEEPSEEK_API_KEY (for DeepSeek) "
            "or OPENROUTER_API_KEY (for OpenRouter) in your environment, "
            "or select Copilot in interactive mode to login via GitHub."
        )

    if args.resume and not args.thread_id:
        if not os.isatty(0):
            parser.error("[ERROR] --resume requires --thread-id")
        try:
            handled, thread_id = _handle_thread_action(parser, config_module, "resume", interactive=False)
        except KeyboardInterrupt:
            _exit_message()
            return
        if not handled or not thread_id:
            parser.error("[ERROR] Failed to select a thread to resume")
        args.thread_id = thread_id

    try:
        try:
            if args.resume:
                code_path = args.code_path or os.path.join(_SCRIPT_DIR, "__resume_placeholder__.cc")
                req_path = args.req_path or os.path.join(_SCRIPT_DIR, "__resume_placeholder__.json")
                top_function_name = args.top_function_name or ""
                input_mode = args.input_mode or "plain_c"
            else:
                input_mode = _resolve_input_mode(args.input_mode)
                code_path = _prompt_path("Code path (--code)", args.code_path, must_be_file=True, icon="📄")
                req_path = _prompt_path("Requirement path (--req)", args.req_path, must_be_file=False, icon="📋")
                top_function_name = _prompt_top_function(args.top_function_name)
        except RuntimeError as exc:
            parser.error(f"[ERROR] {exc}")
        except ValueError as exc:
            parser.error(f"[ERROR] {exc}")
    except KeyboardInterrupt:
        _exit_message()
        return

    skills_dir = config_module.SKILLS_DIR or os.path.join(_SCRIPT_DIR, "skills")
    out_dir = config_module.OUT_DIR or os.path.join(_SCRIPT_DIR, "runs")
    if not skills_dir or not os.path.isdir(skills_dir):
        parser.error("[ERROR] Invalid skills directory")

    _save_provider(provider)
    _info(f"Using provider: {llm_config.get('provider', '<unknown>')}")
    _info(f"Using model: {llm_config.get('model', '<unknown>')}")
    if args.resume and not args.input_mode:
        _info("Using optimization mode: restored from thread state when available")
    else:
        _info(f"Using optimization mode: {input_mode}")
    _info(f"Using pragma DSE candidate parallelism: {llm_config.get('pragma_dse_jobs')}")

    if not args.thread_id:
        args.thread_id = f"hlsclaw-{int(time.time() * 1000)}"

    agent = Agent(
        config=llm_config,
        system_prompt=config_module.SYSTEM_PROMPT,
        code_path=code_path,
        req_path=req_path,
        top_function_name=top_function_name,
        input_mode=input_mode,
        skills_dir=skills_dir,
        out_dir=out_dir,
        interrupt_after=args.interrupt_after,
    )
    working_indicator = _WorkingIndicator(
        "Optimization pipeline running..."
        if not args.resume
        else "Optimization pipeline resumed..."
    )

    try:
        agent_runtime.set_console_status_manager(working_indicator)
        try:
            working_indicator.start()
            if args.resume:
                _info(f"Resuming thread: {args.thread_id}")
                state = agent.resume_langgraph(args.thread_id)
            else:
                state = agent.run_langgraph(thread_id=args.thread_id)
        finally:
            working_indicator.stop()
            agent_runtime.set_console_status_manager(None)
        summary = state.get("final_summary", {})
        thread_id = summary.get("thread_id") or args.thread_id
        if thread_id:
            _info(f"Thread ID: {thread_id}")
        checkpoint_db = summary.get("checkpoint_db_path")
        if checkpoint_db:
            _info(f"Checkpoint DB: {checkpoint_db}")
        execution_status = summary.get("execution_status")
        if execution_status:
            _info(f"Execution status: {execution_status}")
        status_reason = summary.get("status_reason")
        if status_reason:
            _info(f"Status reason: {status_reason}")
        if summary.get("total_tokens_used") is not None:
            _info(f"Total tokens used: {summary.get('total_tokens_used', 0)}")
        pending_nodes = summary.get("pending_nodes") or []
        if pending_nodes:
            _info(f"Pending nodes: {', '.join(pending_nodes)}")
    except KeyboardInterrupt:
        _warn("Interrupted by user (Ctrl+C). Saving resumable thread state...")
        persisted = agent.persist_interrupted_thread(
            args.thread_id,
            reason="Interrupted by user (Ctrl+C)",
            status_reason="keyboard_interrupt",
        )
        if persisted:
            summary = persisted.get("final_summary", {})
            thread_id = summary.get("thread_id") or args.thread_id
            if thread_id:
                _info(f"Thread ID: {thread_id}")
            pending_nodes = summary.get("pending_nodes") or []
            if pending_nodes:
                _info(f"Pending nodes: {', '.join(pending_nodes)}")
            _info("Thread was saved and can be resumed later.")
        raise SystemExit(130)
    except agent_runtime.FatalRunError as exc:
        _warn("Run stopped with a fatal error. Saving thread state...")
        persisted = agent.persist_fatal_error_thread(
            args.thread_id,
            reason=str(exc),
            status_reason=exc.status_reason,
            failing_stage=exc.failing_stage,
            resumable=exc.resumable,
        )
        if persisted:
            summary = persisted.get("final_summary", {})
            thread_id = summary.get("thread_id") or args.thread_id
            if thread_id:
                _info(f"Thread ID: {thread_id}")
            _info(f"Execution status: {summary.get('execution_status', 'fatal_error')}")
            if summary.get("status_reason"):
                _info(f"Status reason: {summary.get('status_reason')}")
            if summary.get("total_tokens_used") is not None:
                _info(f"Total tokens used: {summary.get('total_tokens_used', 0)}")
            if summary.get("failing_stage"):
                _info(f"Failing stage: {summary.get('failing_stage')}")
            if summary.get("resumable"):
                pending_nodes = summary.get("pending_nodes") or []
                if pending_nodes:
                    _info(f"Pending nodes: {', '.join(pending_nodes)}")
                _info("Thread was saved and can be resumed later.")
        raise SystemExit(1)
    except Exception as exc:
        _warn("Run crashed with an unhandled error. Saving thread state as fatal_error...")
        persisted = agent.persist_fatal_error_thread(
            args.thread_id,
            reason=f"Unhandled exception: {exc}",
            status_reason="unhandled_exception",
            failing_stage="unknown",
            resumable=False,
        )
        if persisted:
            summary = persisted.get("final_summary", {})
            thread_id = summary.get("thread_id") or args.thread_id
            if thread_id:
                _info(f"Thread ID: {thread_id}")
            _info(f"Execution status: {summary.get('execution_status', 'fatal_error')}")
            if summary.get("status_reason"):
                _info(f"Status reason: {summary.get('status_reason')}")
            if summary.get("total_tokens_used") is not None:
                _info(f"Total tokens used: {summary.get('total_tokens_used', 0)}")
        raise
    finally:
        agent.close()


if __name__ == "__main__":
    main()
