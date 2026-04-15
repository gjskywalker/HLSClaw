# HLSClaw

HLSClaw is a skill-based HLS optimization workflow for C/C++ kernels.  The
repository includes the orchestration code, reusable optimization/validation
skills, and two small motivation examples under `motivations/`.

## Repository Layout

- `run.py`: main command-line entry point for the end-to-end optimization flow.
- `agent.py`, `langgraph_agent.py`, `config.py`: agent runtime, state machine, and
  provider configuration.
- `skills/`: modular workflow stages for profiling, software rewrite,
  CBMC-based equivalence checking, knowledge retrieval, pragma tuning, and
  pragma DSE.
- `motivations/atax/` and `motivations/conv2d/`: example kernels and Vitis HLS
  TCL scripts.

## Requirements

The full workflow expects:

- Python 3.10 or newer.
- Python packages: `requests`, `langgraph`, and optionally `google-genai` if
  using the Gemini provider.
- Vitis HLS in `PATH` for profiling and pragma DSE stages.
- CBMC in `PATH` for the software equivalence checking stage.
- An API key for one supported LLM provider, supplied through environment
  variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
  `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, or `CUSTOM_API_KEY`.

Minimal Python setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install requests langgraph
```

For Gemini:

```bash
python -m pip install google-genai
```

## Quick Sanity Checks

Check that the CLI loads:

```bash
python run.py --help
```

Run one included Vitis HLS example directly:

```bash
cd motivations/atax
vitis_hls -f naive.tcl
```

The TCL scripts use paths relative to their own directories, so they can be run
after moving or unpacking the repository.

## Running the End-to-End Workflow

Set an LLM provider key first.  For example, define `OPENAI_API_KEY` in your
shell environment before launching the runner.

Then launch a run:

```bash
python run.py \
  --provider openai \
  --input-mode plain_c \
  --code motivations/atax/naive.cc \
  --req motivations/atax \
  --top dut
```

For HLS-native input code, use:

```bash
python run.py \
  --provider openai \
  --input-mode hls_native \
  --code motivations/atax/hls_rewrite.cc \
  --req motivations/atax \
  --top dut
```

Run outputs, checkpoints, logs, generated candidates, and HLS reports are written
under `runs/` by default.  This directory is intentionally ignored by git.

## Useful Options

- `--thread-id <id>`: choose a stable checkpoint thread ID.
- `--resume --thread-id <id>`: resume a saved workflow.
- `--list-threads`: list saved workflow threads.
- `--pragma-dse-jobs <n>`: set pragma-DSE parallelism.
- `--pragma-dse-candidate-timeout-sec <seconds>`: set per-candidate Vitis HLS
  timeout; use `0` to disable the timeout.