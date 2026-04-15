---
name: kg-rag
description: Build and query a PDF-backed HLS RAG library via RAG-Anything, then return optimization strategies grounded in retrieved context.
---

# KG-RAG (Vitis Guide Retrieval)

## When to use this skill
Use this skill when you need retrieval-grounded guidance from the Vitis HLS user guide, or another reference PDF, before rewrite/optimization.

## Inputs
- Current design summary and bottlenecks from scratchpad
- Query focus (timing, II, latency, memory, dataflow, pragma tradeoffs)

## Outputs
- RAG library under `HLSClaw/kg-lib`
- Retrieved evidence and actionable strategy summary

## Commands
1. Build or refresh the default Vitis guide RAG library:
```bash
python <Skill_Script_absolute_Path>/build_vitis_rag.py
```
For quick validation under rate/latency limits:
```bash
python <Skill_Script_absolute_Path>/build_vitis_rag.py --prefer-cache --max-blocks 300
```
To build a library for an arbitrary PDF, provide `--pdf`. Custom PDFs are stored under `HLSClaw/kg-lib/<sanitized-pdf-stem>` unless `--library-name` is provided:
```bash
python <Skill_Script_absolute_Path>/build_vitis_rag.py --pdf /abs/path/to/reference.pdf
python <Skill_Script_absolute_Path>/build_vitis_rag.py --pdf /abs/path/to/reference.pdf --library-name my-hls-guide
```
2. Query the built library for targeted strategy retrieval:
```bash
python <Skill_Script_absolute_Path>/query_vitis_rag.py --question "Given FPGA target/clock and current bottlenecks, suggest concrete HLS optimization strategies, relevant pragmas, and tradeoffs."
```
For a custom library:
```bash
python <Skill_Script_absolute_Path>/query_vitis_rag.py --pdf /abs/path/to/reference.pdf --question "..."
python <Skill_Script_absolute_Path>/query_vitis_rag.py --library-name my-hls-guide --question "..."
```

## Steps
1. Reuse the existing library when its manifest matches the target PDF and the required artifacts are present.
2. Form exactly one targeted question from the current code shape, target constraints, and raw Vitis diagnostics when available; avoid generic tutorial-style questions.
3. Run query command and collect retrieved recommendations.
4. Summarize into `<analysis>` with prioritized actions and risks.

## Failure handling
- Missing API key: ask to set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) in env or `agentskl/.llm_env`.
- Build failure: report parser/dependency error and avoid fabricating retrieval results.

## Optional references
- Query templates: `references/query_templates.md`
