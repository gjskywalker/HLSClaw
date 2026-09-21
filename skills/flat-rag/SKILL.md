---
name: flat-rag
description: Query the Vitis HLS guide with flat/vector retrieval for RAG ablation against KG-RAG.
---

# Flat RAG (Vitis Guide Retrieval)

## When to use this skill
Use this skill for ablation runs that need retrieval-grounded HLS guidance without graph/multi-hop retrieval.

This skill intentionally reuses the same Vitis guide library as `kg-rag`, but queries it in `naive` retrieval mode. This keeps the document corpus and parsing artifacts fixed while changing the retrieval strategy.

## Inputs
- Current design summary and bottlenecks from scratchpad
- Query focus (timing, II, latency, memory, dataflow, pragma tradeoffs)
- Shared source PDF: `/home/eeuser/Desktop/LLM4PragmaDSE/vitis-user-guide.pdf`

## Outputs
- Retrieved flat/vector evidence and actionable strategy summary
- Query result saved under the shared `HLSClaw/kg-lib` retrieval directory

## Commands
1. Build or refresh the shared Vitis guide RAG library:
```bash
python <Skill_Script_absolute_Path>/build_vitis_flat_rag.py
```
By default, this uses the same repository-level guide as KG-RAG: `/home/eeuser/Desktop/LLM4PragmaDSE/vitis-user-guide.pdf`.

For quick validation under rate/latency limits:
```bash
python <Skill_Script_absolute_Path>/build_vitis_flat_rag.py --prefer-cache --max-blocks 300
```

2. Query the built library with flat/vector retrieval:
```bash
python <Skill_Script_absolute_Path>/query_vitis_flat_rag.py --question "Given FPGA target/clock and current bottlenecks, suggest concrete HLS optimization strategies, relevant pragmas, and tradeoffs."
```

For a custom library:
```bash
python <Skill_Script_absolute_Path>/query_vitis_flat_rag.py --pdf /abs/path/to/reference.pdf --question "..."
python <Skill_Script_absolute_Path>/query_vitis_flat_rag.py --library-name my-hls-guide --question "..."
```

## Steps
1. Reuse the existing Vitis guide library when available; build it if needed.
2. Form exactly one targeted question from the current code shape, target constraints, and raw Vitis diagnostics when available.
3. Run the flat query command and collect retrieved recommendations.
4. Summarize into `<analysis>` with prioritized actions and risks.

## Failure handling
- Missing API key: ask to set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in env or `HLSClaw/.llm_env`.
- Build failure: report parser/dependency error and avoid fabricating retrieval results.
- Query failure: report the flat retrieval error and leave retrieval output unavailable.
