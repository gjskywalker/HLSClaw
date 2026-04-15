---
name: rewrite
description: Apply selected strategies to rewrite HLS source code. Use when you need optimized variants after KG strategy selection.
---

# Rewrite (Two-Stage Code Optimization)

## When to use this skill
Use this skill to guide two rewrite stages:
- software rewrite: plain C/C++ only, intended for CBMC/g++ validation
- hardware rewrite: HLS-oriented rewrite after software validation; emit the baseline HLS pragma structure so later `pragma-tuning` / `pragma-dse` only tune parameters

If later hardware validation / pragma-tuning / pragma-dse reports structural HLS errors, revise the hardware rewrite instead of forcing more pragmas onto a broken base design.

## Prerequisites
- Strategy list is available.
- Reference rules directory is available: `agentskl/skills/rewrite/references/rewrite_rules.md`.
- For dataflow rewrite, reference rules directory is available: `agentskl/skills/rewrite/references/dataflow_circuit_patterns.md`.
- For structural dataflow rewrite, reference files `references/dataflow_circuit_structural_strategy.md` and `references/dataflow_circuit_legality_checks.md` are available.
- Auto-learned rewrite guardrails are available in `references/auto_learned_lessons.md`; load them when present.

## Inputs
- Original HLS source
- Profiling summary / bottleneck analysis
- Optimization strategy list

## Outputs
- Rewrite guidance for software-safe transformations
- Rewrite guidance for hardware-oriented transformations
- Change summary and rationale

## Steps
1. Load `rewrite_rules.md`.
2. If present, load `references/auto_learned_lessons.md`.
3. When considering structural dataflow rewrite, load `dataflow_circuit_structural_strategy.md` and `dataflow_circuit_legality_checks.md`.
4. Read the profiling summary and current code shape before proposing changes.
5. Split guidance into:
   - software-safe rewrites that remain plain C/C++ and validation-friendly
   - hardware-only rewrites that may introduce HLS constructs
6. For software rewrite, prefer one concrete candidate per attempt.
7. For hardware rewrite, emit one coherent baseline pragma structure plus 2-4 real tunable sites.
8. If the next move should be a structural dataflow circuit optimization, choose one primary structural hypothesis and explain the intended stage graph, bank factors, and local array changes before proposing more pragmas.
9. Prefer references, rationale, and change guidance in `<analysis>`.
10. Only emit `<optimized_code>` when the caller explicitly requests a concrete rewrite stage.

## Bottleneck-to-Strategy Mapping

- If the dominant bottleneck is external-memory contention, first change memory structure:
  stage top-level arrays locally, reduce repeated `m_axi` traffic, and align `ARRAY_PARTITION/RESHAPE` with the actual parallel access pattern before adding more `DATAFLOW` or aggressive `UNROLL`.
- If the dominant bottleneck is II/resource limitation, first change the local access structure:
  preserve tunable inner loops, increase local memory bandwidth, and make `PIPELINE` / `UNROLL` factors match the real number of concurrent accesses.
- If the dominant bottleneck is timing-critical arithmetic or reductions, first shorten the critical path:
  simplify reduction structure, reduce over-aggressive parallelism, and prefer conservative `PIPELINE II` targets before expanding task-level parallelism.
- Use `DATAFLOW` only when the kernel naturally decomposes into legal producer/consumer stages with isolated external-memory access.
  `DATAFLOW + STREAM + PIPELINE` is often useful, but only after memory interfaces are isolated cleanly.
- Treat `baseline_tied` pragma-DSE results as a signal that the current code shape exposes weak tuning knobs.
  In the next hardware rewrite, change code structure or memory staging instead of adding more pragmas of the same family.

## Hardware Rewrite Rules

- Keep HLS pragmas legal; do not place `INLINE` at file scope.
- Preserve tunable inner loops for later `PIPELINE` / `UNROLL` tuning.
- When an unrolled loop needs multiple values from the same external `m_axi` port, do not partition the top-level `m_axi` array. Stage the data locally first, then partition/reshape the local buffer with a factor aligned to the real parallel access count.
- If using `DATAFLOW`, do not let multiple DATAFLOW processes directly read the same external `m_axi` port/bundle without local staging.
- Later `pragma-tuning` should adjust only parameters of existing pragma sites such as `II`, `factor`, or `depth`, not invent new pragma locations.

## Failure handling
- Compile failure or syntax errors: roll back and record the failure reason.

## Logs & metrics
- Change size (lines/functions)
- Version IDs and rollback records
