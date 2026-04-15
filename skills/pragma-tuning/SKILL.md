---
name: pragma-tuning
description: Generate HLS pragma candidates for a selected optimized design version. Use before pragma-dse to prepare the search set.
---

# Pragma Tuning

## When to use this skill
Use this skill after hardware rewrite to generate a pragma candidate set for the hardware-oriented code version. This skill is lightweight and does not run Vitis HLS.

## Inputs
- Hardware rewrite code path
- Optional tuning plan JSON emitted by hardware rewrite

## Commands
Generate parameter-tuning candidate plans from existing pragmas.
```bash
python <Skill_Script_absolute_Path>/gen_candidates.py --code <Current_Code_Path> --output <Run_Directory>/pragma_candidates.json [--plan <Run_Directory>/pragma_tuning_plan.json]
```

## Notes
- This skill prepares candidate files for `pragma-dse`.
- When a hardware-rewrite tuning plan is available, use it to prioritize only the most important pragma sites.
- Candidate generation should treat the hardware rewrite output as the baseline and only adjust parameters of pragmas that already exist in that code.
- Candidate generation should stay legality-aware; for example, unroll factors should not exceed the inferred loop tripcount when it can be determined from the source.
- Example tunable parameters: `PIPELINE II`, `UNROLL factor`, `ARRAY_PARTITION/ARRAY_RESHAPE factor`, `STREAM depth`.
- Final QoR decisions should be made only by `pragma-dse` with real Vitis HLS synthesis.

## Output Contract
Return one `<json>` block with:
- `skill`
- `candidate_count`
- `candidates`
