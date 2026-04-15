---
name: pragma-dse
description: Search pragma configurations and evaluate QoR with Vitis HLS. Use after fuzzing passes and constraints are defined.
---
# Pragma DSE + Vitis HLS

## When to use this skill
Use this skill after `pragma-tuning` to evaluate pragma candidates with real Vitis HLS synthesis and select the best QoR under target constraints.

## Prerequisites
- Fuzzing test passed.
- Target constraints are defined.

## Inputs
- Hardware rewrite HLS source
- Pragma candidate file from `pragma-tuning`, generated from that hardware rewrite result
- Target constraints (frequency/resource/latency)

## Outputs
- Best pragma combination
- QoR report

## Steps
1. Load pragma candidates from `pragma-tuning`, using the same hardware rewrite source as the DSE baseline.
2. Evaluate the hardware rewrite baseline first, then evaluate single-site candidates.
3. Optionally expand a small number of multi-site combinations from the single-site winners.
4. Rank and select the best solution.

## Commands
1. Run pragma DSE with Vitis HLS evaluation.
```bash
python <Skill_Script_absolute_Path>/run_pragma_dse.py --code <Current_Code_Path> --top-func <Top_Function_Name> --part <FPGA_Board> --target-freq-mhz <Target_Frequency_MHz> --goal <Goal> --candidates <Run_Directory>/pragma_candidates.json --work-dir <Run_Directory>/pragma_dse_runs --output <Run_Directory>/pragma_dse_report.json --jobs <Candidate_Parallelism>
```

Progressive search example:
```bash
python <Skill_Script_absolute_Path>/run_pragma_dse.py --code <Current_Code_Path> --top-func <Top_Function_Name> --part <FPGA_Board> --target-freq-mhz <Target_Frequency_MHz> --goal <Goal> --candidates <Run_Directory>/pragma_candidates.json --work-dir <Run_Directory>/pragma_dse_runs --output <Run_Directory>/pragma_dse_report.json --search-strategy progressive --max-combos 4 --top-per-site 1 --beam-width 2 --jobs 2
```

## Notes
- `run_pragma_dse.py` internally reuses the profiling skill implementation for:
  - TCL generation
  - `vitis-run --mode hls --tcl` invocation
  - `csynth.xml` parsing
- If `<Run_Directory>/pragma_candidates.json` is missing, the script can auto-generate a small fallback search space by tuning parameters of pragmas already present in the baseline code.
- When available, prefer candidates generated from the hardware rewrite tuning plan instead of broad heuristic expansion.
- Candidate generation should modify existing pragma parameters rather than insert brand new pragma sites.
- `progressive` is the default search strategy. It evaluates the baseline first, keeps only single-site winners that beat baseline, and then explores a small beam of multi-site combinations.
- `--jobs` controls candidate parallelism. The default is `4`.
- Since QoR is measured through real Vitis HLS synthesis, this skill is slower than heuristic estimation and may need a longer command timeout.

## Failure handling
- Evaluation failure: record candidates and fall back to a degraded strategy.

## Logs & metrics
- Candidate count
- Top-K QoR metrics
- Search time

## Output Contract
Return one `<json>` block with:
- `skill`
- `evaluator`
- `candidate_count`
- `evaluated_count`
- `best_candidate_id`
- `top_candidates`
