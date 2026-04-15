---
name: fuzzing
description: Validate optimized HLS code via CBMC-based equivalence checking to ensure functional correctness. Use after code rewrite.
---
# Software Equivalence Check (CBMC)

## When to use this skill
Use this skill to verify functional equivalence between the original software baseline and a plain C/C++ rewritten candidate.
Use it specifically after `software rewrite`.
Do not use it for hardware-oriented HLS variants that contain pragmas, streams, `ap_int`, AXI interfaces, or other HLS-only constructs.
This skill is CBMC-only. There is no LLVM/libFuzzer fallback path.

## Prerequisites
- `cbmc` is installed and available in `PATH`
- Optimized source version exists
- Both original and optimized versions are plain C/C++ and expose the same top function signature

## Inputs
- Optimized plain C/C++ source
- Original plain C/C++ source
- Top function name

## Outputs
- PASS/FAIL/TIMEOUT/ABORTED
- Explicit status markers in stdout: `[cbmc_status]=PASS|FAIL|TIMEOUT|ABORTED` and optional `[cbmc_reason]=...`
- Counterexample trace when CBMC reports `VERIFICATION FAILED`
- Harness generation errors when the candidate source is malformed

## Steps
1. Generate the CBMC harness with `gen_cbmc_harness.py`.
```bash
python <Skill_Script_absolute_Path>/gen_cbmc_harness.py <optimized_code> <original_code> <function_name> [output_dir] [version]
```
Example:
```bash
python /abs/path/to/skills/fuzzing/scripts/gen_cbmc_harness.py atax_opt.cc atax.cc dut . v1
```
2. Run CBMC equivalence checking with `run_cbmc_equiv.py`.
```bash
python <Skill_Script_absolute_Path>/run_cbmc_equiv.py --run_dir <run_dir> --version <version> --timeout 180 --unwind 2048
```
Example:
```bash
python /abs/path/to/skills/fuzzing/scripts/run_cbmc_equiv.py --run_dir . --version v1 --timeout 180 --unwind 2048
```
3. Check the process exit status and tool output for explicit CBMC status markers: `PASS`, `FAIL`, or `TIMEOUT`.
4. If the candidate contains HLS-only constructs, skip this stage and route it to the HLS evaluation stages instead.

## Modeling Notes
- The harness applies lightweight assumptions only to obvious index-like inputs such as `addr`, `idx`, or `index`.
- Array side effects are compared in addition to return values.
- Pointer parameters are modeled as fixed-size arrays of 100 elements.
- Arithmetic UB checks are disabled by default in the runner to prioritize relational equivalence; `--strict_safety` can be enabled when needed.

## Failure handling
- Harness generation failure: return FAIL and record the malformed source or signature mismatch
- CBMC failure: record the counterexample trace
- Timeout: return TIMEOUT and treat it as inconclusive but acceptable for pipeline progression
- Internal abort / early termination without `VERIFICATION FAILED`: return ABORTED and treat it as inconclusive but acceptable for pipeline progression
- Only explicit FAIL should trigger another software rewrite iteration

## Logs & metrics
- Verification status
- Counterexample trace
- Runtime
- Unwind bound used
