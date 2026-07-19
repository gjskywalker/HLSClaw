---
name: csim-verification
description: Verify Original C and rewritten C/HLS-C designs with deterministic AMD Vitis HLS C simulation and differential output comparison.
---

# Vitis C-Sim Equivalence Verification

## When to use this skill

Use this skill after both software rewrite and hardware rewrite. It is the
single default verification path for both rewrite stages.

The check is differential testing, not a formal proof: Original C and Rewrite C
are simulated independently with identical deterministic stimuli, then return
values and every array side effect are compared.

## Prerequisites

- `vitis-run` is installed and available in `PATH`.
- Original and rewritten sources expose compatible top-function signatures.
- The target FPGA part and clock period are known.

## Inputs

- Original C/C++ source
- Rewritten C/C++ or HLS-C source
- Top function name
- FPGA part and clock period

## Output

- `[csim_status]=PASS|FAIL|ERROR|TIMEOUT`
- `[csim_reason]=...`
- `csim_equiv_report.json`
- Separate Original and Rewrite C-sim logs and deterministic traces

Only `PASS` establishes equivalence. `FAIL`, `ERROR`, and `TIMEOUT` must not be
reported as equivalent and should feed diagnostics back to the rewrite loop.

## Command

```bash
python <Skill_Script_absolute_Path>/run_csim_equiv.py \
  <rewrite_source> <original_source> <top_function> \
  --part <fpga_part> --clock-period <period> \
  --work-dir <verification_dir> --report <report.json>
```

## Comparison policy

- Integer returns and arrays are compared exactly.
- Floating-point values use configurable absolute and relative tolerances.
- Fixed-size array dimensions are recovered from the source signature and
  integer `#define` expressions.
- Pointer-only parameters use `--buf-size` as their storage extent.
- The same seed, trial count, initialization order, and values are used for
  Original and Rewrite simulations.

## Failure handling

- Signature mismatch or unsupported parameter type: `ERROR`.
- Vitis compilation/C-sim failure: `ERROR` with log diagnostics.
- Runtime limit exceeded: `TIMEOUT`.
- Any output mismatch: `FAIL` with the first mismatching trace key and values.
