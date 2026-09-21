---
name: hls-leveri
description: Verify C-to-HLS-C functional consistency via Dual-Tier (static structural + dynamic behavioral) checking. Use after hardware rewrite, before pragma tuning/DSE.
---
# C-to-HLS-C Consistency Check (HLS-LeVeri)

Reconstruction of the shift-left verification method from *Shift-Left High-Level
Synthesis Verification via Knowledge-Augmented LLM Agent* (repo `HLS-LeVeri`). It
checks that an HLS-oriented rewrite (HLS-C, `P_h`) is functionally consistent
with the original plain-C baseline (`P_c`). This is retained as a standalone
research implementation; HLSClaw's default workflow uses `csim-verification`.

## When to use this skill
Use it after a **hardware rewrite** (a variant that contains HLS-only constructs
such as `#pragma HLS`, `hls::stream`, `ap_int`/`ap_uint`/`ap_fixed`, or AXI
interfaces) to confirm the HLS-C still matches the original C before pragma
tuning and DSE when explicitly evaluating the dual-tier research method.

## Prerequisites
- `clang` (AST dump, static tier) and `g++` (dynamic tier) in `PATH`.
- Both sources expose the same top-function signature.
- Vitis HLS headers are **not** required: the bundled `stubs/` model
  `ap_int.h`, `ap_fixed.h`, and `hls_stream.h` at the C level so HLS-C parses
  and runs. Pass real HLS/user include dirs with `--include` when available.

## Inputs
- HLS-oriented C/C++ source (`P_h`, the "optimized" arg)
- Original plain C/C++ source (`P_c`, the "original" arg)
- Top function name

## Outputs
- `[leveri_status]=PASS|FAIL|ERROR|TIMEOUT` and `[leveri_reason]=...`
- `[leveri_case]=...` — failure diagnosis mapped to the paper's cases
  (`Case2_StaticInconsistency`, `Case4_TargetDesignBug`, `none`, `inconclusive`)
- Static scores: `[leveri_s_io]`, `[leveri_s_cfg]`, `[leveri_s_ddg]`,
  `[leveri_s_static]` (Eq. 2-5)
- Dynamic score: `[leveri_dynamic_rate]` (Eq. 1, `N_match/N_total`)

## Steps
Run the orchestrator; it applies the two-phase short-circuit protocol.
```bash
python <Skill_Script_absolute_Path>/run_leveri.py <hls_c_source> <original_c_source> <function_name> \
    [--tau_static 0.75] [--trials 200] [--buf-size 64] [--include <dir>]
```
Example:
```bash
python /abs/path/to/skills/hls-leveri/scripts/run_leveri.py atax_hls.cpp atax.c atax
```
The two tiers can also be run directly:
```bash
python scripts/static_consistency.py <hls_c> <original_c> --tau_static 0.75
python scripts/dynamic_equiv.py      <hls_c> <original_c> <function_name> --trials 200
```

## Method (Dual-Tier Consistency Checking)
1. **Static tier** (`static_consistency.py`): over the two clang ASTs, compute
   `S_IO` (Eq. 2, normalized Levenshtein on linearized data-injection
   sub-trees), `S_CFG` (Eq. 3, control-node distribution + nesting-depth match),
   and `S_DDG` (Eq. 4, Jaccard over def-use edges), then
   `S_static = w1*S_IO + w2*S_CFG + w3*S_DDG` (Eq. 5).
2. **Dynamic tier** (`dynamic_equiv.py`): wrap `P_c` and `P_h` in separate
   namespaces, compile them together against `stubs/`, run on shared random
   stimuli, and compare return values and array side effects. Require an
   all-agree Dynamic Rate = 1.0 (Eq. 1: forall x, `M_c(x) == M_h(x)`).

### What gates the result
In the paper, the static tier gates *testbench isomorphism* (`Phi(TB_c) ~=
Phi(TB_h)`), i.e. that the two testbenches inject the **same stimuli** — it is
not a similarity check between the two *designs*, which are meant to differ
because HLS rewriting restructures code. In HLSClaw the stimulus is generated
identically for both designs by construction, so:
- **Default**: the **dynamic tier is the equivalence oracle** and decides
  PASS/FAIL; static scores are reported as diagnostics that refine the failure
  case. A legitimately restructured but functionally identical rewrite (e.g. an
  added `ap_int` temporary) correctly PASSes even though `S_static < 1`.
- **`--static-gate`**: restores the paper's strict short-circuit (require
  `S_static >= tau_static`, default 0.75, before simulating). Use this only when
  the two inputs are actual testbenches rather than designs.

## Reconstruction notes and deviations from the paper
- KLEE symbolic stimulus is **substituted** by concrete random differential
  testing; coverage-driven (gcov) stimulus augmentation and the HLS Verification
  Knowledge Graph are **not** in this first version.
- The dynamic tier is **functional C-level equivalence**, not RTL cycle-accurate
  co-simulation; `ap_int`/`ap_fixed` are modeled as native int/double, so it is
  not bit-accurate for width-overflow or fixed-point rounding corner cases.
- Static weights `w1,w2,w3` and `tau_static` are configurable; defaults follow
  the paper's `tau_static = 0.75` with equal weights.
- The dynamic harness drives **scalar** and **pointer/array** top-level
  parameters (integer scalars are clamped to `[0, buf_size]` as index/size-like
  inputs). `hls::stream`/struct interfaces are parsed by the static tier but not
  yet driven by the dynamic tier; such tops return `ERROR` (inconclusive), which
  is acceptable for pipeline progression. Stream-driving is future work.

## Failure handling
- `Case2_StaticInconsistency` (static FAIL): the HLS-C structure diverges from
  the original C; realign the rewrite/testbench, do not proceed to simulation.
- `Case4_TargetDesignBug` (dynamic FAIL): the HLS-C changed behavior; treat as a
  real functional regression and trigger another hardware rewrite.
- `ERROR`/`TIMEOUT`: inconclusive for this standalone research implementation.
  The default HLSClaw workflow does not advance unless `csim-verification`
  returns an explicit `PASS`.
