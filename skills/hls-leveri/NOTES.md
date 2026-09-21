# hls-leveri — Reconstruction Notes

A standalone HLSClaw skill that verifies **C ↔ HLS-C functional consistency**.
It reconstructs the Dual-Tier Consistency Checking method from the paper
*Shift-Left High-Level Synthesis Verification via Knowledge-Augmented LLM Agent*
(repo `HLS-LeVeri`) from the paper text only — the authors' source was not
available.

## Why this skill exists
This directory preserves a standalone reconstruction of the paper's dual-tier
consistency method for research comparisons. It is not wired into HLSClaw's
default workflow. The default workflow uses AMD Vitis HLS C-sim through
`csim-verification` after both software and hardware rewrite.

## What was built
```
skills/hls-leveri/
  SKILL.md                     # skill spec (invocation, I/O markers, method)
  NOTES.md                     # this file
  scripts/
    static_consistency.py      # Static tier:  S_IO / S_CFG / S_DDG -> S_static
    dynamic_equiv.py           # Dynamic tier: differential I/O-trace equivalence
    run_leveri.py              # Orchestrator: two-tier protocol + status markers
  stubs/
    ap_int.h                   # ap_int<W> / ap_uint<W>  (native int + width mask)
    ap_fixed.h                 # ap_fixed<W,I> / ap_ufixed (double)
    hls_stream.h               # hls::stream<T>          (std::deque FIFO)
```

## Method

### Static tier (`static_consistency.py`) — paper Eq. 2–5
Driven by `clang -Xclang -ast-dump=json` (no libclang-python dependency). The
AST is filtered to functions **defined in the main source file** so the HLS
stub/system headers never leak into a score.

- **S_IO** (Eq. 2): `1 - Lev(S_c, S_h) / max(|S_c|,|S_h|)` over the linearized
  node-kind sequence of data-injection sub-trees (`VarDecl`, `InitListExpr`).
- **S_CFG** (Eq. 3): `w_ctrl·Sim_ctrl + w_nest·Sim_nest` — control-node
  distribution similarity + nesting-depth closeness.
- **S_DDG** (Eq. 4): Jaccard `|Ec ∩ Eh| / |Ec ∪ Eh|` over def-use edges.
- **S_static** (Eq. 5): `w1·S_IO + w2·S_CFG + w3·S_DDG`, default weights equal,
  `tau_static = 0.75`.

### Dynamic tier (`dynamic_equiv.py`) — paper Eq. 1
Concrete differential testing. `P_c` and `P_h` are each wrapped in their own
namespace (with `#include`s hoisted to global scope), compiled together against
`stubs/`, and run on **shared random stimuli**. Return values and array side
effects are compared; Dynamic Rate = `N_match / N_total`, PASS requires 1.0.

### Orchestration (`run_leveri.py`)
Emits standalone status markers:
`[leveri_status]=PASS|FAIL|ERROR|TIMEOUT`, `[leveri_reason]=...`,
`[leveri_case]=...`, plus `[leveri_s_*]` and `[leveri_dynamic_rate]`.
`ERROR` and `TIMEOUT` are inconclusive. The default HLSClaw workflow requires
an explicit `PASS` from `csim-verification` before it advances.

## Key design decision: what actually gates the result
The paper's static tier gates **testbench isomorphism** (`Φ(TB_c) ≅ Φ(TB_h)`,
Eq. 8) — that the two testbenches inject the *same stimuli*. It is **not** a
similarity check between the two *designs*, which are expected to differ because
HLS rewriting restructures code. Reproduced empirically: an equivalent rewrite
that merely adds an `ap_int` temporary scores `S_static ≈ 0.36`, well under
`tau_static`, yet is functionally identical.

Therefore, in HLSClaw's design-vs-design setting (stimulus generated identically
for both by construction):

- **Default**: the **dynamic tier is the equivalence oracle** and decides
  PASS/FAIL; static scores are advisory diagnostics that refine the failure
  case.
- **`--static-gate`**: restores the paper's strict short-circuit (require
  `S_static ≥ tau_static` before simulating). Use only when the two inputs are
  actual testbenches, not designs.

## Verified behavior
| Scenario | Result |
|----------|--------|
| Equivalent HLS-C (added `ap_int<32>` temp) | **PASS** (dynamic 100/100) |
| Buggy `-` instead of `+` | **FAIL** → `Case4_TargetDesignBug` (0/100) |
| `ap_int<8>` accumulator (width overflow) vs `int` | **FAIL** (37/100) — HLS width bug caught |
| `ap_int<32>` accumulator | **PASS** (100/100) |
| `--static-gate` on restructured pair | strict **FAIL** → `Case2_StaticInconsistency` |
| `hls::stream` interface | static parses; dynamic → `ERROR` (graceful, inconclusive) |

## Deviations from the paper / v1 limitations
- **KLEE** symbolic stimulus is substituted by concrete random differential
  testing; gcov coverage-driven augmentation and the **HLS Verification KG** are
  not reconstructed (v1 scope = core Dual-Tier).
- Dynamic tier is **functional C-level equivalence**, not RTL cycle-accurate
  co-simulation. `ap_int`/`ap_fixed` are modeled as native int/double, so it is
  not bit-accurate for fixed-point rounding (width overflow *is* emulated via
  masking on assignment).
- Dynamic harness drives **scalar** and **pointer/array** top-level parameters
  (integer scalars clamped to `[0, buf_size]`). `hls::stream`/struct interfaces
  parse statically but are not driven dynamically yet.
- Not integrated into the agent pipeline (`config.py`).

## Suggested next steps
1. gcov coverage tier (`C_total`, statement/branch/call) with `tau_cov = 0.95`.
2. Stream/struct driving in the dynamic harness.
3. HLS Verification KG (Coverage KG + Semantics KG) for retrieval-guided checks.
4. Wire in as Phase 6.5 once the skill is trusted standalone.
