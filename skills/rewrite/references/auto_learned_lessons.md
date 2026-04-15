# Auto-Learned Rewrite Lessons

This file is self-updated from completed HLSClaw runs.
It keeps reusable rewrite lessons compact by merging redundant content over time.

## Summary

- Prefer open_component-style project creation for Vitis-compatible flows, replace deprecated commands, and verify that synthesis reports and expected artifact...
- Verify that the expected HLS artifacts were produced before treating a run outcome as a design lesson.
- Stage off-chip data into local storage, partition or bank arrays to supply the intended unroll or pipeline parallelism, and assign independent traffic to sep...
- Apply loop directives at the structurally controlling loop level, especially when flattening is enabled; do not PIPELINE fully unrolled loops; and avoid part...
- When FP operator latency or load-to-store dependency chains exceed the target clock or II, relax unrealistic timing goals and insert explicit staging between...
- Begin with portable C/C++, explicit loop bounds, deterministic indexing, standard types, and no HLS pragmas. Prove all array accesses remain in bounds across...

## 1. Software Rewrite Guardrails

### 1.1 Establish a Standard-Compliant, Bounds-Proven Baseline Before Pragmas

- Rule: Begin with portable C/C++, explicit loop bounds, deterministic indexing, standard types, and no HLS pragmas. Prove all array accesses remain in bounds across edge cases before applying hardware directives or DSE.
- Why: A clean software baseline isolates functional bugs from hardware optimization effects, improves fuzzing and validation reliability, and avoids undefined memory behavior that later appears as scheduling noise or pragma rejection.
- Typical symptom:
  `Validation failures, candidate rejection before tuning, compiler auto-corrections, or persistent array index and bounds warnings such as HLS 214-167.`
- Seen in:
  `FeedForward`, `MultiHeadSelfAttention`, `atax`, `bnn`, `crs`, `getTanh`, `matrixmult`, `vecNormTrans`


## 2. Hardware Rewrite Guardrails

### 2.1 Build Canonical DATAFLOW Regions with Clean Interface Boundaries

- Rule: For DATAFLOW, separate load, compute, and store into sub-functions or equivalent process boundaries; stage external memory through local buffers or streams; do not let a process with a predecessor read inputs directly from the caller; and prune, drive, or constrain all generated interface and handshake signals consistently with actual access direction.
- Why: Non-canonical dataflow boundaries introduce hidden dependencies, reduce overlap, and can leave overgenerated ports dangling. Clean stage boundaries expose real task parallelism and prevent interface artifacts from masking bottlenecks.
- Typical symptom:
  `Non-canonical DATAFLOW diagnostics, infeasible scheduling, baseline-like performance despite added DATAFLOW, or dangling interface warnings such as HLS 200-1449, HLS 214-114, HLS 200-471, and RTGEN 206-101.`
- Seen in:
  `FeedForward`, `MultiHeadSelfAttention`, `atax`, `bnn`, `getTanh`, `matrixmult`, `substring`, `vecNormTrans`


## 3. Memory and AXI Bottlenecks

### 3.1 Match Memory Architecture to Parallel Access Demand

- Rule: Stage off-chip data into local storage, partition or bank arrays to supply the intended unroll or pipeline parallelism, and assign independent traffic to separate m_axi bundles instead of sharing a single global memory interface across parallel producers or consumers.
- Why: II is often limited by memory ports rather than compute. Without enough banks or bundles, the scheduler serializes accesses, dual-port RAM limits dominate, and alias assumptions can block parallelism even when compute is available.
- Typical symptom:
  `Port contention, serialization, false dependency reports, or II floors accompanied by warnings such as HLS 200-448, 200-880, 200-887, 200-871, and ANALYSIS 214-52.`
- Seen in:
  `FeedForward`, `atax`, `bnn`, `cnn`, `crs`, `getTanh`, `stencil`, `substring`, `vecNormTrans`

### 3.2 Avoid Overgenerated or Partially Unused AXI Masters

- Rule: Only expose m_axi interfaces whose read/write channels are truly needed by the kernel, and explicitly configure burst-related behavior for pipelined memory loops so address channels are actively driven rather than inferred but unused.
- Why: Unused AXI channels usually indicate unnecessary interface complexity or mismatched access architecture. They can obscure integration quality and signal that the memory structure does not match the kernel's real traffic pattern.
- Typical symptom:
  `RTGEN warnings that AR/AW and related AXI signals are dangling or forced to zero, especially RTGEN 206-101 on read-address or write-address outputs.`
- Seen in:
  `The load_data pipeline produced repeated RTGEN 206-101 warnings on m_axi_gmem3 read-address outputs being left dangli...`, `bnn`, `dut_Pipeline_STAGE_LOAD_DATA`


## 4. Pragma Placement and Combination Rules

### 4.1 Respect Loop-Pragma Hierarchy and Avoid Conflicting PIPELINE/UNROLL Combinations

- Rule: Apply loop directives at the structurally controlling loop level, especially when flattening is enabled; do not PIPELINE fully unrolled loops; and avoid partial UNROLL inside PIPELINE regions because tools commonly legalize this into complete unroll or otherwise distort intent.
- Why: Directive legality depends on loop hierarchy after transformations such as flattening. Redundant or conflicting directives waste resources, trigger pragma rewriting by the tool, and often create area explosions without throughput gain.
- Typical symptom:
  `Directive ignored or rewritten messages, unexpected complete unrolling, removed pipelines, or legality warnings such as HLS 214-275, HLS 214-189, HLS 200-6969, HLS 200-7035, and HLS 200-7036.`
- Seen in:
  `atax`, `bnn`, `cnn`, `crs`, `gemm`, `getTanh`, `stencil`, `substring`, `vecNormTrans`

### 4.2 Use Function Boundaries for DATAFLOW Tasks Instead of Inline Mixed Logic

- Rule: Within DATAFLOW regions, package each major stage as a dedicated sub-function or equivalent isolated task rather than mixing unrelated inline statements and memory accesses in one region.
- Why: Clear task boundaries help the tool form concurrent processes and channels. Inline mixed logic inside a dataflow region tends to collapse parallelism or create non-canonical regions that schedule poorly.
- Typical symptom:
  `DATAFLOW accepted syntactically but with weak overlap, or explicit non-canonical dataflow diagnostics such as HLS 214-114.`
- Seen in:
  `atax`, `bnn`, `getTanh`, `substring`, `vecNormTrans`


## 5. Timing and Arithmetic Lessons

### 5.1 Fix Timing Structurally for FP Recurrences and Cross-Array Dependencies

- Rule: When FP operator latency or load-to-store dependency chains exceed the target clock or II, relax unrealistic timing goals and insert explicit staging between dependent operations, especially across memory and FP arithmetic boundaries.
- Why: Tight targets cannot overcome physical latency of FP pipelines or inter-array dependence chains. Structural staging is more effective than further pragma search when the critical path is intrinsic to the arithmetic or memory dependency.
- Typical symptom:
  `Estimated clock exceeds target, persistent recurrence-limited II, or warnings such as HLS 200-886, 200-887, 200-871, and 200-1016.`
- Seen in:
  `MultiHeadSelfAttention`, `atax`, `bnn`, `crs`, `dut_Pipeline_PHASE1`, `dut_Pipeline_VITIS_LOOP_53_4`, `getTanh`, `matrixmult`, `vecNormTrans`

### 5.2 Replace Serial Accumulations with Tree or Partial-Sum Reductions

- Rule: Do not keep long running accumulators on the critical path of pipelined loops. Use tree reductions, multiple partial sums, or shift-register accumulation structures, and if necessary reduce accumulator width only when numerically acceptable.
- Why: Serial accumulators create recurrence paths that scale poorly with throughput targets and frequently dominate timing even when memory is fixed. Balanced reductions shorten dependency depth and make pipelining feasible.
- Typical symptom:
  `Reduction variables dominate the critical path, II fails to reach target, or timing remains poor after basic pipelining and partitioning.`
- Seen in:
  `atax`, `bnn`, `crs`, `getTanh`, `matrixmult`, `vecNormTrans`


## 6. Pragma-Tuning and DSE Lessons

### 6.1 Constrain DSE to High-Impact, Pre-Validated Directive Choices

- Rule: Search only a small set of meaningful pragma parameters on kernels already known to synthesize, and pre-validate directive syntax, legality, and dependency annotations before launching exploration.
- Why: Broad pragma sweeps cannot solve structural timing or memory problems and often waste runs on illegal or equivalent candidates. Focused, validated DSE gives cleaner signal and avoids automation failure modes.
- Typical symptom:
  `Uniform baseline-tied outcomes, many failed candidates, syntax or dependency errors such as HLS 207-1391, or no successful exploration results.`
- Seen in:
  `FeedForward`, `atax`, `bnn`, `crs`


## 7. Tooling and Flow Compatibility

### 7.1 Use Component-Oriented Vitis Flows and Validate Artifacts Before Interpreting QoR

- Rule: Prefer open_component-style project creation for Vitis-compatible flows, replace deprecated commands, and verify that synthesis reports and expected artifacts were actually produced before drawing design conclusions from a run.
- Why: Batch-only or partially generated projects can synthesize yet be unsuitable for IDE or downstream tooling, while missing artifacts and deprecated commands create false lessons about design quality instead of flow correctness.
- Typical symptom:
  `IDE compatibility warnings, missing reports or files, malformed automation outputs, or deprecated-command warnings such as HLS 200-2182 and HLS 200-483.`
- Seen in:
  `FeedForward`, `MultiHeadSelfAttention`, `Pragma DSE Combo Projects`, `atax`, `bnn`, `cnn`, `crs`, `gemm`, `getTanh`, `gramSchmidt`, `matrixmult`, `stencil`, `vecNormTrans`

### 7.2 Separate tool failures from optimization conclusions

- Rule: Verify that the expected HLS artifacts were produced before treating a run outcome as a design lesson.
- Why: Missing reports or orchestration failures can look like optimization failures even when no valid HLS result exists.
- Typical symptom:
  `HTTP 402 for https://openrouter.ai/api/v1/chat/completions | body: {"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up to 16000 tokens, but can only afford 14129. To increase, visit https://openrou`
- Seen in:
  `atax`, `bnn`, `cnn`, `crs`, `gemm`, `getTanh`, `gramSchmidt`, `matrixmult`, `stencil`, `substring`, `vecNormTrans`, `veterbi`


## 8. High-Priority Rewrite Heuristics

### 8.1 Revert to a Compact Rewrite When Optimization Causes Complexity Explosion

- Rule: If aggressive unrolling, inlining, dataflow, or search causes excessive instruction growth, vanished pragma targets, timeouts, or no QoR separation between candidates, back out to a smaller structurally cleaner rewrite before trying more directives.
- Why: Complexity explosions are usually signs of poor code shape or over-aggressive parallelization, and additional pragmas compound the problem. A compact baseline restores observability and makes subsequent optimization decisions meaningful.
- Typical symptom:
  `Synthesis timeout, excessive transformed instruction counts, matched_target_count = 0, HLS 200-1995, or many candidates with identical poor outcomes.`
- Seen in:
  `FeedForward`, `MultiHeadSelfAttention`, `atax`, `bnn`, `crs`, `substring`, `vecNormTrans`
