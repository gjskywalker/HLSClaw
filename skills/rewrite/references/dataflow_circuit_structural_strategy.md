# Dataflow Circuit Structural Strategy

Use this reference when the next hardware rewrite should change the **circuit structure**
rather than only adding or retuning pragmas.

This is a source-level optimization family. The goal is to expose a better dataflow graph
for later pragma/interface/storage/op/dataflow DSE, not to replace that DSE.

## When To Use This Strategy

- The current code is still array/loop dominated and has not already been fully converted into a banked dataflow circuit.
- `DATAFLOW` exists but downstream pragma DSE is `baseline_tied` or only marginally improves QoR.
- The bottleneck is stage interaction, local memory bandwidth, or reduction structure, not just a single pragma value.
- The design contains regular tensor/loop structure with clean divisors for tiling and banking.
- The code has natural producer/consumer boundaries such as load -> transform -> reduction -> store.

## Structural Candidate Families

Treat these as mutually competing hypotheses. Pick one primary structural hypothesis per round.

### 1. Stage Decomposition
Use when the code still has large monolithic loops or array-to-array glue code.

- Split load/transpose/broadcast/compute/reduction/store into explicit stages.
- Make stage boundaries visible before adding more task-level pragmas.
- Prefer this before aggressive banking if the current graph shape is unclear.

### 2. Banked Stream Introduction
Use when scalar streams or array traffic serialize otherwise parallel work.

- Convert scalar channels to banked stream arrays such as `stream[Tm][Tn]`.
- Retile loops so the bank dimensions correspond to real tile factors.
- Partition local arrays on the exact dimensions that feed those banks.

### 3. Schedule-Aware Bank Factorization
Use when the design is already banked but the bank geometry is weak.

- Keep the same stage graph but change factor shapes, such as `[8][8]` vs `[2][32]`.
- Rebalance bank factors across producer and consumer stages.
- Update `ARRAY_PARTITION/RESHAPE` to match the new bank geometry.

### 4. Topology Rebalance
Use when the graph is already strongly dataflow-oriented and the main issue is mismatch.

- Reassign factorization per stage.
- Reduce redundant intermediate storage.
- Rebalance broadcast, transpose, and reduction stages to fit the actual traffic pattern.

## Core Rewrite Rules

- Keep one top-level `#pragma HLS DATAFLOW` only after stage boundaries are explicit and legal.
- Every compute stage should have a single clear role: load, transpose, broadcast, GEMM-like compute, reduction, activation, or store.
- Only introduce banked streams when you can also identify the matching local array partitioning and loop tile factors.
- For GEMM-like or reduction nodes, use the init/accumulate/flush structure:
  - initialize on the first reduction step
  - accumulate in local storage
  - flush only on the final reduction step
- Broadcast should be explicit:
  - one read, multiple writes
- Transpose/reshape should be explicit:
  - do not hide them inside a large fused node if they determine bank layout
- Prefer local staging over direct multi-process reads from external memory.
- For loop-carried floating-point reductions inside a stage, do not assume `PIPELINE II=1` is viable. Start from a conservative II near the recurrence latency and only tighten it after timing remains feasible.

## What To Change Before More Pragmas

If the current design is `baseline_tied` after pragma DSE, prefer changing:

- stage boundaries
- local staging buffers
- stream banking shape
- loop tile factors
- reduction structure
- broadcast/transpose placement

Do not respond only by adding more `UNROLL`, `PIPELINE`, or `STREAM depth` pragmas to the same weak graph.

## What To Avoid

- Adding `DATAFLOW` when stages still directly contend on the same external interface.
- Banking streams without matching array partitioning.
- Choosing bank factors that are not divisors of the real iteration/domain sizes.
- Creating multi-stage graphs where producer/consumer token counts obviously do not match.
- Applying an aggressive structural dataflow rewrite and a highly aggressive pragma hypothesis in the same round.
- Forcing `II=1` on a dependent floating-point accumulation stage just because the surrounding graph is dataflow-oriented.

## Expected Output Of This Strategy

When this strategy is chosen, the rewrite analysis should explain:

- the chosen structural candidate family
- the intended stage graph
- the proposed tile/bank factors
- which local arrays must be partitioned or reshaped
- why this should expose stronger later DSE knobs
