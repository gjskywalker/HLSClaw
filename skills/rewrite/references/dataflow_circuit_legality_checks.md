# Dataflow Circuit Legality Checks

Use this checklist before proposing or emitting a structural dataflow rewrite.

## Interface Safety

- Do not let multiple DATAFLOW processes directly read the same top-level `m_axi` port or bundle.
- If multiple stages need the same external data, create a local staging/broadcast structure first.
- Do not treat top-level `m_axi` arrays as `ARRAY_PARTITION` tuning targets.

## Token Conservation

- For every rewritten edge, the producer and consumer must move the same total number of logical elements.
- If a scalar channel becomes `stream[Tm][Tn]`, verify the tiled loop nest covers the original iteration space exactly.
- Bank dimensions should correspond to real tile factors, not arbitrary constants.

## Producer/Consumer Compatibility

- Producer bank shape must match consumer bank shape or an explicit reshape/transpose stage must exist.
- Broadcast must be explicit when one producer fans out to multiple consumers.
- Reduction stages must preserve init/flush semantics after retiming or banking.

## Local Memory Alignment

- Local arrays used by banked stages must be partitioned or reshaped on the dimensions driven in parallel.
- Partition factors must align with the actual number of concurrent accesses.
- If an array remains single-port while the rewrite increases per-cycle accesses, the new structure is not legal enough for good QoR.

## Loop/Tile Consistency

- Tile factors should divide the real problem dimensions, or the rewrite must include correct boundary handling.
- Bank factors should be chosen from clean divisors when possible.
- Do not flatten away all useful loop structure if later DSE still needs tunable loops.

## Resource Sanity

- Do not combine maximum banking, maximum unroll, and deep multi-stage DATAFLOW in one speculative rewrite unless the bottleneck analysis clearly supports it.
- If the prior round hit timeout or timing collapse, reduce structural aggressiveness in the next rewrite.

## Good Failure Response

If the legality story is weak, prefer:

- a smaller number of stages
- smaller bank factors
- explicit local buffers
- conservative DATAFLOW with one clear producer/consumer chain

over an elaborate but fragile stream graph.
