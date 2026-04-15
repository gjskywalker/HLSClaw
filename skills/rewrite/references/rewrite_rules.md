## C/C++ Optimization Rewrite Rules (General)

### 1) Loop Optimization
- **Loop-invariant code motion**: Move computations that do not depend on the loop index outside the loop.
- **Strength reduction**: Replace expensive operations inside loops (e.g., multiplication/division) with cheaper ones (e.g., addition/bit shifts) when safe.
- **Loop unrolling**: Unroll small fixed-iteration loops to reduce branch overhead and enable instruction-level parallelism.
- **Loop fusion**: Merge adjacent loops with the same iteration space to improve cache locality.
- **Loop fission (distribution)**: Split loops to isolate expensive operations or enable vectorization.
- **Loop interchange**: Reorder nested loops to improve memory access patterns.
- **Loop peeling**: Extract first/last few iterations to simplify bounds and enable vectorization.
- **Loop tiling (blocking)**: Break loops into blocks to improve cache reuse.
- **Remove empty loops**: Eliminate loops with no observable side effects.

### 2) Memory Access & Data Layout
- **Replace AoS with SoA**: Convert Array-of-Structs to Struct-of-Arrays to improve SIMD/vectorization.
- **Align memory accesses**: Ensure data is aligned (e.g., using `alignas` or aligned allocators) for SIMD and cache efficiency.
- **Use restrict qualifiers**: Add `__restrict` (or `restrict` in C) when pointers do not alias.
- **Prefetch where beneficial**: Insert prefetch hints for predictable access patterns.
- **Avoid redundant loads/stores**: Cache repeated loads in a local variable.
- **Use `memcpy`/`memmove` for bulk copy**: Replace manual loops with optimized library calls when safe.

### 3) Branch & Control Flow Simplification
- **Branch elimination**: Replace conditional branches with arithmetic or conditional moves when beneficial.
- **Short-circuit constant conditions**: Simplify if/else when conditions are constant or can be proven.
- **Flatten nested conditionals**: Combine nested `if` statements into a single condition when possible.
- **Replace switch with lookup table**: Use arrays for dense switch cases.
- **Early exit**: Move invariant error checks outside loops; return early when conditions are met.

### 4) Function-Level Optimizations
- **Inline small functions**: Inline tiny functions called frequently to remove call overhead.
- **Mark hot/cold functions**: Use attributes (e.g., `__attribute__((hot))`) where supported.
- **Remove unused functions**: Delete dead code or unused static functions.
- **Const-correctness**: Mark parameters and methods `const` to enable better optimization.

### 5) Algebraic Simplification
- **Constant folding**: Precompute constant expressions at compile-time.
- **Common subexpression elimination (CSE)**: Reuse repeated expressions.
- **Reassociate operations**: Reorder associative operations to improve vectorization or reduce stalls (respect floating-point rules).
- **Replace `pow(x,2)`**: Use `x * x` for integer or safe floating-point cases.
- **Use `x << n` for `x * (2^n)`**: For integer arithmetic where overflow behavior is acceptable.

### 6) Vectorization-Friendly Patterns
- **Use contiguous arrays**: Replace pointer chasing with flat arrays.
- **Eliminate loop-carried dependencies**: Refactor to remove dependencies that block vectorization.
- **Convert branches to masks**: Use boolean masks or ternaries for vectorization.
- **Use compiler intrinsics when needed**: e.g., `<immintrin.h>` for SIMD.

### 7) Resource & Lifetime Optimization
- **Reuse buffers**: Avoid frequent allocations in loops; allocate once and reuse.
- **Prefer stack allocation for small objects**: Avoid heap overhead.
- **Move invariant allocations outside loops**.
- **Use move semantics** (C++): Replace copies with moves when safe.

### 8) I/O and Logging
- **Batch I/O**: Replace frequent small I/O calls with buffered I/O.
- **Disable debug logging in hot paths**: Guard logging with compile-time flags.

### 9) Type & Precision Choices
- **Use narrower types when safe**: `int32_t` instead of `int64_t` for better cache usage.
- **Prefer integer math when exact**: Replace floating operations with integer operations if precision allows.
- **Use `constexpr` for compile-time constants**.

### 10) Parallelism Hints (Compiler-Directed)
- **Add `#pragma omp parallel for`**: When iterations are independent.
- **Add `#pragma unroll`**: When loop count is small or known.
- **Add `#pragma ivdep`/`#pragma simd`**: To help auto-vectorization when safe.

---

## C/C++ Optimization Rewrite Rules (HLS-Oriented)

### 1) Pipeline & Parallelism

#### Pipeline Critical Loops
Add `#pragma HLS pipeline` to throughput-bound loops to enable overlapping iterations.

```c
// Before: Sequential execution
for (int i = 0; i < N; i++) {
    result[i] = compute(data[i]);  // N * latency cycles
}

// After: Pipelined execution
for (int i = 0; i < N; i++) {
    #pragma HLS pipeline II=1
    result[i] = compute(data[i]);  // N + latency cycles
}
```

#### Unroll Inner Loops
For small constant trip counts, unrolling creates parallel hardware.

```c
// Before: Sequential dot product
double sum = 0;
for (int j = 0; j < 5; j++) {  // Small fixed size
    sum += A[i][j] * x[j];
}

// After: Fully parallel (5 multipliers in hardware)
double sum = 0;
#pragma HLS unroll
for (int j = 0; j < 5; j++) {
    sum += A[i][j] * x[j];
}
```

#### Partial Unroll for Resource Trade-off
```c
// Unroll with factor for balanced parallelism
for (int i = 0; i < 100; i++) {
    #pragma HLS unroll factor=4
    // Creates 4 parallel copies, iterates 25 times
    process(data[i]);
}
```

#### Flatten Nested Loops
Combine perfectly nested loops for better pipelining.

```c
// Before: Nested loops with overhead
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

// After: Single loop (M*N iterations)
#pragma HLS loop_flatten
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        #pragma HLS pipeline II=1
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### 2) Memory & Interface Optimization

#### Array Partitioning for Parallel Access
Enable simultaneous access to array elements.

```c
// Before: Single-port memory access
void dut(int A[100], int x[5], int y[5]) {
    for (int i = 0; i < 5; i++) {
        y[i] = A[i] * x[i];  // Sequential memory access
    }
}

// After: Partitioned arrays for parallel access
void dut(int A[100], int x[5], int y[5]) {
    #pragma HLS array_partition variable=x complete
    #pragma HLS array_partition variable=y complete
    
    for (int i = 0; i < 5; i++) {
        #pragma HLS unroll
        y[i] = A[i] * x[i];  // All 5 elements accessed in parallel
    }
}
```

#### Partitioning Strategies
```c
// Complete partitioning (registers)
#pragma HLS array_partition variable=A complete

// Block partitioning (multiple memory banks)
#pragma HLS array_partition variable=A block factor=4

// Cyclic partitioning (interleaved)
#pragma HLS array_partition variable=A cyclic factor=4
```

#### Local BRAM Buffers
Copy data to local buffers for repeated access.

```c
// Before: Accessing global memory repeatedly
void filter(int data[1024], int result[1024]) {
    for (int i = 1; i < 1023; i++) {
        // Each element accessed 3 times from external memory
        result[i] = (data[i-1] + data[i] + data[i+1]) / 3;
    }
}

// After: Local buffer in BRAM
void filter(int data[1024], int result[1024]) {
    int local_buf[1024];
    #pragma HLS array_partition variable=local_buf block factor=4
    
    // Copy to local buffer (burst transfer)
    for (int i = 0; i < 1024; i++) {
        #pragma HLS pipeline II=1
        local_buf[i] = data[i];
    }
    
    // Compute with fast local access
    for (int i = 1; i < 1023; i++) {
        #pragma HLS pipeline II=1
        result[i] = (local_buf[i-1] + local_buf[i] + local_buf[i+1]) / 3;
    }
}
```

#### Streaming for Dataflow
Replace arrays with streams for producer-consumer patterns.

```c
#include "hls_stream.h"

void dataflow_example(hls::stream<int>& in, hls::stream<int>& out) {
    #pragma HLS dataflow
    
    hls::stream<int> tmp1("tmp1");
    hls::stream<int> tmp2("tmp2");
    #pragma HLS stream variable=tmp1 depth=16
    #pragma HLS stream variable=tmp2 depth=16
    
    stage1(in, tmp1);    // Producer
    stage2(tmp1, tmp2);  // Consumer + Producer
    stage3(tmp2, out);   // Consumer
}
```

### 3) Control & Dataflow

#### Dataflow Optimization
Enable task-level parallelism for producer-consumer patterns.

```c
// Before: Sequential execution
void top(int A[100], int B[100], int C[100]) {
    int tmp[100];
    funcA(A, tmp);   // Must complete
    funcB(tmp, B);   // before this starts
    funcC(B, C);
}

// After: Concurrent execution with dataflow
void top(int A[100], int B[100], int C[100]) {
    #pragma HLS dataflow
    int tmp1[100], tmp2[100];
    funcA(A, tmp1);  // Produces tmp1
    funcB(tmp1, tmp2);  // Consumes tmp1, produces tmp2 (overlaps with funcA)
    funcC(tmp2, C);  // Consumes tmp2 (overlaps with funcB)
}
```

#### Reduce Control Dependencies
Convert conditional updates to avoid pipeline stalls.

```c
// Before: Conditional causes pipeline stall
for (int i = 0; i < N; i++) {
    #pragma HLS pipeline II=1
    if (i % 2 == 0) {
        out[i] = in[i] * 2;
    } else {
        out[i] = in[i] + 1;
    }
}

// After: Predicated execution (no stall)
for (int i = 0; i < N; i++) {
    #pragma HLS pipeline II=1
    int is_even = (i % 2 == 0);
    int mul_result = in[i] * 2;
    int add_result = in[i] + 1;
    out[i] = is_even ? mul_result : add_result;
}
```

### 4) Arithmetic Optimization
- **Fixed-point conversion**: Replace floating-point with fixed-point when accuracy allows.
- **Resource sharing**: Reuse operators with scheduling rather than fully parallel instantiation.
- **Use bit-accurate types**: `ap_uint`, `ap_int`, `ap_fixed` for smaller hardware.

### 5) Interface Cleanup
- **Remove unused ports**: Delete unused function parameters or arrays.
- **Simplify top-level interface**: Merge related inputs to reduce interface complexity.

---

## Common Kernel Optimization Templates

### Matrix-Vector Multiplication (GEMV)
```c
void gemv(float A[100][100], float x[100], float y[100]) {
    #pragma HLS interface mode=ap_memory port=A
    #pragma HLS array_partition variable=x complete
    
    for (int i = 0; i < 100; i++) {
        #pragma HLS pipeline II=1
        float sum = 0;
        #pragma HLS unroll factor=10
        for (int j = 0; j < 100; j++) {
            sum += A[i][j] * x[j];
        }
        y[i] = sum;
    }
}
```

### FIR Filter
```c
void fir(float x[128], float h[16], float y[128]) {
    #pragma HLS array_partition variable=h complete
    
    float shift_reg[16];
    #pragma HLS array_partition variable=shift_reg complete
    
    for (int n = 0; n < 128; n++) {
        #pragma HLS pipeline II=1
        
        // Shift register
        for (int i = 15; i > 0; i--) {
            #pragma HLS unroll
            shift_reg[i] = shift_reg[i-1];
        }
        shift_reg[0] = x[n];
        
        // Compute tap
        float acc = 0;
        #pragma HLS unroll
        for (int i = 0; i < 16; i++) {
            acc += shift_reg[i] * h[i];
        }
        y[n] = acc;
    }
}
```

### 2D Convolution (Image Processing)
```c
void conv2d(float in[64][64], float out[64][64], float kernel[3][3]) {
    #pragma HLS array_partition variable=kernel complete dim=0
    
    float local_buf[3][66];  // Line buffer
    #pragma HLS array_partition variable=local_buf complete dim=1
    
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            #pragma HLS pipeline II=1
            
            // Load window
            float window[3][3];
            #pragma HLS array_partition variable=window complete dim=0
            for (int ky = 0; ky < 3; ky++) {
                #pragma HLS unroll
                for (int kx = 0; kx < 3; kx++) {
                    int py = y + ky - 1;
                    int px = x + kx - 1;
                    window[ky][kx] = (py >= 0 && py < 64 && px >= 0 && px < 64) 
                                     ? in[py][px] : 0;
                }
            }
            
            // Compute convolution
            float sum = 0;
            #pragma HLS unroll
            for (int ky = 0; ky < 3; ky++) {
                #pragma HLS unroll
                for (int kx = 0; kx < 3; kx++) {
                    sum += window[ky][kx] * kernel[ky][kx];
                }
            }
            out[y][x] = sum;
        }
    }
}
```

---

## Safety and Correctness Notes
- Apply rules only when semantics are preserved.
- For floating-point, respect IEEE behavior and compiler flags (e.g., `-ffast-math`).
- Validate aliasing assumptions before adding `restrict`.
- Benchmark changes; some rewrites may hurt performance on certain targets.
