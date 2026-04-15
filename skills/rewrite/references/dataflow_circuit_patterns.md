# Dataflow Circuit Design Patterns

Reference patterns extracted from an external FPGA 2025 dataflow-circuit compiler artifact set.

These patterns demonstrate **stream-based dataflow** HLS design — an advanced optimization
strategy that decomposes computations into producer-consumer pipeline stages connected
via `hls::stream`, enabling task-level parallelism with `#pragma HLS DATAFLOW`.

---

## Pattern 1: Streaming Matrix Multiply (GEMM)

**Artifact family:** Polybench opt5 GEMM

**Core Idea:** Decompose GEMM into 3 dataflow nodes connected by stream arrays.
Each node is fully pipelined (II=1). Arrays are partitioned for parallel port access.

**Key Techniques:**
- `hls::stream<float> v[4][11]` — stream arrays for parallel element transfer
- `#pragma HLS array_partition variable=X cyclic dim=D factor=F` — match unroll factors
- Conditional init/flush with `if (idx == 0)` / `if (idx == LAST)`
- Accumulate in local buffer, write to stream only on last reduction index

```cpp
// Tiled GEMM node: accumulates A*B into local buffer, streams result
void gemm_compute(
  float A[M][K], float B[K][N],
  hls::stream<float> out[TILE_M][TILE_N], float init_val
) {
  #pragma HLS array_partition variable=A cyclic dim=1 factor=TILE_M
  #pragma HLS array_partition variable=B cyclic dim=2 factor=TILE_N
  float acc[M][N];
  #pragma HLS array_partition variable=acc cyclic dim=1 factor=TILE_M
  #pragma HLS array_partition variable=acc cyclic dim=2 factor=TILE_N

  for (int k_tile = 0; k_tile < K/TILE_K; k_tile++) {
    for (int m_tile = 0; m_tile < M/TILE_M; m_tile++) {
      for (int n_tile = 0; n_tile < N/TILE_N; n_tile++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_flatten
        for (int tk = 0; tk < TILE_K; tk++) {
          for (int tm = 0; tm < TILE_M; tm++) {
            for (int tn = 0; tn < TILE_N; tn++) {
              int k = tk + k_tile * TILE_K;
              int m = tm + m_tile * TILE_M;
              int n = tn + n_tile * TILE_N;
              if (k == 0) acc[m][n] = init_val;      // init on first
              acc[m][n] += A[m][k] * B[k][n];
              if (k == K-1) out[tm][tn].write(acc[m][n]); // flush on last
            }
          }
        }
      }
    }
  }
}
```

**When to use:** Any matrix multiply or reduction-heavy kernel. Scale parallelism via tile factors.

---

## Pattern 2: CNN Depthwise Separable Convolution (Streaming)

**Source:** `cnns/opt5/DepthwiseSeparableConvBlock`

**Core Idea:** Decompose depthwise-separable conv into sequential stages within a single
`#pragma HLS DATAFLOW` function:
1. Zero-pad input → local buffer
2. Copy input data to padded buffer
3. Depthwise conv → stream output
4. Pointwise 1×1 conv → stream output
5. BatchNorm + ReLU → stream output
6. Write to output array

**Key Techniques:**
- All stages in one `forward()` with `#pragma HLS DATAFLOW`
- `hls::stream<float> v[1][2][14]` — multi-dimensional stream arrays for spatial tiling
- Inline activation: `bool v67 = v66 > 0; float v68 = v67 ? v66 : 0.0f;` (ReLU as ternary)
- Inline BatchNorm: `1.0/sqrt(eps + 1.0) * x + bias`

```cpp
void forward(float in[1][C][H][W], float pw[C][C][1][1],
             float dw[C][3][3], float out[1][C][H][W]) {
  #pragma HLS DATAFLOW
  // Stage 1: zero-pad
  float padded[C][H+2][W+2];
  // ... init to 0, copy in with +1 offset

  // Stage 2: depthwise 3×3 conv → stream
  hls::stream<float> dw_out[1][TILE_H][TILE_W];
  // ... accumulate over 3×3 kernel, flush to stream

  // Stage 3: pointwise 1×1 conv → stream
  hls::stream<float> pw_out[1][TILE_H][TILE_W];
  // ... reduce over input channels, flush to stream

  // Stage 4: BN + ReLU → stream
  hls::stream<float> act_out[1][TILE_H][TILE_W];
  // ... x = x * rsqrt(eps+1) + bias; x = max(x, 0)

  // Stage 5: write back
  // ... read from stream, store to out[]
}
```

**When to use:** Any multi-stage CNN layer. Each conv/activation becomes a pipeline stage.

---

## Pattern 3: Multi-Head Self-Attention (Streaming Dataflow)

**Source:** `transformers/opt5/MultiHeadSelfAttention1`

**Core Idea:** Decompose full MHSA into 25+ fine-grained nodes:
1. Input broadcast (1→3 streams for Q, K, V projections)
2. Weight transpose nodes
3. Linear projections (matmul nodes)
4. Bias add nodes
5. Reshape to multi-head (split by head_dim)
6. Q·Kᵀ matmul → scaled scores
7. Softmax decomposition: max → subtract+exp → sum → divide (4 separate nodes!)
8. Score · V matmul
9. Reshape back + output projection

**Key Techniques:**
- Fan-out via broadcast node: one input, multiple output streams
- Softmax split into 4 streaming nodes (numerically stable):
  - `node11`: row-max reduction → stream
  - `node10`: subtract max + exp → two output streams (one for sum, one for divide)
  - `node9`: row-sum reduction → stream
  - `node8`: element-wise divide by sum
- All matmuls follow same Pattern 1 (conditional init/flush)
- `#pragma HLS STREAM variable=v depth=8192` for deep FIFOs

```cpp
void forward(...) {
  #pragma HLS DATAFLOW
  // 16+ stream declarations with depth=8192 or 32768

  node25(input, q_stream, k_stream, v_stream);     // broadcast
  node24(W_q, W_q_T);                               // transpose
  node23(q_stream, W_q_T, proj_q, 0.0);            // Q = input @ W_q
  node22(proj_q, bias_q, Q);                         // + bias
  // ... similar for K, V

  node15(Q_reshaped, q_head_stream);                // reshape to heads
  node14(K_reshaped, k_head_transposed);            // K^T for each head
  node13(q_head_stream, k_head_transposed, scores, 0.0);  // Q·K^T
  node12(scores, scaled1, scaled2, sqrt_dk);        // scale + fan-out

  // Streaming softmax (4 nodes)
  node11(scaled1, row_max, -INFINITY);              // max per row
  node10(scaled2, row_max, exp_vals, exp_vals2);    // exp(x - max)
  node9(exp_vals, row_sum, 0.0);                    // sum per row
  node8(exp_vals2, row_sum, attention);             // divide by sum

  node6(V_heads, attention, context, 0.0);          // score · V
  // ... reshape + output projection
}
```

**When to use:** Any attention mechanism. The streaming softmax decomposition is the most
important pattern — it avoids materializing the full attention matrix.

---

## Pattern 4: Residual MLP (Streaming with Skip Connections)

**Source:** `mlps/opt5/ResMLP`

**Core Idea:** MLP with residual connections, all linear layers as streaming matmuls.
Skip connections implemented by reading from two streams and adding.

**Key Techniques:**
- Stream arrays `hls::stream<float> v[4][1]` — width matches unroll factor
- Weight matrix streaming: read weight from array, stream to consumer
- Residual add: `output = linear_out + skip_stream`
- GELU activation inline: `x * 0.5 * (1 + erf(x / sqrt(2)))`
- Deep pipeline: 10+ nodes chained with DATAFLOW

```cpp
// Residual connection pattern
void residual_add(
  hls::stream<float> &linear_out,
  hls::stream<float> &skip,
  hls::stream<float> &result
) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      #pragma HLS pipeline II=1
      float a = linear_out.read();
      float b = skip.read();
      result.write(a + b);
    }
  }
}

// Streaming GELU
void gelu_stream(hls::stream<float> &in, hls::stream<float> &out) {
  for (...) {
    #pragma HLS pipeline II=1
    float x = in.read();
    float v = x * 0.5f * (1.0f + erf(x * 0.7071067811865475f));
    out.write(v);
  }
}
```

**When to use:** Any network with residual/skip connections. Fan-out the input to both
the transform path and skip path using separate streams.

---

## Universal Dataflow Circuit Rules

1. **Every compute stage is a separate function** connected by `hls::stream`
2. **Top function uses `#pragma HLS DATAFLOW`** to enable concurrent execution
3. **All inner loops use `#pragma HLS pipeline II=1` + `#pragma HLS loop_flatten`**
4. **Conditional init/flush pattern** for reductions:
   ```cpp
   if (reduce_idx == 0) acc = init_val;      // reset accumulator
   acc += a * b;                              // accumulate
   if (reduce_idx == LAST) out.write(acc);    // flush result
   ```
5. **Array partitioning matches parallelism**: `factor=` equals the unroll/tile factor
6. **Stream depth** should cover worst-case producer-consumer latency mismatch
7. **Fan-out** uses one read + multiple writes (broadcast node)
8. **Complex ops decompose** into multiple nodes (e.g., softmax → 4 nodes)

---

## Reference Optimization Levels (opt1 -> opt5)

The external artifact set exposes five representative optimization levels:
- **opt1**: Minimal — basic streaming, no array partitioning
- **opt2**: Moderate tiling
- **opt3**: Balanced partition + pipeline
- **opt4**: Aggressive partitioning
- **opt5**: Maximum parallelism — full array partition, largest tile factors, deepest streams

When rewriting HLS code, start with opt1-level patterns and escalate to opt5 based on
resource budget (DSP count). The opt5 patterns above use the most parallel configurations.
