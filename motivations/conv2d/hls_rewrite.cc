#define IN_H 32
#define IN_W 32
#define K 3
#define OUT_H (IN_H - K + 1)
#define OUT_W (IN_W - K + 1)

static float convolve_window(float window[K][K], const float kernel[K][K]) {
#pragma HLS INLINE
    float sum = 0.0f;
    for (int kr = 0; kr < K; ++kr) {
#pragma HLS UNROLL
        for (int kc = 0; kc < K; ++kc) {
#pragma HLS UNROLL
            sum += window[kr][kc] * kernel[kr][kc];
        }
    }
    return sum;
}

void dut(
    float input[IN_H][IN_W],
    float kernel[K][K],
    float output[OUT_H][OUT_W]
) {
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

    float linebuf[K - 1][IN_W];
    float window[K][K];
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int lb = 0; lb < K - 1; ++lb) {
        for (int c = 0; c < IN_W; ++c) {
#pragma HLS PIPELINE II=1
            linebuf[lb][c] = 0.0f;
        }
    }

    for (int wr = 0; wr < K; ++wr) {
        for (int wc = 0; wc < K; ++wc) {
#pragma HLS UNROLL
            window[wr][wc] = 0.0f;
        }
    }

    for (int r = 0; r < IN_H; ++r) {
        for (int c = 0; c < IN_W; ++c) {
#pragma HLS PIPELINE II=1
            float pixel = input[r][c];

            for (int wr = 0; wr < K; ++wr) {
#pragma HLS UNROLL
                for (int wc = 0; wc < K - 1; ++wc) {
#pragma HLS UNROLL
                    window[wr][wc] = window[wr][wc + 1];
                }
            }

            for (int wr = 0; wr < K - 1; ++wr) {
#pragma HLS UNROLL
                window[wr][K - 1] = linebuf[wr][c];
            }
            window[K - 1][K - 1] = pixel;

            for (int lb = 0; lb < K - 2; ++lb) {
#pragma HLS UNROLL
                linebuf[lb][c] = linebuf[lb + 1][c];
            }
            linebuf[K - 2][c] = pixel;

            if (r >= K - 1 && c >= K - 1) {
                output[r - (K - 1)][c - (K - 1)] = convolve_window(window, kernel);
            }
        }
    }
}
