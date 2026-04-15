#define IN_H 32
#define IN_W 32
#define K 3
#define OUT_H (IN_H - K + 1)
#define OUT_W (IN_W - K + 1)

void dut(
    float input[IN_H][IN_W],
    float kernel[K][K],
    float output[OUT_H][OUT_W]
) {
    for (int r = 0; r < OUT_H; ++r) {
        for (int c = 0; c < OUT_W; ++c) {
            // Pipelining this loop alone still leaves every output pixel reloading
            // the full KxK window from memory.
            // #pragma HLS PIPELINE II=1
            float sum = 0.0f;
            for (int kr = 0; kr < K; ++kr) {
                for (int kc = 0; kc < K; ++kc) {
                    sum += input[r + kr][c + kc] * kernel[kr][kc];
                }
            }
            output[r][c] = sum;
        }
    }
}
