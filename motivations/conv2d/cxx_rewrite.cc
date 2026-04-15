#include <hls_stream.h>

#define IN_H 32
#define IN_W 32
#define K 3
#define OUT_H (IN_H - K + 1)
#define OUT_W (IN_W - K + 1)

struct WindowPacket {
    float data[K][K];
};

static void load_windows(
    float input[IN_H][IN_W],
    hls::stream<WindowPacket>& window_stream
) {
    for (int r = 0; r < OUT_H; ++r) {
        for (int c = 0; c < OUT_W; ++c) {
#pragma HLS PIPELINE II=1
            WindowPacket packet;
            for (int kr = 0; kr < K; ++kr) {
#pragma HLS UNROLL
                for (int kc = 0; kc < K; ++kc) {
#pragma HLS UNROLL
                    packet.data[kr][kc] = input[r + kr][c + kc];
                }
            }
            window_stream.write(packet);
        }
    }
}

static float convolve_window(
    const WindowPacket& window,
    float kernel[K][K]
) {
#pragma HLS INLINE
    float sum = 0.0f;
    for (int kr = 0; kr < K; ++kr) {
#pragma HLS UNROLL
        for (int kc = 0; kc < K; ++kc) {
#pragma HLS UNROLL
            sum += window.data[kr][kc] * kernel[kr][kc];
        }
    }
    return sum;
}

static void compute_outputs(
    hls::stream<WindowPacket>& window_stream,
    float kernel[K][K],
    float output[OUT_H][OUT_W]
) {
    for (int r = 0; r < OUT_H; ++r) {
        for (int c = 0; c < OUT_W; ++c) {
#pragma HLS PIPELINE II=1
            WindowPacket packet = window_stream.read();
            output[r][c] = convolve_window(packet, kernel);
        }
    }
}

void dut(
    float input[IN_H][IN_W],
    float kernel[K][K],
    float output[OUT_H][OUT_W]
) {
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=0
#pragma HLS DATAFLOW

    hls::stream<WindowPacket> window_stream;
#pragma HLS STREAM variable=window_stream depth=16

    load_windows(input, window_stream);
    compute_outputs(window_stream, kernel, output);
}
