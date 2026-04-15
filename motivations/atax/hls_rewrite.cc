#include <hls_stream.h>

#define M 100
#define N 5

static void compute_ax(
    const double A[M][N],
    const double x[N],
    double tmp[M],
    hls::stream<double>& tmp_stream
) {
    for (int i = 0; i < M; ++i) {
#pragma HLS PIPELINE II=1
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
            sum += A[i][j] * x[j];
        }
        tmp[i] = sum;
        tmp_stream.write(sum);
    }
}

static void accumulate_at_tmp(
    const double A[M][N],
    hls::stream<double>& tmp_stream,
    double y[N]
) {
    double acc[N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
        acc[j] = 0.0;
    }

    for (int i = 0; i < M; ++i) {
        // Allow the double-precision accumulation to use a multi-cycle recurrence.
#pragma HLS PIPELINE II=8
        double tmp_i = tmp_stream.read();
        for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
            acc[j] += A[i][j] * tmp_i;
        }
    }

    for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
        y[j] = acc[j];
    }
}

void dut(double A[M][N], double x[N], double y[N], double tmp[M]) {
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=5 dim=2
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#pragma HLS DATAFLOW

    hls::stream<double> tmp_stream;
#pragma HLS STREAM variable=tmp_stream depth=16

    compute_ax(A, x, tmp, tmp_stream);
    accumulate_at_tmp(A, tmp_stream, y);
}
