#define M 100
#define N 5

static double dot(const double row[N], const double x[N]) {
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
        sum += row[j] * x[j];
    }
    return sum;
}

void dut(double A[M][N], double x[N], double y[N], double tmp[M]) {
    (void)tmp;

    for (int j = 0; j < N; ++j) {
        y[j] = 0.0;
        for (int i = 0; i < M; ++i) {
            // Adding loop pragmas here does not remove the redundant dot-product work.
            // #pragma HLS unroll factor=5
            y[j] += A[i][j] * dot(A[i], x);
        }
    }
}
