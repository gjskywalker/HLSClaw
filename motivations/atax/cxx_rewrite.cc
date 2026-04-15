#define M 100
#define N 5

static void compute_ax(
    const double A[M][N],
    const double x[N],
    double tmp[M]
) {
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[i][j] * x[j];
        }
        tmp[i] = sum;
    }
}

static void accumulate_at_tmp(
    const double A[M][N],
    const double tmp[M],
    double y[N]
) {
    for (int j = 0; j < N; ++j) {
        y[j] = 0.0;
    }

    for (int i = 0; i < M; ++i) {
        double tmp_i = tmp[i];
        for (int j = 0; j < N; ++j) {
            y[j] += A[i][j] * tmp_i;
        }
    }
}

void dut(double A[M][N], double x[N], double y[N], double tmp[M]) {
    compute_ax(A, x, tmp);
    accumulate_at_tmp(A, tmp, y);
}
