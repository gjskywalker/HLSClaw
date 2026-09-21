void dut(const int input[16], int output[16], int bias) {
    for (int i = 15; i >= 0; --i) {
        output[i] = bias + 3 * input[i];
    }
}
