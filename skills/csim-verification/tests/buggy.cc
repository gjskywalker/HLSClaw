void dut(const int input[16], int output[16], int bias) {
    for (int i = 0; i < 16; ++i) {
        output[i] = input[i] * 3 + bias + 1;
    }
}
