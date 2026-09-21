// Minimal functional stub of Vitis HLS <ap_fixed.h> for HLS-LeVeri.
// Models ap_fixed<W,I>/ap_ufixed<W,I> as double. Not bit-accurate; it lets
// HLS-C parse and run at the C level for consistency checking.
#ifndef LEVERI_AP_FIXED_H
#define LEVERI_AP_FIXED_H

template <int W, int I>
struct ap_fixed {
    double v;
    ap_fixed(double x = 0) : v(x) {}
    operator double() const { return v; }
    ap_fixed &operator=(double x) { v = x; return *this; }
    ap_fixed &operator+=(double x) { v += x; return *this; }
    ap_fixed &operator-=(double x) { v -= x; return *this; }
    ap_fixed &operator*=(double x) { v *= x; return *this; }
    ap_fixed &operator/=(double x) { v /= x; return *this; }
};

template <int W, int I>
struct ap_ufixed {
    double v;
    ap_ufixed(double x = 0) : v(x < 0 ? 0 : x) {}
    operator double() const { return v; }
    ap_ufixed &operator=(double x) { v = (x < 0 ? 0 : x); return *this; }
    ap_ufixed &operator+=(double x) { v += x; return *this; }
    ap_ufixed &operator-=(double x) { v -= x; return *this; }
    ap_ufixed &operator*=(double x) { v *= x; return *this; }
    ap_ufixed &operator/=(double x) { v /= x; return *this; }
};

#endif  // LEVERI_AP_FIXED_H
