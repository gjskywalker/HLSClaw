// Minimal functional stub of Vitis HLS <ap_int.h> for HLS-LeVeri.
// Models ap_int<W>/ap_uint<W> as native integers with width masking, so that
// HLS-C parses (static tier) and executes (dynamic tier) at the C level.
// This is a reconstruction aid, not a bit-accurate HLS arbitrary-precision type.
#ifndef LEVERI_AP_INT_H
#define LEVERI_AP_INT_H
#include <cstdint>

template <int W>
struct ap_int {
    long long v;
    ap_int(long long x = 0) { v = _mask(x); }
    static long long _mask(long long x) {
        if (W >= 64) return x;
        long long m = (1LL << W) - 1;
        long long r = x & m;
        if (r & (1LL << (W - 1))) r |= ~m;  // sign extend
        return r;
    }
    operator long long() const { return v; }
    ap_int &operator=(long long x) { v = _mask(x); return *this; }
    // Compound assignments re-mask to W bits so width overflow is emulated.
    ap_int &operator+=(long long x) { v = _mask(v + x); return *this; }
    ap_int &operator-=(long long x) { v = _mask(v - x); return *this; }
    ap_int &operator*=(long long x) { v = _mask(v * x); return *this; }
    ap_int &operator/=(long long x) { v = _mask(v / x); return *this; }
    ap_int &operator%=(long long x) { v = _mask(v % x); return *this; }
    ap_int &operator&=(long long x) { v = _mask(v & x); return *this; }
    ap_int &operator|=(long long x) { v = _mask(v | x); return *this; }
    ap_int &operator^=(long long x) { v = _mask(v ^ x); return *this; }
    ap_int &operator<<=(int s) { v = _mask(v << s); return *this; }
    ap_int &operator>>=(int s) { v = _mask(v >> s); return *this; }
    ap_int &operator++() { v = _mask(v + 1); return *this; }
    ap_int operator++(int) { ap_int t = *this; v = _mask(v + 1); return t; }
    ap_int &operator--() { v = _mask(v - 1); return *this; }
    ap_int operator--(int) { ap_int t = *this; v = _mask(v - 1); return t; }
};

template <int W>
struct ap_uint {
    unsigned long long v;
    ap_uint(unsigned long long x = 0) { v = _mask(x); }
    static unsigned long long _mask(unsigned long long x) {
        if (W >= 64) return x;
        return x & ((1ULL << W) - 1);
    }
    operator unsigned long long() const { return v; }
    ap_uint &operator=(unsigned long long x) { v = _mask(x); return *this; }
    ap_uint &operator+=(unsigned long long x) { v = _mask(v + x); return *this; }
    ap_uint &operator-=(unsigned long long x) { v = _mask(v - x); return *this; }
    ap_uint &operator*=(unsigned long long x) { v = _mask(v * x); return *this; }
    ap_uint &operator/=(unsigned long long x) { v = _mask(v / x); return *this; }
    ap_uint &operator%=(unsigned long long x) { v = _mask(v % x); return *this; }
    ap_uint &operator&=(unsigned long long x) { v = _mask(v & x); return *this; }
    ap_uint &operator|=(unsigned long long x) { v = _mask(v | x); return *this; }
    ap_uint &operator^=(unsigned long long x) { v = _mask(v ^ x); return *this; }
    ap_uint &operator<<=(int s) { v = _mask(v << s); return *this; }
    ap_uint &operator>>=(int s) { v = _mask(v >> s); return *this; }
    ap_uint &operator++() { v = _mask(v + 1); return *this; }
    ap_uint operator++(int) { ap_uint t = *this; v = _mask(v + 1); return t; }
    ap_uint &operator--() { v = _mask(v - 1); return *this; }
    ap_uint operator--(int) { ap_uint t = *this; v = _mask(v - 1); return t; }
};

#endif  // LEVERI_AP_INT_H
