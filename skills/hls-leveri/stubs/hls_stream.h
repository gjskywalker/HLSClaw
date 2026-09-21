// Minimal functional stub of Vitis HLS <hls_stream.h> for HLS-LeVeri.
// Models hls::stream<T> as a FIFO queue so streaming HLS-C parses and executes
// at the C level. Blocking semantics are approximated by returning T() on empty.
#ifndef LEVERI_HLS_STREAM_H
#define LEVERI_HLS_STREAM_H
#include <deque>
#include <string>

namespace hls {

template <typename T>
class stream {
    std::deque<T> q;
    std::string nm;

  public:
    stream() {}
    stream(const char *name) : nm(name) {}
    void write(const T &x) { q.push_back(x); }
    bool write_nb(const T &x) { q.push_back(x); return true; }
    T read() {
        if (q.empty()) return T();
        T x = q.front();
        q.pop_front();
        return x;
    }
    bool read_nb(T &x) {
        if (q.empty()) return false;
        x = q.front();
        q.pop_front();
        return true;
    }
    bool empty() const { return q.empty(); }
    T &operator<<(const T &x) { q.push_back(x); return q.back(); }
    size_t size() const { return q.size(); }
};

}  // namespace hls

#endif  // LEVERI_HLS_STREAM_H
