// CPU fallback implementing the same C ABI as the CUDA backend.
// Compiled only when `nvcc` is unavailable (see build.rs).
//
// Mirrors both API tiers exposed by cuda_host.cpp so call sites do not have
// to special-case the backend:
//   * stateless   shield_cuda_clamp
//   * stateful    shield_cuda_ctx_create / destroy / clamp
//
// The "context" here is a tiny header that the Rust side opaquely owns;
// no device memory is involved, so create/destroy are O(1) and clamp_ctx
// just defers to the stateless loop.

#include <stddef.h>

namespace {

struct ShieldCudaCtx {
    size_t capacity_floats;
};

inline int clamp_loop(
    const float* in,
    const float* lim,
    float* out,
    size_t n
) {
    if (in == nullptr || lim == nullptr || out == nullptr) {
        return 1;  // mirrors cudaErrorInvalidValue
    }
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        float l = lim[i];
        if (x > l) x = l;
        if (x < -l) x = -l;
        out[i] = x;
    }
    return 0;
}

}  // namespace

extern "C" int shield_cuda_clamp(
    const float* host_input,
    const float* host_limit,
    float* host_output,
    size_t n
) {
    return clamp_loop(host_input, host_limit, host_output, n);
}

extern "C" int shield_cuda_ctx_create(size_t initial_capacity, void** out_ctx) {
    if (out_ctx == nullptr) return 1;
    auto* c = new ShieldCudaCtx{initial_capacity};
    *out_ctx = c;
    return 0;
}

extern "C" void shield_cuda_ctx_destroy(void* opaque) {
    if (opaque == nullptr) return;
    delete static_cast<ShieldCudaCtx*>(opaque);
}

extern "C" int shield_cuda_ctx_clamp(
    void* opaque,
    const float* host_input,
    const float* host_limit,
    float* host_output,
    size_t n
) {
    if (opaque == nullptr) return 1;
    auto* c = static_cast<ShieldCudaCtx*>(opaque);
    if (n > c->capacity_floats) {
        c->capacity_floats = n;
    }
    return clamp_loop(host_input, host_limit, host_output, n);
}
