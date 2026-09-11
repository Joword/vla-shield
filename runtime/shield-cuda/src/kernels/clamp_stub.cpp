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

extern "C" int shield_cuda_aabb_hits(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* hits
) {
    if ((n_links > 0 && (link_min == nullptr || link_max == nullptr || hits == nullptr)) ||
        (n_obs > 0 && (obs_min == nullptr || obs_max == nullptr))) {
        return 1;
    }
    for (size_t i = 0; i < n_links; ++i) {
        unsigned char hit = 0;
        const float* a0 = link_min + i * 3;
        const float* a1 = link_max + i * 3;
        for (size_t j = 0; j < n_obs; ++j) {
            const float* b0 = obs_min + j * 3;
            const float* b1 = obs_max + j * 3;
            if (a0[0] <= b1[0] && a1[0] >= b0[0] &&
                a0[1] <= b1[1] && a1[1] >= b0[1] &&
                a0[2] <= b1[2] && a1[2] >= b0[2]) {
                hit = 1;
                break;
            }
        }
        hits[i] = hit;
    }
    return 0;
}
