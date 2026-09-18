// Host glue: Rust FFI ↔ CUDA kernel.
//
// Clamp:
//   1. One-shot shield_cuda_clamp — alloc / copy / free every call.
//   2. Context shield_cuda_ctx_*  — cached device + pinned buffers + stream.
//
// AABB:
//   shield_cuda_aabb_mask / hits. Below kMinGpuPairs the host fill runs.
//   At ≥64 pairs: grow-only AabbDeviceScratch + pair-parallel kernel.
//   Any GPU failure falls back to host fill (never fail closed).

#include "cuda_runtime_compat.h"

extern "C" int shield_cuda_launch_aabb_mask(
    const float* device_link_min,
    const float* device_link_max,
    size_t n_links,
    const float* device_obs_min,
    const float* device_obs_max,
    size_t n_obs,
    unsigned char* device_mask,
    cudaStream_t stream
);

extern "C" int shield_cuda_launch_clamp(
    const float* device_input,
    const float* device_limit,
    float* device_output,
    size_t n,
    cudaStream_t stream
);

extern "C" void shield_cuda_aabb_mask_host(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* mask
);

namespace {

struct ShieldCudaCtx {
    size_t capacity_floats = 0;
    float* d_input  = nullptr;
    float* d_limit  = nullptr;
    float* d_output = nullptr;
    float* h_input  = nullptr;  // pinned
    float* h_limit  = nullptr;  // pinned
    float* h_output = nullptr;  // pinned
    cudaStream_t stream = nullptr;
};

void free_device_buffers(ShieldCudaCtx* c) {
    if (c->d_input)  { cudaFree(c->d_input);  c->d_input  = nullptr; }
    if (c->d_limit)  { cudaFree(c->d_limit);  c->d_limit  = nullptr; }
    if (c->d_output) { cudaFree(c->d_output); c->d_output = nullptr; }
}

void free_pinned_buffers(ShieldCudaCtx* c) {
    if (c->h_input)  { cudaFreeHost(c->h_input);  c->h_input  = nullptr; }
    if (c->h_limit)  { cudaFreeHost(c->h_limit);  c->h_limit  = nullptr; }
    if (c->h_output) { cudaFreeHost(c->h_output); c->h_output = nullptr; }
}

// Grow device + pinned buffers to `new_capacity` floats. 0 = ok, else CUDA code.
int reserve(ShieldCudaCtx* c, size_t new_capacity) {
    if (new_capacity <= c->capacity_floats) {
        return 0;
    }
    free_device_buffers(c);
    free_pinned_buffers(c);

    const size_t bytes = new_capacity * sizeof(float);
    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&c->d_input),  bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(reinterpret_cast<void**>(&c->d_limit),  bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(reinterpret_cast<void**>(&c->d_output), bytes);
    if (err != cudaSuccess) return static_cast<int>(err);

    err = cudaMallocHost(reinterpret_cast<void**>(&c->h_input),  bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMallocHost(reinterpret_cast<void**>(&c->h_limit),  bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMallocHost(reinterpret_cast<void**>(&c->h_output), bytes);
    if (err != cudaSuccess) return static_cast<int>(err);

    c->capacity_floats = new_capacity;
    return 0;
}

}  // namespace

// --- One-shot API (tests / simple callers) ----------------------------------

extern "C" int shield_cuda_clamp(
    const float* host_input,
    const float* host_limit,
    float* host_output,
    size_t n
) {
    if (host_input == nullptr || host_limit == nullptr || host_output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0) {
        return 0;
    }

    ShieldCudaCtx local;
    int rc = reserve(&local, n);
    auto cleanup = [&local]() {
        free_device_buffers(&local);
        free_pinned_buffers(&local);
    };
    if (rc != 0) { cleanup(); return rc; }

    memcpy(local.h_input, host_input, n * sizeof(float));
    memcpy(local.h_limit, host_limit, n * sizeof(float));

    cudaError_t err;
    err = cudaMemcpy(local.d_input, local.h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }
    err = cudaMemcpy(local.d_limit, local.h_limit, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    int launch_rc = shield_cuda_launch_clamp(
        local.d_input, local.d_limit, local.d_output, n, /*stream=*/0);
    if (launch_rc != 0) { cleanup(); return launch_rc; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaMemcpy(host_output, local.d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cleanup();
    return static_cast<int>(err);
}

// --- Context API (hot path) -------------------------------------------------

extern "C" int shield_cuda_ctx_create(size_t initial_capacity, void** out_ctx) {
    if (out_ctx == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    auto* c = new ShieldCudaCtx();
    cudaError_t err = cudaStreamCreate(&c->stream);
    if (err != cudaSuccess) {
        delete c;
        return static_cast<int>(err);
    }
    if (initial_capacity > 0) {
        int rc = reserve(c, initial_capacity);
        if (rc != 0) {
            cudaStreamDestroy(c->stream);
            delete c;
            return rc;
        }
    }
    *out_ctx = c;
    return 0;
}

extern "C" void shield_cuda_ctx_destroy(void* opaque) {
    if (opaque == nullptr) return;
    auto* c = static_cast<ShieldCudaCtx*>(opaque);
    free_device_buffers(c);
    free_pinned_buffers(c);
    if (c->stream != nullptr) {
        cudaStreamDestroy(c->stream);
    }
    delete c;
}

extern "C" int shield_cuda_ctx_clamp(
    void* opaque,
    const float* host_input,
    const float* host_limit,
    float* host_output,
    size_t n
) {
    if (opaque == nullptr || host_input == nullptr ||
        host_limit == nullptr || host_output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0) {
        return 0;
    }
    auto* c = static_cast<ShieldCudaCtx*>(opaque);
    int rc = reserve(c, n);
    if (rc != 0) {
        return rc;
    }

    // Stage into pinned buffers so the HtoD copy can run async on our stream.
    memcpy(c->h_input, host_input, n * sizeof(float));
    memcpy(c->h_limit, host_limit, n * sizeof(float));

    cudaError_t err;
    err = cudaMemcpyAsync(c->d_input, c->h_input, n * sizeof(float),
                          cudaMemcpyHostToDevice, c->stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpyAsync(c->d_limit, c->h_limit, n * sizeof(float),
                          cudaMemcpyHostToDevice, c->stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    int launch_rc = shield_cuda_launch_clamp(
        c->d_input, c->d_limit, c->d_output, n, c->stream);
    if (launch_rc != 0) return launch_rc;

    err = cudaMemcpyAsync(c->h_output, c->d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost, c->stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    err = cudaStreamSynchronize(c->stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    memcpy(host_output, c->h_output, n * sizeof(float));
    return 0;
}

// AABB overlap: N×M mask (row-major). Below kMinGpuPairs we skip the
// launch — a 7-link × 4-obstacle scene is faster on the host. The kernel
// is still there for dense scenes / benches (force by packing ≥64 pairs).
static const size_t kMinGpuPairs = 64;

struct AabbDeviceScratch {
    size_t n_links_cap = 0;
    size_t n_obs_cap = 0;
    float* d_lmin = nullptr;
    float* d_lmax = nullptr;
    float* d_omin = nullptr;
    float* d_omax = nullptr;
    unsigned char* d_mask = nullptr;
};

static AabbDeviceScratch g_aabb;
static std::mutex g_aabb_mu;

static void aabb_scratch_free(AabbDeviceScratch* s) {
    if (s->d_lmin) { cudaFree(s->d_lmin); s->d_lmin = nullptr; }
    if (s->d_lmax) { cudaFree(s->d_lmax); s->d_lmax = nullptr; }
    if (s->d_omin) { cudaFree(s->d_omin); s->d_omin = nullptr; }
    if (s->d_omax) { cudaFree(s->d_omax); s->d_omax = nullptr; }
    if (s->d_mask) { cudaFree(s->d_mask); s->d_mask = nullptr; }
    s->n_links_cap = 0;
    s->n_obs_cap = 0;
}

static int aabb_scratch_reserve(AabbDeviceScratch* s, size_t n_links, size_t n_obs) {
    if (n_links <= s->n_links_cap && n_obs <= s->n_obs_cap && s->d_mask != nullptr) {
        return 0;
    }
    aabb_scratch_free(s);
    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&s->d_lmin), n_links * 3 * sizeof(float));
    if (err != cudaSuccess) { aabb_scratch_free(s); return static_cast<int>(err); }
    err = cudaMalloc(reinterpret_cast<void**>(&s->d_lmax), n_links * 3 * sizeof(float));
    if (err != cudaSuccess) { aabb_scratch_free(s); return static_cast<int>(err); }
    err = cudaMalloc(reinterpret_cast<void**>(&s->d_omin), n_obs * 3 * sizeof(float));
    if (err != cudaSuccess) { aabb_scratch_free(s); return static_cast<int>(err); }
    err = cudaMalloc(reinterpret_cast<void**>(&s->d_omax), n_obs * 3 * sizeof(float));
    if (err != cudaSuccess) { aabb_scratch_free(s); return static_cast<int>(err); }
    err = cudaMalloc(reinterpret_cast<void**>(&s->d_mask), n_links * n_obs);
    if (err != cudaSuccess) { aabb_scratch_free(s); return static_cast<int>(err); }
    s->n_links_cap = n_links;
    s->n_obs_cap = n_obs;
    return 0;
}

static int aabb_args_ok(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* out
) {
    if (out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_links > 0 && (link_min == nullptr || link_max == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_obs > 0 && (obs_min == nullptr || obs_max == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return 0;
}

// Device path. Caller holds g_aabb_mu. Non-zero = fall back to host fill.
static int aabb_mask_gpu(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* mask
) {
    int rc = aabb_scratch_reserve(&g_aabb, n_links, n_obs);
    if (rc != 0) {
        return rc;
    }
    const size_t pairs = n_links * n_obs;
    cudaError_t err;
    err = cudaMemcpy(g_aabb.d_lmin, link_min, n_links * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpy(g_aabb.d_lmax, link_max, n_links * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpy(g_aabb.d_omin, obs_min, n_obs * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpy(g_aabb.d_omax, obs_max, n_obs * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return static_cast<int>(err);

    int launch_rc = shield_cuda_launch_aabb_mask(
        g_aabb.d_lmin, g_aabb.d_lmax, n_links,
        g_aabb.d_omin, g_aabb.d_omax, n_obs,
        g_aabb.d_mask, /*stream=*/0);
    if (launch_rc != 0) {
        return launch_rc;
    }
    err = cudaMemcpy(mask, g_aabb.d_mask, pairs, cudaMemcpyDeviceToHost);
    return static_cast<int>(err);
}

extern "C" int shield_cuda_aabb_mask(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* mask
) {
    int bad = aabb_args_ok(link_min, link_max, n_links, obs_min, obs_max, n_obs, mask);
    if (bad != 0) {
        return bad;
    }
    if (n_links == 0 || n_obs == 0) {
        return 0;
    }
    const size_t pairs = n_links * n_obs;
    if (pairs >= kMinGpuPairs) {
        std::lock_guard<std::mutex> lock(g_aabb_mu);
        if (aabb_mask_gpu(
                link_min, link_max, n_links, obs_min, obs_max, n_obs, mask) == 0) {
            return 0;
        }
    }
    shield_cuda_aabb_mask_host(
        link_min, link_max, n_links, obs_min, obs_max, n_obs, mask);
    return 0;
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
    int bad = aabb_args_ok(link_min, link_max, n_links, obs_min, obs_max, n_obs, hits);
    if (bad != 0) {
        return bad;
    }
    if (n_links == 0) {
        return 0;
    }
    if (n_obs == 0) {
        memset(hits, 0, n_links);
        return 0;
    }
    unsigned char* mask = static_cast<unsigned char*>(malloc(n_links * n_obs));
    if (mask == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    int rc = shield_cuda_aabb_mask(
        link_min, link_max, n_links, obs_min, obs_max, n_obs, mask);
    if (rc == 0) {
        for (size_t i = 0; i < n_links; ++i) {
            unsigned char hit = 0;
            for (size_t j = 0; j < n_obs; ++j) {
                if (mask[i * n_obs + j]) {
                    hit = 1;
                    break;
                }
            }
            hits[i] = hit;
        }
    }
    free(mask);
    return rc;
}
