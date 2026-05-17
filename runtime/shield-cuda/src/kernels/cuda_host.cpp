// Host-side C++ glue between the Rust FFI boundary and the CUDA kernel.
//
// Two API tiers are exposed across the C ABI:
//
//   1. Stateless one-shot:
//        int shield_cuda_clamp(host*, host*, host*, n);
//      Allocates / copies / frees per call.  Convenient for tests; not for
//      the hot path.
//
//   2. Stateful context (preferred on hot path):
//        int  shield_cuda_ctx_create(size_t capacity, void** out_ctx);
//        void shield_cuda_ctx_destroy(void* ctx);
//        int  shield_cuda_ctx_clamp(void* ctx, host*, host*, host*, n);
//      The context caches:
//        * three device buffers sized for `capacity` floats (cudaMalloc once),
//        * three pinned host staging buffers (cudaMallocHost),
//        * one persistent CUDA stream for async copies + launches.
//      Subsequent calls with n <= capacity reuse all of the above; calls with
//      n > capacity transparently grow the buffers.

#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>

extern "C" int shield_cuda_launch_clamp(
    const float* device_input,
    const float* device_limit,
    float* device_output,
    size_t n,
    cudaStream_t stream
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

// (Re)allocate all three device buffers and the three pinned host buffers
// for `new_capacity` floats.  Returns 0 on success or a CUDA error code.
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

// --- Stateless API (kept for tests and simple callers) ----------------------

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

// --- Stateful context API (preferred on the hot path) -----------------------

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

    // Copy callers' host buffers into pinned staging buffers so the
    // subsequent HtoD transfer can run async on our private stream.
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
