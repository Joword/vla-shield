// Symmetric per-dim clamp kernel.
//
// Device-only:
//   * `clamp_kernel`              – one thread per element
//   * `shield_cuda_launch_clamp`  – C-ABI launcher; pointers already on device,
//                                   runs on the caller's stream
//
// Host malloc/memcpy/free lives in `cuda_host.cpp`. This file only needs the
// CUDA runtime headers.
//
// `<<<>>>` / `blockIdx` compile under nvcc only (`__NVCC__`). clangd treats
// `.cu` as C++ (see /.clangd) and takes the host loop below.

#define SHIELD_CUDA_KERNEL_TU
#include "cuda_runtime_compat.h"

static void clamp_one(const float* input, const float* limit, float* output, size_t i) {
    float x = input[i];
    float l = limit[i];
    if (x > l) {
        x = l;
    }
    if (x < -l) {
        x = -l;
    }
    output[i] = x;
}

#ifdef __NVCC__
extern "C" __global__ void clamp_kernel(
    const float* __restrict__ device_input,
    const float* __restrict__ device_limit,
    float* __restrict__ device_output,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    clamp_one(device_input, device_limit, device_output, i);
}
#endif

extern "C" int shield_cuda_launch_clamp(
    const float* device_input,
    const float* device_limit,
    float* device_output,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) {
        return 0;
    }
#ifdef __NVCC__
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    clamp_kernel<<<blocks, threads, 0, stream>>>(
        device_input, device_limit, device_output, n);
    cudaError_t err = cudaGetLastError();
    return static_cast<int>(err);
#else
    (void)stream;
    for (size_t i = 0; i < n; ++i) {
        clamp_one(device_input, device_limit, device_output, i);
    }
    return 0;
#endif
}
