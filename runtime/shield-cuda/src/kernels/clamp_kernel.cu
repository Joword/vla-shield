// CUDA kernel for symmetric per-dimension action clamping.
//
// Pure device-side code:
//   * `clamp_kernel`     – __global__ entry that each thread runs once per element.
//   * `shield_cuda_launch_clamp` – C-ABI launcher that takes already-on-device
//                                  pointers and triggers the kernel on the
//                                  caller's stream.
//
// Host-side memory management (cudaMalloc / cudaMemcpy / cudaFree) lives in the
// neighbouring `cuda_host.cpp`, so this translation unit only depends on the
// CUDA runtime headers and contains no allocation logic.
//
// Device syntax (`<<<>>>`, `blockIdx`) is compiled only by nvcc (`__NVCC__`).
// clangd parses `.cu` as C++ (see /.clangd) and uses the host loop below.

#include "cuda_runtime_compat.h"
#include <stddef.h>

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
