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

#include <cuda_runtime.h>
#include <stddef.h>

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
    float x = device_input[i];
    float l = device_limit[i];
    if (x > l) x = l;
    if (x < -l) x = -l;
    device_output[i] = x;
}

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
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    clamp_kernel<<<blocks, threads, 0, stream>>>(
        device_input, device_limit, device_output, n);
    cudaError_t err = cudaGetLastError();
    return static_cast<int>(err);
}
