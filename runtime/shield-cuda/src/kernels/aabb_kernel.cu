// AABB overlap kernel: N link boxes vs M obstacles.
//
// Packed as xyz-min / xyz-max (3 floats per box). Output is a row-major
// N×M mask: mask[i * n_obs + j] = 1 if link i overlaps obstacle j.
//
// Device syntax is nvcc-only (`__NVCC__`). clangd parses this as C++.

#define SHIELD_CUDA_KERNEL_TU
#include "cuda_runtime_compat.h"

static int boxes_overlap(
    const float* a0,
    const float* a1,
    const float* b0,
    const float* b1
) {
    return a0[0] <= b1[0] && a1[0] >= b0[0] &&
           a0[1] <= b1[1] && a1[1] >= b0[1] &&
           a0[2] <= b1[2] && a1[2] >= b0[2];
}

static void fill_aabb_mask(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* mask
) {
    for (size_t i = 0; i < n_links; ++i) {
        const float* a0 = link_min + i * 3;
        const float* a1 = link_max + i * 3;
        for (size_t j = 0; j < n_obs; ++j) {
            const float* b0 = obs_min + j * 3;
            const float* b1 = obs_max + j * 3;
            mask[i * n_obs + j] = boxes_overlap(a0, a1, b0, b1) ? 1 : 0;
        }
    }
}

#ifdef __NVCC__
// One thread per (link, obstacle) pair so occupancy tracks N×M, not N.
extern "C" __global__ void aabb_mask_kernel(
    const float* __restrict__ link_min,
    const float* __restrict__ link_max,
    size_t n_links,
    const float* __restrict__ obs_min,
    const float* __restrict__ obs_max,
    size_t n_obs,
    unsigned char* __restrict__ mask
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = n_links * n_obs;
    if (idx >= n) {
        return;
    }
    size_t i = idx / n_obs;
    size_t j = idx - i * n_obs;
    const float* a0 = link_min + i * 3;
    const float* a1 = link_max + i * 3;
    const float* b0 = obs_min + j * 3;
    const float* b1 = obs_max + j * 3;
    mask[idx] = boxes_overlap(a0, a1, b0, b1) ? 1 : 0;
}
#endif

extern "C" int shield_cuda_launch_aabb_mask(
    const float* device_link_min,
    const float* device_link_max,
    size_t n_links,
    const float* device_obs_min,
    const float* device_obs_max,
    size_t n_obs,
    unsigned char* device_mask,
    cudaStream_t stream
) {
    if (n_links == 0 || n_obs == 0) {
        return 0;
    }
#ifdef __NVCC__
    const size_t pairs = n_links * n_obs;
    const int threads = 256;
    const int blocks = static_cast<int>((pairs + threads - 1) / threads);
    aabb_mask_kernel<<<blocks, threads, 0, stream>>>(
        device_link_min,
        device_link_max,
        n_links,
        device_obs_min,
        device_obs_max,
        n_obs,
        device_mask);
    return static_cast<int>(cudaGetLastError());
#else
    (void)stream;
    fill_aabb_mask(
        device_link_min,
        device_link_max,
        n_links,
        device_obs_min,
        device_obs_max,
        n_obs,
        device_mask);
    return 0;
#endif
}

// Host-side fill used by cuda_host.cpp when N×M is too small to launch.
extern "C" void shield_cuda_aabb_mask_host(
    const float* link_min,
    const float* link_max,
    size_t n_links,
    const float* obs_min,
    const float* obs_max,
    size_t n_obs,
    unsigned char* mask
) {
    fill_aabb_mask(link_min, link_max, n_links, obs_min, obs_max, n_obs, mask);
}
