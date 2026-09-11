//! CUDA runtime include for both nvcc and editor clang.
//!
//! `nvcc` always has `<cuda_runtime.h>`.  clangd often does not, and when it
//! treats `.cu` as CUDA it also demands libdevice / a CUDA-wrapped libstdc++.
//! Device syntax (`<<<>>>`, `blockIdx`) is compiled only under `__NVCC__`;
//! the editor and host C++ compilers use the stubs below.

#pragma once

#ifdef __NVCC__
#include <cuda_runtime.h>
#define SHIELD_CUDA_RUNTIME 1
#elif !defined(__CUDACC__) && defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#    define SHIELD_CUDA_RUNTIME 1
#  endif
#endif

#ifndef SHIELD_CUDA_RUNTIME

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
typedef void* cudaStream_t;

enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2
};

#ifndef __CUDACC__
#define __global__
#define __device__
#define __host__
#ifndef __restrict__
#define __restrict__
#endif
#endif

struct dim3 {
    unsigned x, y, z;
};

inline cudaError_t cudaMalloc(void** p, size_t n) {
    if (p == nullptr) {
        return cudaErrorInvalidValue;
    }
    *p = malloc(n);
    return *p ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t cudaFree(void* p) {
    free(p);
    return cudaSuccess;
}

inline cudaError_t cudaMallocHost(void** p, size_t n) { return cudaMalloc(p, n); }
inline cudaError_t cudaFreeHost(void* p) { return cudaFree(p); }

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (dst == nullptr || src == nullptr) {
        return cudaErrorInvalidValue;
    }
    memcpy(dst, src, n);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(
    void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
}

inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    if (s == nullptr) {
        return cudaErrorInvalidValue;
    }
    *s = reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(1));
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

#endif  // !SHIELD_CUDA_RUNTIME
