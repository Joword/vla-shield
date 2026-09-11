//! CUDA runtime header for nvcc *and* editor clang.
//!
//! nvcc gets the real `<cuda_runtime.h>`. clangd / host C++ never include it
//! (or libc) — Windows clangd usually has no MSVC CRT on the include path.
//! `<<<>>>` / `blockIdx` stay behind `__NVCC__`; the editor uses the stubs.

#pragma once

#ifdef __NVCC__
#include <cuda_runtime.h>
#define SHIELD_CUDA_RUNTIME 1
#endif

#ifndef SHIELD_CUDA_RUNTIME

// No <stdlib.h> / <stddef.h> here. clangd on this machine can't see the CRT.
#if defined(__SIZE_TYPE__)
typedef __SIZE_TYPE__ size_t;
#elif defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
typedef unsigned long long size_t;
#else
typedef unsigned long size_t;
#endif

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

#ifndef SHIELD_CUDA_KERNEL_TU
extern "C" void* malloc(size_t);
extern "C" void free(void*);
extern "C" void* memcpy(void*, const void*, size_t);

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
    *s = reinterpret_cast<cudaStream_t>(static_cast<size_t>(1));
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
#endif  // !SHIELD_CUDA_KERNEL_TU

inline cudaError_t cudaGetLastError() { return cudaSuccess; }

#endif  // !SHIELD_CUDA_RUNTIME
