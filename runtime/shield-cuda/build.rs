use std::process::Command;

fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/clamp_kernel.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda_host.cpp");
    println!("cargo:rerun-if-changed=src/kernels/clamp_stub.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_DISABLE");
    // Declare custom cfg flag so rustc 1.80+ does not warn on cfg!(has_cuda_kernel).
    println!("cargo:rustc-check-cfg=cfg(has_cuda_kernel)");

    if std::env::var_os("CUDA_DISABLE").is_some() {
        compile_fallback();
        println!("cargo:warning=shield-cuda: CUDA_DISABLE set, forcing CPU fallback backend");
        return;
    }

    if has_nvcc() {
        // Two-unit build:
        //   * clamp_kernel.cu  – pure __global__ kernel (compiled by nvcc as device code)
        //   * cuda_host.cpp    – C++ host-side glue (cudaMalloc / Memcpy / Free)
        // Both are passed to nvcc so cudart linkage is set up automatically.
        let mut build = cc::Build::new();
        build.cuda(true);
        build.cpp(true);
        build.file("src/kernels/clamp_kernel.cu");
        build.file("src/kernels/cuda_host.cpp");
        build.compile("shield_cuda_kernels");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-cfg=has_cuda_kernel");
        println!(
            "cargo:warning=shield-cuda: nvcc found, building CUDA kernel + C++ host backend"
        );
    } else {
        compile_fallback();
        println!(
            "cargo:warning=shield-cuda: nvcc not found, building CPU fallback backend"
        );
    }
}

fn compile_fallback() {
    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("src/kernels/clamp_stub.cpp");
    build.compile("shield_cuda_kernels");
}
