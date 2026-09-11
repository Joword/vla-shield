use std::process::Command;

fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// On Windows, nvcc wants MSVC `cl.exe`. A CUDA toolkit + MinGW isn't enough —
/// nvcc looks for cl.exe, fails, and used to take the whole workspace with it.
fn nvcc_can_compile() -> bool {
    if !has_nvcc() {
        return false;
    }
    if cfg!(windows) {
        let cl = Command::new("where")
            .arg("cl")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !cl {
            println!(
                "cargo:warning=shield-cuda: nvcc found but cl.exe is missing; using CPU stub"
            );
            return false;
        }
    }
    true
}

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/clamp_kernel.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda_host.cpp");
    println!("cargo:rerun-if-changed=src/kernels/clamp_stub.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_DISABLE");
    // Tell rustc 1.80+ that `has_cuda_kernel` is a real cfg, not a typo.
    println!("cargo:rustc-check-cfg=cfg(has_cuda_kernel)");

    if std::env::var_os("CUDA_DISABLE").is_some() {
        compile_fallback();
        println!("cargo:warning=shield-cuda: CUDA_DISABLE set, forcing CPU fallback backend");
        return;
    }

    if nvcc_can_compile() {
        // Two-unit build:
        //   * clamp_kernel.cu  – __global__ kernel (nvcc, device)
        //   * cuda_host.cpp    – host glue (cudaMalloc / Memcpy / Free)
        // Both go through nvcc so cudart linkage just happens.
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
            "cargo:warning=shield-cuda: nvcc not usable, building CPU fallback backend"
        );
    }
}

fn compile_fallback() {
    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("src/kernels/clamp_stub.cpp");
    build.compile("shield_cuda_kernels");
}
