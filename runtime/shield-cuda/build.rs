use std::path::PathBuf;
use std::process::Command;

fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"))
}

/// Ask nvcc itself (no `cc` `-ccbin=g++` rewrite). On Windows the default host
/// is MSVC `cl.exe`; a miss is a handled exception, not a crate build error.
fn probe_nvcc_host() -> Result<(), String> {
    let obj = out_dir().join("nvcc_host_probe.obj");
    let output = Command::new("nvcc")
        .args([
            "-c",
            "src/kernels/aabb_kernel.cu",
            "-o",
        ])
        .arg(&obj)
        .output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let raw = format!("{stderr}{stdout}");
    Err(raw)
}

/// Turn nvcc / cc failures into a short reason. Missing `cl.exe` is expected
/// on MinGW Windows — handled exception, crate still builds.
fn describe_cuda_failure(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("cl.exe") || lower.contains("cannot find compiler") {
        "nvcc host compiler cl.exe missing (handled exception -> CPU stub)".into()
    } else if cfg!(windows)
        && (lower.contains("nvcc") || lower.contains("ccbin") || lower.contains("toolexecerror"))
    {
        "nvcc host compile failed on Windows (cl.exe / unsupported ccbin; handled exception -> CPU stub)"
            .into()
    } else {
        let first = raw
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or(raw)
            .trim();
        let snippet: String = first.chars().take(180).collect();
        format!("nvcc compile failed (handled exception -> CPU stub): {snippet}")
    }
}

fn try_compile_cuda() -> Result<(), String> {
    let mut build = cc::Build::new();
    build.cuda(true);
    build.cpp(true);
    build.file("src/kernels/clamp_kernel.cu");
    build.file("src/kernels/aabb_kernel.cu");
    build.file("src/kernels/cuda_host.cpp");
    build
        .try_compile("shield_cuda_kernels")
        .map_err(|e| describe_cuda_failure(&e.to_string()))
}

fn compile_fallback() {
    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("src/kernels/clamp_stub.cpp");
    build.compile("shield_cuda_kernels");
    println!("cargo:rustc-cfg=cuda_cpu_stub");
}

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/clamp_kernel.cu");
    println!("cargo:rerun-if-changed=src/kernels/aabb_kernel.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda_host.cpp");
    println!("cargo:rerun-if-changed=src/kernels/clamp_stub.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_DISABLE");
    println!("cargo:rustc-check-cfg=cfg(has_cuda_kernel)");
    println!("cargo:rustc-check-cfg=cfg(cuda_cpu_stub)");

    if std::env::var_os("CUDA_DISABLE").is_some() {
        compile_fallback();
        println!("cargo:warning=shield-cuda: CUDA_DISABLE set, forcing CPU fallback backend");
        return;
    }

    if !has_nvcc() {
        compile_fallback();
        println!("cargo:warning=shield-cuda: nvcc not on PATH, building CPU fallback backend");
        return;
    }

    // Probe first so a missing cl.exe never reaches cc::Build::compile (panic).
    if let Err(raw) = probe_nvcc_host() {
        println!("cargo:warning=shield-cuda: {}", describe_cuda_failure(&raw));
        compile_fallback();
        return;
    }

    match try_compile_cuda() {
        Ok(()) => {
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-cfg=has_cuda_kernel");
            println!(
                "cargo:warning=shield-cuda: nvcc ok, building CUDA kernel + C++ host backend"
            );
        }
        Err(reason) => {
            println!("cargo:warning=shield-cuda: {reason}");
            compile_fallback();
        }
    }
}
