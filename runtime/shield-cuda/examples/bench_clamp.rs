//! Micro-benchmark: CPU bypass vs cached C backend vs stateless one-shot.
//!
//! Run with:
//!
//! ```text
//! cd runtime
//! cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
//! cargo run -p shield-cuda --example bench_clamp --release -- --iters 20000 --dof 256
//! ```
//!
//! Default `--dof 8` is a typical VLA arm.  The cached-context path then
//! stays on the CPU (`n < DEFAULT_MIN_GPU_N`).  Pass `--force-gpu` (or
//! `--dof` ≥ 64) to force the C backend for an A/B comparison.

use std::time::Instant;

use shield_cuda::{
    clamp_action_cuda, clamp_cpu, is_cuda_kernel_enabled, ClampPath, CudaCtx, DEFAULT_MIN_GPU_N,
};

fn percentile(data: &mut [f64], p: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = ((data.len() - 1) as f64) * p / 100.0;
    let f = k.floor() as usize;
    let c = (f + 1).min(data.len() - 1);
    data[f] + (k - f as f64) * (data[c] - data[f])
}

struct Args {
    iters: usize,
    dof: usize,
    force_gpu: bool,
}

fn parse_args() -> Args {
    let mut iters = 50_000_usize;
    let mut dof = 8_usize;
    let mut force_gpu = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
            "--dof" => dof = args.next().and_then(|v| v.parse().ok()).unwrap_or(dof),
            "--force-gpu" => force_gpu = true,
            _ => {}
        }
    }
    Args {
        iters,
        dof,
        force_gpu,
    }
}

fn time_loop<F: FnMut()>(iters: usize, mut body: F) -> Vec<f64> {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        body();
        samples.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    samples
}

fn main() {
    let args = parse_args();
    let iters = args.iters;
    let dof = args.dof;
    println!(
        "shield-cuda bench   compiled={}   iters={iters}   dof={dof}   default_min_gpu_n={}",
        if is_cuda_kernel_enabled() {
            "CUDA kernel"
        } else {
            "CPU stub"
        },
        DEFAULT_MIN_GPU_N
    );

    let input: Vec<f32> = (0..dof)
        .map(|i| ((i as f32) - (dof as f32) / 2.0) * 3.0)
        .collect();
    let limit: Vec<f32> = vec![1.0; dof];
    let mut out_buf = vec![0.0_f32; dof];

    // Warmup.
    clamp_cpu(&input, &limit, &mut out_buf);
    let _ = clamp_action_cuda(&input, &limit).unwrap();

    let mut cpu_ctx = CudaCtx::new(dof).expect("ctx_create");
    cpu_ctx.set_min_gpu_n(usize::MAX); // never use backend
    cpu_ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();

    let mut gpu_ctx = CudaCtx::new(dof).expect("ctx_create");
    gpu_ctx.set_min_gpu_n(0); // always use backend
    gpu_ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();

    let mut default_ctx = CudaCtx::new(dof).expect("ctx_create");
    if args.force_gpu {
        default_ctx.set_min_gpu_n(0);
    }
    default_ctx
        .clamp_into(&input, &limit, &mut out_buf)
        .unwrap();

    let mut cpu_samples = time_loop(iters, || {
        cpu_ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();
    });
    assert_eq!(cpu_ctx.last_path(), ClampPath::Cpu);

    let mut gpu_samples = time_loop(iters, || {
        gpu_ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();
    });
    assert_eq!(gpu_ctx.last_path(), ClampPath::Backend);

    let mut default_samples = time_loop(iters, || {
        default_ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();
    });

    let mut stateless = time_loop(iters, || {
        let _ = clamp_action_cuda(&input, &limit).unwrap();
    });

    println!(
        "\nCPU bypass          (scalar loop, min_gpu_n=MAX)     last={:?}",
        cpu_ctx.last_path()
    );
    report(&mut cpu_samples);

    println!(
        "\nforced backend      (C ABI / kernel, min_gpu_n=0)    last={:?}",
        gpu_ctx.last_path()
    );
    report(&mut gpu_samples);

    println!(
        "\ndefault ctx         (min_gpu_n={}, force_gpu={}) last={:?}",
        default_ctx.min_gpu_n(),
        args.force_gpu,
        default_ctx.last_path()
    );
    report(&mut default_samples);

    println!("\nstateless one-shot  (allocates / frees per call, always C ABI)");
    report(&mut stateless);

    let cpu_p50 = percentile(&mut cpu_samples.clone(), 50.0);
    let gpu_p50 = percentile(&mut gpu_samples.clone(), 50.0);
    if gpu_p50 > 0.0 {
        println!(
            "\nCPU bypass vs forced backend  median speedup : {:.2}x  ({})",
            gpu_p50 / cpu_p50,
            if cpu_p50 < gpu_p50 {
                "CPU faster — expected for small DoF"
            } else {
                "backend faster — expected for large n / real GPU"
            }
        );
    }
}

fn report(data: &mut Vec<f64>) {
    let p50 = percentile(data, 50.0);
    let p95 = percentile(data, 95.0);
    let p99 = percentile(data, 99.0);
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let max = data.iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "  p50={p50:.2}us  p95={p95:.2}us  p99={p99:.2}us  mean={mean:.2}us  max={max:.2}us"
    );
}
