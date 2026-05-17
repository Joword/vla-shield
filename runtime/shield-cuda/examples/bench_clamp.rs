//! Tiny built-in micro-benchmark comparing the stateless one-shot path vs.
//! the cached-context path.
//!
//! Run with:
//!
//! ```text
//! cd runtime
//! cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
//! ```
//!
//! The headline number to watch is the per-call median in microseconds; the
//! cached-context path should be **dramatically** faster on the GPU backend
//! (no per-call `cudaMalloc`, async copies via pinned memory) and *slightly*
//! faster on the CPU fallback (skips the per-call allocation of `Vec` plus
//! `cudaMalloc`-mimicking bookkeeping).

use std::time::Instant;

use shield_cuda::{clamp_action_cuda, is_cuda_kernel_enabled, CudaCtx};

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

fn parse_args() -> (usize, usize) {
    let mut iters = 50_000_usize;
    let mut dof = 8_usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
            "--dof" => dof = args.next().and_then(|v| v.parse().ok()).unwrap_or(dof),
            _ => {}
        }
    }
    (iters, dof)
}

fn main() {
    let (iters, dof) = parse_args();
    println!(
        "shield-cuda bench   backend={}   iters={iters}   dof={dof}",
        if is_cuda_kernel_enabled() { "CUDA" } else { "CPU fallback" }
    );

    let input: Vec<f32> = (0..dof).map(|i| ((i as f32) - (dof as f32) / 2.0) * 3.0).collect();
    let limit: Vec<f32> = vec![1.0; dof];

    // Warmup both paths.
    let _ = clamp_action_cuda(&input, &limit).unwrap();
    let mut ctx = CudaCtx::new(dof).expect("ctx_create");
    let mut out_buf = vec![0.0_f32; dof];
    ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();

    // Stateless one-shot
    let mut stateless = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = clamp_action_cuda(&input, &limit).unwrap();
        stateless.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // Cached context, clamp_into (no allocations on hot path)
    let mut cached = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        ctx.clamp_into(&input, &limit, &mut out_buf).unwrap();
        cached.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    println!("\nstateless one-shot  (allocates / frees per call)");
    report(&mut stateless);
    println!("\ncached context     (reuses device + pinned buffers + stream)");
    report(&mut cached);

    let speedup = percentile(&mut stateless.clone(), 50.0) / percentile(&mut cached.clone(), 50.0);
    println!("\nmedian speedup    : {:.2}x", speedup);
}

fn report(data: &mut Vec<f64>) {
    let p50 = percentile(data, 50.0);
    let p95 = percentile(data, 95.0);
    let p99 = percentile(data, 99.0);
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let max = data.iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "  p50={p50:.2}us  p95={p95:.2}us  p99={p99:.2}us  mean={mean:.2}us  max={max:.2}us",
        p50 = p50,
        p95 = p95,
        p99 = p99,
        mean = mean,
        max = max
    );
}
