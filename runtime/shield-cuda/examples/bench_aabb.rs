//! Micro-bench: Rust nested-loop AABB vs `aabb_overlap_mask` C backend.
//!
//! ```text
//! cd runtime
//! cargo run -p shield-cuda --example bench_aabb --release -- --iters 20000 --n-links 8 --n-obs 8
//! cargo run -p shield-cuda --example bench_aabb --release -- --iters 5000 --n-links 32 --n-obs 32
//! ```
//!
//! 8×8 = 64 pairs is the GPU threshold in `cuda_host.cpp`. Smaller scenes stay
//! on the host loop even when the crate was built with nvcc.

use std::time::Instant;

use shield_cuda::{aabb_overlap_mask, is_cuda_kernel_enabled};

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
    n_links: usize,
    n_obs: usize,
}

fn parse_args() -> Args {
    let mut iters = 20_000_usize;
    let mut n_links = 8_usize;
    let mut n_obs = 8_usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
            "--n-links" => n_links = args.next().and_then(|v| v.parse().ok()).unwrap_or(n_links),
            "--n-obs" => n_obs = args.next().and_then(|v| v.parse().ok()).unwrap_or(n_obs),
            _ => {}
        }
    }
    Args {
        iters,
        n_links,
        n_obs,
    }
}

fn make_boxes(n: usize, stride: f32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut mins = vec![[0.0_f32; 3]; n];
    let mut maxs = vec![[0.0_f32; 3]; n];
    for i in 0..n {
        let x = i as f32 * stride;
        mins[i] = [x, 0.0, 0.0];
        maxs[i] = [x + 1.5, 1.0, 1.0];
    }
    (mins, maxs)
}

fn cpu_mask(
    link_min: &[[f32; 3]],
    link_max: &[[f32; 3]],
    obs_min: &[[f32; 3]],
    obs_max: &[[f32; 3]],
    out: &mut [u8],
) {
    let n = link_min.len();
    let m = obs_min.len();
    for i in 0..n {
        for j in 0..m {
            let overlap = link_min[i][0] <= obs_max[j][0]
                && link_max[i][0] >= obs_min[j][0]
                && link_min[i][1] <= obs_max[j][1]
                && link_max[i][1] >= obs_min[j][1]
                && link_min[i][2] <= obs_max[j][2]
                && link_max[i][2] >= obs_min[j][2];
            out[i * m + j] = u8::from(overlap);
        }
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

fn report(label: &str, data: &mut [f64]) {
    let p50 = percentile(data, 50.0);
    let p95 = percentile(data, 95.0);
    let p99 = percentile(data, 99.0);
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let max = data.iter().cloned().fold(f64::MIN, f64::max);
    println!("{label}");
    println!("  p50={p50:.2}us  p95={p95:.2}us  p99={p99:.2}us  mean={mean:.2}us  max={max:.2}us");
}

fn main() {
    let args = parse_args();
    let pairs = args.n_links * args.n_obs;
    println!(
        "shield-cuda aabb bench   compiled={}   iters={}   links={}   obs={}   pairs={}",
        if is_cuda_kernel_enabled() {
            "CUDA kernel"
        } else {
            "CPU stub"
        },
        args.iters,
        args.n_links,
        args.n_obs,
        pairs
    );

    let (link_min, link_max) = make_boxes(args.n_links, 2.0);
    let (obs_min, obs_max) = make_boxes(args.n_obs, 2.0);
    let mut cpu_out = vec![0u8; pairs];

    cpu_mask(&link_min, &link_max, &obs_min, &obs_max, &mut cpu_out);
    let backend = aabb_overlap_mask(&link_min, &link_max, &obs_min, &obs_max).unwrap();
    assert_eq!(cpu_out, backend, "CPU loop and C ABI disagree");

    let mut cpu_samples = time_loop(args.iters, || {
        cpu_mask(&link_min, &link_max, &obs_min, &obs_max, &mut cpu_out);
    });
    let mut backend_samples = time_loop(args.iters, || {
        let _ = aabb_overlap_mask(&link_min, &link_max, &obs_min, &obs_max).unwrap();
    });

    println!();
    report("Rust nested loop", &mut cpu_samples);
    println!();
    report("aabb_overlap_mask (C ABI / kernel or stub)", &mut backend_samples);

    let cpu_p50 = percentile(&mut cpu_samples.clone(), 50.0);
    let backend_p50 = percentile(&mut backend_samples.clone(), 50.0);
    if backend_p50 > 0.0 {
        println!(
            "\nCPU loop vs C ABI  median speedup : {:.2}x  ({})",
            backend_p50 / cpu_p50,
            if cpu_p50 < backend_p50 {
                "CPU loop faster — expected on stub / tiny scenes"
            } else {
                "C ABI faster — expected for dense N×M on a real GPU"
            }
        );
    }
}
