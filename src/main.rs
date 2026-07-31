use tokio::runtime;

use crate::gpu::matrix::GpuMatrix;
use crate::gpu::state::GpuState;

mod gpu;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use std::{rc::Rc, time::Instant};

    use wgpu::wgt::PollType;
    let rt = runtime::Runtime::new().unwrap();
    let gpu_state = Rc::new(rt.block_on(GpuState::default()));
    let (m, n, k) = (1024, 1024, 1024);

    let (a_data, b_data) = random_matmul_inputs(m, n, k);

    let a = GpuMatrix::new(m, k, &a_data, gpu_state.clone());
    let b = GpuMatrix::new(k, n, &b_data, gpu_state.clone());
    let result = GpuMatrix::new(m, n, &vec![1.0; m * n], gpu_state.clone());

    let start = Instant::now();
    GpuMatrix::matmul(&a, &b, &result);
    gpu_state
        .gpu_context
        .device
        .poll(PollType::wait_indefinitely())
        .unwrap();
    let elapsed = start.elapsed();

    println!("Multiplying {m}x{k} x {k}x{n}");
    println!("Time: {elapsed:?}");
}

fn random_matmul_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a_data: Vec<f32> = (0..m * k).map(|_| rand::random_range(-1.0..1.0)).collect();
    let b_data: Vec<f32> = (0..k * n).map(|_| rand::random_range(-1.0..1.0)).collect();

    (a_data, b_data)
}
