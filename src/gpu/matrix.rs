use std::rc::Rc;

use tokio::sync::oneshot;
use wgpu::{BufferUsages, util::DeviceExt, wgt};

use crate::gpu::{kernels::GpuKernels, state::GpuState, utils::GpuContext};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dimensions {
    m: u32,
    n: u32,
    k: u32,
}

pub struct GpuMatrix {
    gpu_state: Rc<GpuState>,
    gpu_buffer: wgpu::Buffer,
    rows: usize,
    columns: usize,
}

impl GpuMatrix {
    pub fn new(rows: usize, columns: usize, data: &Vec<f32>, gpu_state: Rc<GpuState>) -> GpuMatrix {
        assert_eq!(
            data.len(),
            rows * columns,
            "Insufficient data to create the matrix"
        );
        let byte_data: &[u8] = bytemuck::cast_slice(data);
        let buffer =
            gpu_state
                .gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("matrix_buffer_{rows}x{columns}")),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    contents: byte_data,
                });

        GpuMatrix {
            gpu_state,
            gpu_buffer: buffer,
            rows,
            columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub async fn to_cpu(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let download_buffer =
            self.gpu_state
                .gpu_context
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!(
                        "matrix_buffer_{}x{}_download",
                        self.rows, self.columns
                    )),
                    size: self.gpu_buffer.size(),
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
        let mut encoder = self.gpu_state.gpu_context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("matrix_download_encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            &self.gpu_buffer,
            0,
            &download_buffer,
            0,
            download_buffer.size(),
        );

        let command_buffer = encoder.finish();
        self.gpu_state.gpu_context.queue.submit([command_buffer]);
        let (tx, rx) = oneshot::channel();
        download_buffer.map_async(wgpu::MapMode::Read, .., |x| {
            let _ = tx.send(x);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.gpu_state
            .gpu_context
            .device
            .poll(wgpu::wgt::PollType::wait_indefinitely())?;

        rx.await??;
        let gpu_data = download_buffer.get_mapped_range(..)?;
        Ok(bytemuck::allocation::pod_collect_to_vec(&gpu_data))
    }

    pub fn matmul(a: &GpuMatrix, b: &GpuMatrix, result: &GpuMatrix) {
        debug_assert_eq!(a.columns, b.rows);
        debug_assert_eq!((a.rows, b.columns), (result.rows, result.columns));
        let kernel = &a.gpu_state.kernels.matmul;
        let layout = kernel.get_bind_group_layout(0);
        let dimensions = Dimensions {
            m: a.rows as u32,
            n: b.columns as u32,
            k: a.columns as u32,
        };
        let dimension_buffer =
            a.gpu_state
                .gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("buffer_uniform_matmul"),
                    contents: bytemuck::cast_slice(&[dimensions]),
                    usage: BufferUsages::UNIFORM,
                });
        let bind_group =
            a.gpu_state
                .gpu_context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bind_group_matmul"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: dimension_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: a.gpu_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: b.gpu_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: result.gpu_buffer.as_entire_binding(),
                        },
                    ],
                });
        let mut encoder =
            a.gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgt::CommandEncoderDescriptor {
                    label: Some("command_encoder_matmul"),
                });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass_matmul"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(kernel);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let x = b.columns.div_ceil(16) as u32;
        let y = a.rows.div_ceil(16) as u32;
        compute_pass.dispatch_workgroups(x, y, 1);

        drop(compute_pass);

        let command_buffer = encoder.finish();

        a.gpu_state.gpu_context.queue.submit([command_buffer]);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::utils::init_gpu;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn download_to_cpu() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        let gpu_state = Rc::new(GpuState::default().await);
        let mat_data = vec![0.0, 1.0, 2.0, 1.0];
        let m = GpuMatrix::new(2, 2, &mat_data, gpu_state.clone());

        let downloaded_data = m.to_cpu().await.expect("Download to cpu failed");

        assert_eq!(downloaded_data, mat_data);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matmul() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let a_data = vec![1.0, 0.0, 3.0, 2.0];
        let b_data = vec![1.0, 0.0, 3.0, 2.0, 4.0, 2.0];
        let a = GpuMatrix::new(2, 2, &a_data, gpu_state.clone());
        let b = GpuMatrix::new(2, 3, &b_data, gpu_state.clone());

        let result_expected = vec![1.0, 0.0, 3.0, 7.0, 8.0, 13.0];
        let result = GpuMatrix::new(2, 3, &vec![1.0; 6], gpu_state.clone());
        GpuMatrix::matmul(&a, &b, &result);
        let result_actual = result.to_cpu().await.unwrap();
        assert_eq!(result_expected, result_actual);
    }

    // Non-square, larger than one workgroup (16x16) in both directions.
    // This test catches a swapped-dispatch-axis bug that small square
    // tests hide, since div_ceil(16) rounds to 1 either way for small sizes.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matmul_large_non_square() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // a: 20x5, b: 5x30, result: 20x30
        let rows = 20;
        let inner = 5;
        let cols = 30;

        let a_data: Vec<f32> = (0..rows * inner).map(|i| (i % 7) as f32).collect();
        let b_data: Vec<f32> = (0..inner * cols).map(|i| (i % 5) as f32).collect();

        let a = GpuMatrix::new(rows, inner, &a_data, gpu_state.clone());
        let b = GpuMatrix::new(inner, cols, &b_data, gpu_state.clone());
        let result = GpuMatrix::new(rows, cols, &vec![0.0; rows * cols], gpu_state.clone());

        GpuMatrix::matmul(&a, &b, &result);
        let result_actual = result.to_cpu().await.unwrap();

        // CPU reference implementation, plain and simple.
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut sum = 0.0f32;
                for k in 0..inner {
                    sum += a_data[r * inner + k] * b_data[k * cols + c];
                }
                expected[r * cols + c] = sum;
            }
        }

        assert_eq!(result_actual, expected);
    }

    // Multiplying by an identity matrix should return the original matrix.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matmul_identity() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let identity_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3

        let a = GpuMatrix::new(2, 3, &a_data, gpu_state.clone());
        let identity = GpuMatrix::new(3, 3, &identity_data, gpu_state.clone());
        let result = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());

        GpuMatrix::matmul(&a, &identity, &result);
        let result_actual = result.to_cpu().await.unwrap();

        assert_eq!(result_actual, a_data);
    }

    // rows() and columns() should report the values passed to new().
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn dimensions_are_reported_correctly() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let m = GpuMatrix::new(4, 7, &vec![0.0; 28], gpu_state.clone());
        assert_eq!(m.rows(), 4);
        assert_eq!(m.columns(), 7);
    }
}
