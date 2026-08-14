use std::rc::Rc;

use tokio::sync::oneshot;
use wgpu::{BufferUsages, util::DeviceExt};

use crate::gpu::state::GpuState;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDimensions {
    m: u32,
    n: u32,
    k: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InplaceAddUniform {
    columns_a: u32,
    is_broadcast: u32,
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
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    contents: byte_data,
                });

        GpuMatrix {
            gpu_state,
            gpu_buffer: buffer,
            rows,
            columns,
        }
    }

    pub fn empty(rows: usize, columns: usize, gpu_state: Rc<GpuState>) -> GpuMatrix {
        let buffer = gpu_state
            .gpu_context
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("matrix_buffer_{rows}x{columns}")),
                size: (rows * columns * std::mem::size_of::<f32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
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

    pub fn shape(&self) -> (usize, usize) {
        (self.columns, self.rows)
    }

    pub fn gpu_buffer(&self) -> &wgpu::Buffer {
        &self.gpu_buffer
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

    fn record_compute_pass(
        encoder: &mut wgpu::CommandEncoder,
        kernel: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass_matrix"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(kernel);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    pub fn matmul(
        a: &GpuMatrix,
        b: &GpuMatrix,
        result: &GpuMatrix,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(a.columns, b.rows);
        assert_eq!((a.rows, b.columns), (result.rows, result.columns));
        let kernel = &a.gpu_state.kernels.matmul;
        let layout = kernel.get_bind_group_layout(0);
        let dimensions = MatmulDimensions {
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
        let x = b.columns.div_ceil(16) as u32;
        let y = a.rows.div_ceil(16) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, y, 1));
    }

    pub fn hadamard(
        a: &GpuMatrix,
        b: &GpuMatrix,
        result: &GpuMatrix,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(b.shape(), result.shape());

        let kernel = &a.gpu_state.kernels.hadamard;
        let layout = kernel.get_bind_group_layout(0);
        let device = &a.gpu_state.gpu_context.device;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_hadamard"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = (a.rows * a.columns).div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn add_assign(&self, rhs: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        let shapes_match = self.shape() == rhs.shape();
        let row_broadcast = self.rows == rhs.rows && rhs.columns == 1;
        assert!(
            shapes_match || row_broadcast,
            "Either shapes must match or rhs must have one column to do row broadcasting"
        );

        let device = &self.gpu_state.gpu_context.device;
        let uniforms = InplaceAddUniform {
            columns_a: self.columns as u32,
            is_broadcast: row_broadcast as u32,
        };
        let dimension_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer_uniform_inplace_add"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: BufferUsages::UNIFORM,
        });

        let kernel = &self.gpu_state.kernels.inplace_add;
        let bind_group_layout = kernel.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_inplace_add"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dimension_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rhs.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = (self.rows * self.columns).div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn sigmoid(&self, result: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.shape(), result.shape());
        let device = &self.gpu_state.gpu_context.device;
        let kernel = &self.gpu_state.kernels.sigmoid;

        let bind_group_layout = kernel.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_sigmoid"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });
        let x = (self.rows * self.columns).div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn sigmoid_prime(&self, result: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.shape(), result.shape());
        let device = &self.gpu_state.gpu_context.device;
        let kernel = &self.gpu_state.kernels.sigmoid_prime;

        let bind_group_layout = kernel.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_sigmoid_prime"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });
        let x = (self.rows * self.columns).div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_matmul"),
                });
        GpuMatrix::matmul(&a, &b, &result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_matmul_large_non_square"),
                });
        GpuMatrix::matmul(&a, &b, &result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_matmul_identity"),
                });
        GpuMatrix::matmul(&a, &identity, &result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_inplace_add() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let mut mat_a = GpuMatrix::new(
            3,
            3,
            &vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 9.0, 1.0, 0.0],
            Rc::clone(&gpu_state),
        );
        let mat_b = GpuMatrix::new(
            3,
            3,
            &vec![4.0, 1.0, 8.0, 7.0, 1.0, 3.0, 2.0, 9.0, 3.0],
            Rc::clone(&gpu_state),
        );

        let result_pred = vec![5.0, 3.0, 9.0, 10.0, 2.0, 7.0, 11.0, 10.0, 3.0];

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_inplace_add"),
                });
        mat_a.add_assign(&mat_b, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = mat_a.to_cpu().await.unwrap();
        assert_eq!(result_actual, result_pred);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_inplace_add_broadcast() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let mut mat_a = GpuMatrix::new(
            3,
            3,
            &vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 9.0, 1.0, 0.0],
            Rc::clone(&gpu_state),
        );
        let mat_b = GpuMatrix::new(3, 1, &vec![4.0, 9.0, 8.0], Rc::clone(&gpu_state));

        let result_pred = vec![5.0, 6.0, 5.0, 12.0, 10.0, 13.0, 17.0, 9.0, 8.0];

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_inplace_add_broadcast"),
                });
        mat_a.add_assign(&mat_b, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = mat_a.to_cpu().await.unwrap();
        assert_eq!(result_actual, result_pred);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_inplace_add_broadcast_non_square() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 4x2 matrix, broadcast a 4x1 column vector across both columns.
        let mut mat_a = GpuMatrix::new(
            4,
            2,
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Rc::clone(&gpu_state),
        );
        let mat_b = GpuMatrix::new(4, 1, &vec![10.0, 20.0, 30.0, 40.0], Rc::clone(&gpu_state));

        let result_pred = vec![11.0, 12.0, 23.0, 24.0, 35.0, 36.0, 47.0, 48.0];

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_inplace_add_broadcast_non_square"),
                });
        mat_a.add_assign(&mat_b, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = mat_a.to_cpu().await.unwrap();
        assert_eq!(result_actual, result_pred);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "Either shapes must match")]
    async fn matrix_inplace_add_incompatible_shapes_panics() {
        let gpu_state = Rc::new(GpuState::default().await);

        let mut mat_a = GpuMatrix::new(3, 3, &vec![0.0; 9], Rc::clone(&gpu_state));
        let mat_b = GpuMatrix::new(2, 2, &vec![0.0; 4], Rc::clone(&gpu_state));

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_incompatible_shapes"),
                });
        mat_a.add_assign(&mat_b, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_inplace_add_incompatible_shapes_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let mut mat_a = GpuMatrix::new(3, 3, &vec![0.0; 9], Rc::clone(&gpu_state));
        let mat_b = GpuMatrix::new(2, 2, &vec![0.0; 4], Rc::clone(&gpu_state));

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_incompatible_shapes"),
                });
        mat_a.add_assign(&mat_b, &mut encoder);
    }
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn sigmoid() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![0.0, 1.0, -1.0, 2.0, -2.0, 10.0];
        let input = GpuMatrix::new(2, 3, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_sigmoid"),
                });
        input.sigmoid(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();

        let expected: Vec<f32> = input_data
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        for (actual, expected) in result_actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    // Sigmoid should map any input into the open interval (0, 1).
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn sigmoid_output_range() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![-100.0, -10.0, 0.0, 10.0, 100.0];
        let input = GpuMatrix::new(1, 5, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(1, 5, &vec![0.0; 5], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_sigmoid_output_range"),
                });
        input.sigmoid(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();

        for value in result_actual {
            assert!(
                value >= 0.0 && value <= 1.0,
                "sigmoid output {value} out of range"
            );
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn sigmoid_prime() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // sigmoid_prime takes a sigmoid OUTPUT as input, not a raw pre-activation.
        let sigmoid_output_data = vec![0.5, 0.7310586, 0.26894143, 0.8807971, 0.11920292];
        let sigmoid_output = GpuMatrix::new(1, 5, &sigmoid_output_data, gpu_state.clone());
        let result = GpuMatrix::new(1, 5, &vec![0.0; 5], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_sigmoid_prime"),
                });
        sigmoid_output.sigmoid_prime(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();

        let expected: Vec<f32> = sigmoid_output_data.iter().map(|s| s * (1.0 - s)).collect();

        for (actual, expected) in result_actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    // sigmoid_prime is maximized at s = 0.5, where the value is 0.25.
    // This checks a known analytical property, not just an arbitrary computed value.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn sigmoid_prime_max_at_half() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let sigmoid_output = GpuMatrix::new(1, 1, &vec![0.5], gpu_state.clone());
        let result = GpuMatrix::new(1, 1, &vec![0.0], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_sigmoid_prime_max"),
                });
        sigmoid_output.sigmoid_prime(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();
        assert!((result_actual[0] - 0.25).abs() < 1e-5);
    }
}
