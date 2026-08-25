use std::{ops::Div, panic::resume_unwind, rc::Rc};

use tokio::sync::oneshot;
use wgpu::{BindGroupEntry, BufferUsages, util::DeviceExt};

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SubtractAssignScaledUniform {
    scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DimensionsUniform {
    rows: u32,
    columns: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColumnSliceUniform {
    rows: u32,
    columns: u32,
    start_col: u32,
    batch_width: u32,
}

#[derive(Clone)]
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
        (self.rows, self.columns)
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

    fn create_dimension_buffer(&self) -> wgpu::Buffer {
        let device = &self.gpu_state.gpu_context.device;
        let dimension = DimensionsUniform {
            rows: self.rows as u32,
            columns: self.columns as u32,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!(
                "buffer_uiniform_dimension_matrix_{}x{}",
                self.rows, self.columns
            )),
            contents: bytemuck::cast_slice(&[dimension]),
            usage: BufferUsages::UNIFORM,
        })
    }

    pub fn column_max(&self, result: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.columns, result.columns);
        assert_eq!(result.rows, 1);

        let kernel = &self.gpu_state.kernels.column_max;
        let device = &self.gpu_state.gpu_context.device;
        let layout = kernel.get_bind_group_layout(0);
        let dimension_buffer = self.create_dimension_buffer();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_column_max"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: dimension_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = self.columns.div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn exp_shifted(&self, column_max: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.columns, column_max.columns);
        assert_eq!(column_max.rows, 1);

        let kernel = &self.gpu_state.kernels.exp_shifted;
        let device = &self.gpu_state.gpu_context.device;
        let layout = kernel.get_bind_group_layout(0);
        let dimension_buffer = self.create_dimension_buffer();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_exp_shifted"),
            layout: &layout,
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
                    resource: column_max.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = self.columns.div_ceil(16) as u32;
        let y = self.rows.div_ceil(16) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, y, 1));
    }

    pub fn column_sum(&self, result: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.columns, result.columns);
        assert_eq!(result.rows, 1);

        let kernel = &self.gpu_state.kernels.column_sum;
        let device = &self.gpu_state.gpu_context.device;
        let layout = kernel.get_bind_group_layout(0);
        let dimension_buffer = self.create_dimension_buffer();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_column_sum"),
            layout: &layout,
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
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = self.columns.div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn column_slice(
        &self,
        result: &GpuMatrix,
        start_col: u32,
        size: u32,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(self.rows, result.rows);
        assert_eq!(size, result.columns as u32);
        assert!(start_col + size <= self.columns as u32);

        let device = &self.gpu_state.gpu_context.device;
        let kernel = &self.gpu_state.kernels.column_slice;
        let layout = kernel.get_bind_group_layout(0);

        let dims = ColumnSliceUniform {
            rows: self.rows as u32,
            columns: self.columns as u32,
            start_col: start_col,
            batch_width: size,
        };

        let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer_uniform_column_slice"),
            contents: bytemuck::cast_slice(&[dims]),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_column_slice"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = size.div_ceil(16) as u32;
        let y = self.rows.div_ceil(16) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, y, 1));
    }

    pub fn normalize(&self, column_sum: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.columns, column_sum.columns);
        assert_eq!(column_sum.rows, 1);

        let kernel = &self.gpu_state.kernels.normalize;
        let device = &self.gpu_state.gpu_context.device;
        let layout = kernel.get_bind_group_layout(0);
        let dimension_buffer = self.create_dimension_buffer();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_normalize"),
            layout: &layout,
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
                    resource: column_sum.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = self.columns.div_ceil(16) as u32;
        let y = self.rows.div_ceil(16) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, y, 1));
    }

    pub fn cross_entropy_gradient(
        &self,
        result: &GpuMatrix,
        target_one_hot: &GpuMatrix,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(self.shape(), result.shape());
        assert_eq!(self.shape(), target_one_hot.shape());

        let kernel = &self.gpu_state.kernels.cross_entropy_gradient;
        let device = &self.gpu_state.gpu_context.device;
        let layout = kernel.get_bind_group_layout(0);
        let dimension_buffer = self.create_dimension_buffer();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_cross_entropy_gradient"),
            layout: &layout,
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
                    resource: result.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: target_one_hot.gpu_buffer.as_entire_binding(),
                },
            ],
        });

        let x = self.columns.div_ceil(16) as u32;
        let y = self.rows.div_ceil(16) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, y, 1));
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

    pub fn subtract_assign_scaled(
        &self,
        rhs: &GpuMatrix,
        scale: f32,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(self.shape(), rhs.shape());
        let device = &self.gpu_state.gpu_context.device;
        let uniforms = SubtractAssignScaledUniform { scale };
        let scale_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer_uniform_subtract_assign_scaled_uniform"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: BufferUsages::UNIFORM,
        });
        let kernel = &self.gpu_state.kernels.subtract;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_subtract_assign"),
            layout: &kernel.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scale_buffer.as_entire_binding(),
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
        let x = (self.rows() * self.columns()).div_ceil(256) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn row_sum(input: &GpuMatrix, output: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(input.rows(), output.rows());
        assert_eq!(output.columns(), 1);

        let device = &input.gpu_state.gpu_context.device;
        let kernel = &input.gpu_state.kernels.row_sum;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_row_sum"),
            layout: &kernel.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.gpu_buffer.as_entire_binding(),
                },
            ],
        });
        let x = input.rows() as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, 1, 1));
    }

    pub fn sigmoid(&self, encoder: &mut wgpu::CommandEncoder) {
        let device = &self.gpu_state.gpu_context.device;
        let kernel = &self.gpu_state.kernels.sigmoid;

        let bind_group_layout = kernel.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_sigmoid"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.gpu_buffer.as_entire_binding(),
            }],
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

    pub fn transpose(&self, result: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(self.rows, result.columns);
        assert_eq!(self.columns, result.rows);

        let device = &self.gpu_state.gpu_context.device;
        let kernel = &self.gpu_state.kernels.transpose;

        let uniforms = DimensionsUniform {
            rows: self.rows as u32,
            columns: self.columns as u32,
        };

        let columns_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer_uniform_transpose"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group_layout = kernel.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_transpose"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: columns_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: result.gpu_buffer.as_entire_binding(),
                },
            ],
        });
        let x = self.columns.div_ceil(16) as u32;
        let y = self.rows.div_ceil(16) as u32;
        Self::record_compute_pass(encoder, kernel, &bind_group, (x, y, 1));
    }
}

#[cfg(test)]
mod test {
    use super::*;

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

        let mat_a = GpuMatrix::new(
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

        let mat_a = GpuMatrix::new(
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
        let mat_a = GpuMatrix::new(
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

        let mat_a = GpuMatrix::new(3, 3, &vec![0.0; 9], Rc::clone(&gpu_state));
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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_sigmoid"),
                });
        input.sigmoid(&mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = input.to_cpu().await.unwrap();

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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_sigmoid_output_range"),
                });
        input.sigmoid(&mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = input.to_cpu().await.unwrap();

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
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_inplace_subtract_scaled() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let mat_a = GpuMatrix::new(
            3,
            3,
            &vec![5.0, 3.0, 9.0, 10.0, 2.0, 7.0, 11.0, 10.0, 3.0],
            Rc::clone(&gpu_state),
        );
        let mat_b = GpuMatrix::new(
            3,
            3,
            &vec![4.0, 1.0, 8.0, 7.0, 1.0, 3.0, 2.0, 9.0, 3.0],
            Rc::clone(&gpu_state),
        );
        let scale = 0.5;

        // a - b * scale
        let result_pred = vec![3.0, 2.5, 5.0, 6.5, 1.5, 5.5, 10.0, 5.5, 1.5];

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_inplace_subtract_scaled"),
                });
        mat_a.subtract_assign_scaled(&mat_b, scale, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = mat_a.to_cpu().await.unwrap();
        for (actual, expected) in result_actual.iter().zip(result_pred.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_inplace_subtract_scaled_zero_is_noop() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let a_data = vec![5.0, 3.0, 9.0, 10.0, 2.0, 7.0, 11.0, 10.0, 3.0];
        let mat_a = GpuMatrix::new(3, 3, &a_data, Rc::clone(&gpu_state));
        let mat_b = GpuMatrix::new(
            3,
            3,
            &vec![4.0, 1.0, 8.0, 7.0, 1.0, 3.0, 2.0, 9.0, 3.0],
            Rc::clone(&gpu_state),
        );

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_inplace_subtract_scaled_zero"),
                });
        mat_a.subtract_assign_scaled(&mat_b, 0.0, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = mat_a.to_cpu().await.unwrap();
        assert_eq!(
            result_actual, a_data,
            "scale of 0.0 should leave mat_a unchanged"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_inplace_subtract_scaled_one_matches_plain_subtract() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let mat_a = GpuMatrix::new(
            3,
            3,
            &vec![5.0, 3.0, 9.0, 10.0, 2.0, 7.0, 11.0, 10.0, 3.0],
            Rc::clone(&gpu_state),
        );
        let mat_b = GpuMatrix::new(
            3,
            3,
            &vec![4.0, 1.0, 8.0, 7.0, 1.0, 3.0, 2.0, 9.0, 3.0],
            Rc::clone(&gpu_state),
        );

        let result_pred = vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 9.0, 1.0, 0.0];

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_inplace_subtract_scaled_one"),
                });
        mat_a.subtract_assign_scaled(&mat_b, 1.0, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = mat_a.to_cpu().await.unwrap();
        assert_eq!(result_actual, result_pred);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic]
    async fn matrix_inplace_subtract_scaled_incompatible_shapes_panics() {
        let gpu_state = Rc::new(GpuState::default().await);

        let mat_a = GpuMatrix::new(3, 3, &vec![0.0; 9], Rc::clone(&gpu_state));
        let mat_b = GpuMatrix::new(2, 2, &vec![0.0; 4], Rc::clone(&gpu_state));

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_subtract_scaled_incompatible_shapes"),
                });
        mat_a.subtract_assign_scaled(&mat_b, 0.5, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_inplace_subtract_scaled_incompatible_shapes_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let mat_a = GpuMatrix::new(3, 3, &vec![0.0; 9], Rc::clone(&gpu_state));
        let mat_b = GpuMatrix::new(2, 2, &vec![0.0; 4], Rc::clone(&gpu_state));

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_subtract_scaled_incompatible_shapes"),
                });
        mat_a.subtract_assign_scaled(&mat_b, 0.5, &mut encoder);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_row_sum() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 3x4 matrix
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, // row 0 sum = 10
            5.0, 5.0, 5.0, 5.0, // row 1 sum = 20
            0.0, -1.0, 2.0, -2.0, // row 2 sum = -1
        ];
        let input = GpuMatrix::new(3, 4, &input_data, gpu_state.clone());
        let output = GpuMatrix::new(3, 1, &vec![0.0; 3], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_row_sum"),
                });
        GpuMatrix::row_sum(&input, &output, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = output.to_cpu().await.unwrap();
        assert_eq!(result_actual, vec![10.0, 20.0, -1.0]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_row_sum_single_column() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // A single-column matrix should just pass its values through unchanged.
        let input_data = vec![3.0, -2.0, 7.0];
        let input = GpuMatrix::new(3, 1, &input_data, gpu_state.clone());
        let output = GpuMatrix::new(3, 1, &vec![0.0; 3], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_row_sum_single_column"),
                });
        GpuMatrix::row_sum(&input, &output, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = output.to_cpu().await.unwrap();
        assert_eq!(result_actual, input_data);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_row_sum_larger_batch() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 20 rows x 30 columns, larger than one workgroup (256 threads covers
        // this in one dispatch, but this still exercises the per-row loop
        // with a non-trivial column count).
        let rows = 20;
        let cols = 30;
        let input_data: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32).collect();
        let input = GpuMatrix::new(rows, cols, &input_data, gpu_state.clone());
        let output = GpuMatrix::new(rows, 1, &vec![0.0; rows], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_row_sum_larger_batch"),
                });
        GpuMatrix::row_sum(&input, &output, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = output.to_cpu().await.unwrap();

        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            for c in 0..cols {
                expected[r] += input_data[r * cols + c];
            }
        }

        assert_eq!(result_actual, expected);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn matrix_row_sum_wrong_output_shape_panics() {
        let gpu_state = Rc::new(GpuState::default().await);

        let input = GpuMatrix::new(3, 4, &vec![0.0; 12], gpu_state.clone());
        // wrong: 2 rows instead of 3, and 2 columns instead of 1
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_row_sum_wrong_shape"),
                });
        GpuMatrix::row_sum(&input, &output, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_row_sum_wrong_output_shape_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input = GpuMatrix::new(3, 4, &vec![0.0; 12], gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_row_sum_wrong_shape"),
                });
        GpuMatrix::row_sum(&input, &output, &mut encoder);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_transpose() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 2x3 matrix
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuMatrix::new(2, 3, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose"),
                });
        input.transpose(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();

        // input (2x3): [[1,2,3],[4,5,6]]
        // transposed (3x2): [[1,4],[2,5],[3,6]]
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(result_actual, expected);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_transpose_square() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = GpuMatrix::new(3, 3, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(3, 3, &vec![0.0; 9], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose_square"),
                });
        input.transpose(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();

        // input (3x3): [[1,2,3],[4,5,6],[7,8,9]]
        // transposed:  [[1,4,7],[2,5,8],[3,6,9]]
        let expected = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        assert_eq!(result_actual, expected);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_transpose_column_vector() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 4x1 column vector -> 1x4 row vector
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = GpuMatrix::new(4, 1, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(1, 4, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose_column_vector"),
                });
        input.transpose(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();
        assert_eq!(result_actual, input_data);
    }

    // Transposing twice should return a matrix equal to the original.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_transpose_twice_is_identity() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuMatrix::new(2, 3, &input_data, gpu_state.clone());
        let once = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone());
        let twice = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose_twice"),
                });
        input.transpose(&once, &mut encoder);
        once.transpose(&twice, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = twice.to_cpu().await.unwrap();
        assert_eq!(result_actual, input_data);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_transpose_large_non_square() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 20x30, larger than one workgroup (16x16) in both directions —
        // catches a swapped-dispatch-axis bug the same way matmul's
        // large_non_square test does.
        let rows = 20;
        let cols = 30;
        let input_data: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32).collect();
        let input = GpuMatrix::new(rows, cols, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(cols, rows, &vec![0.0; rows * cols], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose_large_non_square"),
                });
        input.transpose(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = result.to_cpu().await.unwrap();

        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                expected[c * rows + r] = input_data[r * cols + c];
            }
        }

        assert_eq!(result_actual, expected);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn matrix_transpose_wrong_result_shape_panics() {
        let gpu_state = Rc::new(GpuState::default().await);

        let input = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());
        // wrong: should be 3x2, not 2x3
        let result = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose_wrong_shape"),
                });
        input.transpose(&result, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_transpose_wrong_result_shape_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());
        let result = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_transpose_wrong_shape"),
                });
        input.transpose(&result, &mut encoder);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_max() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 3 rows (classes), 4 columns (batch)
        let input_data = vec![1.0, 5.0, 2.0, 0.0, 3.0, 2.0, 9.0, 1.0, 2.0, 8.0, 1.0, 4.0];
        let input = GpuMatrix::new(3, 4, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(1, 4, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_max"),
                });
        input.column_max(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        // col0: max(1,3,2)=3, col1: max(5,2,8)=8, col2: max(2,9,1)=9, col3: max(0,1,4)=4
        assert_eq!(actual, vec![3.0, 8.0, 9.0, 4.0]);
    }

    // Non-square: catches the rows-vs-columns dispatch bug directly.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_max_non_square_more_columns_than_rows() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 2 rows, 10 columns — with the rows-based dispatch bug, only the
        // first 2 columns would ever get computed; the rest stay at 0.
        let rows = 2;
        let columns = 10;
        let input_data: Vec<f32> = (0..rows * columns).map(|i| (i % 13) as f32).collect();
        let input = GpuMatrix::new(rows, columns, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(1, columns, &vec![0.0; columns], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_max_non_square"),
                });
        input.column_max(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();

        let mut expected = vec![0.0f32; columns];
        for c in 0..columns {
            let mut m = input_data[c];
            for r in 1..rows {
                let v = input_data[r * columns + c];
                if v > m {
                    m = v;
                }
            }
            expected[c] = m;
        }

        assert_eq!(
            actual, expected,
            "column_max likely dispatched with wrong dimension (rows vs columns)"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_exp_shifted() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let mut input = GpuMatrix::new(2, 2, &input_data, gpu_state.clone());
        let col_max = GpuMatrix::new(1, 2, &vec![3.0, 4.0], gpu_state.clone()); // per-column max, precomputed

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_exp_shifted"),
                });
        input.exp_shifted(&col_max, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = input.to_cpu().await.unwrap();

        // col 0: [1,3], max=3 -> exp(1-3)=exp(-2), exp(3-3)=exp(0)=1
        // col 1: [2,4], max=4 -> exp(2-4)=exp(-2), exp(4-4)=exp(0)=1
        let expected = vec![
            (1.0f32 - 3.0).exp(),
            (2.0f32 - 4.0).exp(),
            (3.0f32 - 3.0).exp(),
            (4.0f32 - 4.0).exp(),
        ];

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_sum() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let input = GpuMatrix::new(3, 2, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(1, 2, &vec![0.0; 2], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_sum"),
                });
        input.column_sum(&result, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        // col0: 1+3+5=9, col1: 2+4+6=12
        assert_eq!(actual, vec![9.0, 12.0]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_normalize() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let mut input = GpuMatrix::new(2, 2, &input_data, gpu_state.clone());
        let col_sum = GpuMatrix::new(1, 2, &vec![4.0, 6.0], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_normalize"),
                });
        input.normalize(&col_sum, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = input.to_cpu().await.unwrap();
        // col0: [1,3]/4 -> [0.25, 0.75]; col1: [2,4]/6 -> [0.333.., 0.666..]
        let expected = vec![1.0 / 4.0, 2.0 / 6.0, 3.0 / 4.0, 4.0 / 6.0];

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    // Full pipeline: column_max -> exp_shifted -> column_sum -> normalize
    // should produce a valid softmax distribution (each column sums to 1).
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_softmax_pipeline_columns_sum_to_one() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![2.0, -1.0, 0.5, 1.0, 3.0, 0.5, 0.1, 0.2, 0.5]; // 3x3, 3 classes, batch 3
        let mut logits = GpuMatrix::new(3, 3, &input_data, gpu_state.clone());
        let col_max = GpuMatrix::new(1, 3, &vec![0.0; 3], gpu_state.clone());
        let col_sum = GpuMatrix::new(1, 3, &vec![0.0; 3], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_pipeline"),
                });
        logits.column_max(&col_max, &mut encoder);
        logits.exp_shifted(&col_max, &mut encoder);
        logits.column_sum(&col_sum, &mut encoder);
        logits.normalize(&col_sum, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let softmax_output = logits.to_cpu().await.unwrap();

        for col in 0..3 {
            let sum: f32 = (0..3).map(|row| softmax_output[row * 3 + col]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "column {col} softmax values sum to {sum}, expected 1.0"
            );
        }
        for value in &softmax_output {
            assert!(
                *value >= 0.0 && *value <= 1.0,
                "softmax value {value} out of [0,1]"
            );
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_cross_entropy_gradient() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // pretend these are already-computed softmax outputs, 2 classes, batch 2
        let softmax_data = vec![0.3, 0.6, 0.7, 0.4];
        let softmax_output = GpuMatrix::new(2, 2, &softmax_data, gpu_state.clone());
        let one_hot = GpuMatrix::new(2, 2, &vec![1.0, 0.0, 0.0, 1.0], gpu_state.clone());
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_ce_gradient"),
                });
        softmax_output.cross_entropy_gradient(&result, &one_hot, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();

        // (softmax - one_hot) / columns(=2)
        let expected: Vec<f32> = softmax_data
            .iter()
            .zip([1.0, 0.0, 0.0, 1.0].iter())
            .map(|(s, t)| (s - t) / 2.0)
            .collect();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn matrix_cross_entropy_gradient_shape_mismatch_panics() {
        let gpu_state = Rc::new(GpuState::default().await);

        let softmax_output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let one_hot = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone()); // wrong rows
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_ce_gradient_mismatch"),
                });
        softmax_output.cross_entropy_gradient(&result, &one_hot, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_cross_entropy_gradient_shape_mismatch_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let softmax_output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let one_hot = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone());
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_ce_gradient_mismatch"),
                });
        softmax_output.cross_entropy_gradient(&result, &one_hot, &mut encoder);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_slice_middle() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 3 rows, 6 columns
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ];
        let input = GpuMatrix::new(3, 6, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_middle"),
                });
        input.column_slice(&result, 2, 2, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        // columns 2..4 of each row
        let expected = vec![3.0, 4.0, 9.0, 10.0, 15.0, 16.0];
        assert_eq!(actual, expected);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_slice_start() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let input = GpuMatrix::new(2, 2, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(2, 1, &vec![0.0; 2], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_start"),
                });
        input.column_slice(&result, 0, 1, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        assert_eq!(actual, vec![1.0, 3.0]); // first column of each row
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_slice_end() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let input = GpuMatrix::new(2, 2, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(2, 1, &vec![0.0; 2], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_end"),
                });
        input.column_slice(&result, 1, 1, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        assert_eq!(actual, vec![2.0, 4.0]); // last column of each row
    }

    // Full-width slice should reproduce the original matrix exactly.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_slice_full_width_is_identity() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let input = GpuMatrix::new(2, 3, &input_data, gpu_state.clone());
        let result = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_full_width"),
                });
        input.column_slice(&result, 0, 3, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        assert_eq!(actual, input_data);
    }

    // Non-square, larger than one workgroup, to catch a swapped-dispatch-axis
    // bug the way matmul's/transpose's equivalent tests do.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_slice_large_non_square() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let rows = 20;
        let columns = 30;
        let input_data: Vec<f32> = (0..rows * columns).map(|i| (i % 17) as f32).collect();
        let input = GpuMatrix::new(rows, columns, &input_data, gpu_state.clone());

        let start_col = 10;
        let size = 15;
        let result = GpuMatrix::new(rows, size, &vec![0.0; rows * size], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_large_non_square"),
                });
        input.column_slice(&result, start_col as u32, size as u32, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();

        let mut expected = vec![0.0f32; rows * size];
        for r in 0..rows {
            for c in 0..size {
                expected[r * size + c] = input_data[r * columns + start_col + c];
            }
        }

        assert_eq!(actual, expected);
    }

    // Mimics the CPU train() loop's last-batch case: batch_size doesn't evenly
    // divide total columns, so the final slice is smaller than the others.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn matrix_column_slice_partial_last_batch() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 7 columns total, batch_size 3 -> batches of 3, 3, 1
        let input_data: Vec<f32> = (0..2 * 7).map(|i| i as f32).collect();
        let input = GpuMatrix::new(2, 7, &input_data, gpu_state.clone());

        let result = GpuMatrix::new(2, 1, &vec![0.0; 2], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_partial_last_batch"),
                });
        input.column_slice(&result, 6, 1, &mut encoder); // last remaining column
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = result.to_cpu().await.unwrap();
        // column 6 of each row: row0 -> index 6, row1 -> index 13
        assert_eq!(actual, vec![input_data[6], input_data[13]]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn matrix_column_slice_out_of_bounds_panics() {
        let gpu_state = Rc::new(GpuState::default().await);
        let input = GpuMatrix::new(2, 4, &vec![0.0; 8], gpu_state.clone());
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_oob"),
                });
        // start_col=3, size=2 -> would read columns 3 and 4, but only 4 columns exist (0..3)
        input.column_slice(&result, 3, 2, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_column_slice_out_of_bounds_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let input = GpuMatrix::new(2, 4, &vec![0.0; 8], gpu_state.clone());
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_oob"),
                });
        input.column_slice(&result, 3, 2, &mut encoder);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn matrix_column_slice_wrong_result_rows_panics() {
        let gpu_state = Rc::new(GpuState::default().await);
        let input = GpuMatrix::new(3, 4, &vec![0.0; 12], gpu_state.clone());
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone()); // wrong: 2 rows instead of 3

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_wrong_rows"),
                });
        input.column_slice(&result, 0, 2, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn matrix_column_slice_wrong_result_rows_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let input = GpuMatrix::new(3, 4, &vec![0.0; 12], gpu_state.clone());
        let result = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_column_slice_wrong_rows"),
                });
        input.column_slice(&result, 0, 2, &mut encoder);
    }
}
