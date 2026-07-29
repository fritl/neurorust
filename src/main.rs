use std::{collections::HashMap, hint::black_box, num::NonZeroU64, time::Instant};

use wgpu::{
    BufferBindingType, BufferSlice, BufferUsages, DownlevelFlags, ExperimentalFeatures, Features,
    Limits, MapMode, MemoryHints, PipelineCompilationOptions, ShaderStages, Trace, util::DeviceExt,
    wgt::PollType,
};

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct DoubleGpu<'a> {
    gpu_context: &'a GpuContext,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> DoubleGpu<'a> {
    pub fn new(gpu_context: &'a GpuContext) -> DoubleGpu<'a> {
        let module = gpu_context
            .device
            .create_shader_module(wgpu::include_wgsl!("./shader.wgsl"));
        let bind_group_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: None,
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });
        DoubleGpu {
            gpu_context,
            bind_group_layout,
            pipeline,
        }
    }
    pub fn double<'b>(
        &self,
        input_data_buffer: &wgpu::Buffer,
        output_data_buffer: &wgpu::Buffer,
        download_buffer: &'b wgpu::Buffer,
        workgroup_count: usize,
    ) -> BufferSlice<'b> {
        // let data_bytes: &[u8] = bytemuck::cast_slice(data);
        // let input_data_buffer =
        //     self.gpu_context
        //         .device
        //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: None,
        //             contents: data_bytes,
        //             usage: BufferUsages::STORAGE,
        //         });
        //
        // let output_data_buffer = self
        //     .gpu_context
        //     .device
        //     .create_buffer(&wgpu::BufferDescriptor {
        //         label: None,
        //         size: input_data_buffer.size(),
        //         usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        //         mapped_at_creation: false,
        //     });
        //
        // let download_buffer = self
        //     .gpu_context
        //     .device
        //     .create_buffer(&wgpu::BufferDescriptor {
        //         label: None,
        //         size: input_data_buffer.size(),
        //         usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        //         mapped_at_creation: false,
        //     });
        let bind_group = self
            .gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_data_buffer.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = self
            .gpu_context
            .device
            .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // let workgroup_count = data.len().div_ceil(256);
        let max_x = 65535;
        let x = workgroup_count.min(max_x);
        let y = workgroup_count.div_ceil(max_x);
        compute_pass.dispatch_workgroups(x as u32, y as u32, 1);

        drop(compute_pass);

        encoder.copy_buffer_to_buffer(
            output_data_buffer,
            0,
            download_buffer,
            0,
            output_data_buffer.size(),
        );

        let command_buffer = encoder.finish();
        self.gpu_context.queue.submit([command_buffer]);

        let buffer_slice = download_buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |_| {});

        self.gpu_context
            .device
            .poll(PollType::wait_indefinitely())
            .unwrap();

        buffer_slice

        // let data = buffer_slice.get_mapped_range().unwrap();
        // let result: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&data);
        // result
    }
}

fn double_cpu(data: &Vec<f32>) -> Vec<f32> {
    data.iter().map(|x| x * 2.0).collect()
}

fn main() {
    env_logger::init();
    let gpu_context = pollster::block_on(init_gpu());
    let gpu_doubler = DoubleGpu::new(&gpu_context);

    let test_sizes = vec![
        1_000, 10_000, 100_000, 500_000, 700_000, 1_000_000, 10_000_000, 33_554_432,
    ];
    let mut cpu_times = HashMap::new();
    let mut gpu_times = HashMap::new();
    for i in test_sizes {
        let data: Vec<f32> = (0..i).map(|_| rand::random::<f32>()).collect();
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);
        let input_data_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data_bytes,
                    usage: BufferUsages::STORAGE,
                });

        let output_data_buffer = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_data_buffer.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download_buffer = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: input_data_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let workgroup_count = data.len().div_ceil(256);
        let gpu_start = Instant::now();
        let buffer_slice = gpu_doubler.double(
            &input_data_buffer,
            &output_data_buffer,
            &download_buffer,
            workgroup_count,
        );
        let gpu_time = gpu_start.elapsed();
        gpu_times.insert(i, gpu_time);
        let gpu_data = buffer_slice.get_mapped_range().unwrap();
        let result: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&gpu_data);
        black_box(result);

        let cpu_start = Instant::now();
        let result = double_cpu(&data);
        black_box(result);
        let cpu_time = cpu_start.elapsed();
        cpu_times.insert(i, cpu_time);
    }
    let mut sizes: Vec<_> = gpu_times.keys().collect();
    sizes.sort();
    for size in sizes {
        println!(
            "{size}: GPU {:?}, CPU {:?}",
            gpu_times[size], cpu_times[size]
        );
    }
    // println!("GPU Times: {gpu_times:#?}");
    // println!("CPU Times: {cpu_times:#?}");
}

async fn init_gpu() -> GpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: true,
        })
        .await
        .expect("Failed to create adapter");

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders")
    };

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            required_limits: Limits::downlevel_defaults(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::MemoryUsage,
            trace: Trace::Off,
        })
        .await
        .expect("Failed to create device");

    GpuContext { device, queue }
}
