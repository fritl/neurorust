use std::num::NonZeroU64;

use wgpu::{
    BufferBindingType, BufferUsages, DownlevelFlags, ExperimentalFeatures, Features, Limits,
    MapMode, MemoryHints, PipelineCompilationOptions, ShaderStages, Trace, util::DeviceExt,
    wgt::PollType,
};

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn main() {
    env_logger::init();
    let gpu_context = pollster::block_on(init_gpu());
    let module = gpu_context
        .device
        .create_shader_module(wgpu::include_wgsl!("./shader.wgsl"));

    let input_data: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let input_bytes = bytemuck::cast_slice(&input_data);
    let input_data_buffer =
        gpu_context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: input_bytes,
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

    let bind_group = gpu_context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
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

    let pipeline_layout =
        gpu_context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

    let pipeline = gpu_context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: None,
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut encoder = gpu_context
        .device
        .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor { label: None });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    let workgroup_count = input_data.len().div_ceil(64);
    compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

    drop(compute_pass);

    encoder.copy_buffer_to_buffer(
        &output_data_buffer,
        0,
        &download_buffer,
        0,
        output_data_buffer.size(),
    );

    let command_buffer = encoder.finish();
    gpu_context.queue.submit([command_buffer]);

    let buffer_slice = download_buffer.slice(..);
    buffer_slice.map_async(MapMode::Read, |_| {});

    gpu_context
        .device
        .poll(PollType::wait_indefinitely())
        .unwrap();

    let data = buffer_slice.get_mapped_range().unwrap();
    let result: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&data);
    println!("Result {result:?}");
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
