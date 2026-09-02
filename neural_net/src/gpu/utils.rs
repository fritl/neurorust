use wgpu::{
    DownlevelFlags, ExperimentalFeatures, Features, Limits, MemoryHints, Trace,
    util::new_instance_with_webgpu_detection,
};

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub async fn init_gpu() -> GpuContext {
    let instance =
        new_instance_with_webgpu_detection(wgpu::InstanceDescriptor::new_without_display_handle())
            .await;
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
            required_limits: Limits {
                max_storage_buffer_binding_size: 188160000,
                ..Limits::downlevel_defaults()
            },
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::MemoryUsage,
            trace: Trace::Off,
        })
        .await
        .expect("Failed to create device");

    GpuContext { device, queue }
}
