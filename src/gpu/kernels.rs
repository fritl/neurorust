use wgpu::ShaderModuleDescriptor;

use crate::gpu::utils::GpuContext;

pub struct GpuKernels {
    pub matmul: wgpu::ComputePipeline,
    pub inplace_add: wgpu::ComputePipeline,
}

impl GpuKernels {
    pub fn new(gpu_context: &GpuContext) -> GpuKernels {
        let matmul = Self::init_matmul_pipeline(gpu_context);
        let inplace_add = Self::init_add_pipeline(gpu_context);
        GpuKernels {
            matmul,
            inplace_add,
        }
    }
    fn init_matmul_pipeline(gpu_context: &GpuContext) -> wgpu::ComputePipeline {
        let module = gpu_context
            .device
            .create_shader_module(wgpu::include_wgsl!("./kernels/matmul.wgsl"));
        gpu_context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_pipeline_matmul"),
                layout: None,
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn init_add_pipeline(gpu_context: &GpuContext) -> wgpu::ComputePipeline {
        let module = gpu_context
            .device
            .create_shader_module(wgpu::include_wgsl!("./kernels/add.wgsl"));
        gpu_context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_pipeline_inplace_add"),
                layout: None,
                module: &module,
                entry_point: Some("inplace_add"),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}
