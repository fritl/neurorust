use crate::gpu::utils::GpuContext;

pub struct GpuKernels {
    pub matmul: wgpu::ComputePipeline,
    pub inplace_add: wgpu::ComputePipeline,
    pub hadamard: wgpu::ComputePipeline,
    pub sigmoid: wgpu::ComputePipeline,
    pub sigmoid_prime: wgpu::ComputePipeline,
}

impl GpuKernels {
    pub fn new(gpu_context: &GpuContext) -> GpuKernels {
        let matmul = Self::init_matmul_pipeline(gpu_context);
        let inplace_add = Self::init_add_pipeline(gpu_context);
        let hadamard = Self::init_hadamard_pipeline(gpu_context);
        let (sigmoid, sigmoid_prime) = Self::init_sigmoid_pipeline(gpu_context);
        GpuKernels {
            matmul,
            inplace_add,
            hadamard,
            sigmoid,
            sigmoid_prime,
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

    fn init_hadamard_pipeline(gpu_context: &GpuContext) -> wgpu::ComputePipeline {
        let module = gpu_context
            .device
            .create_shader_module(wgpu::include_wgsl!("./kernels/hadamard.wgsl"));

        gpu_context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_pipeline_hadamard"),
                layout: None,
                module: &module,
                entry_point: Some("hadamard"),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn init_sigmoid_pipeline(
        gpu_context: &GpuContext,
    ) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
        let module = gpu_context
            .device
            .create_shader_module(wgpu::include_wgsl!("./kernels/sigmoid.wgsl"));
        let sigmoid =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute_pipeline_sigmoid"),
                    layout: None,
                    module: &module,
                    entry_point: Some("sigmoid"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let sigmoid_prime =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute_pipeline_sigmoid"),
                    layout: None,
                    module: &module,
                    entry_point: Some("sigmoid_prime"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        (sigmoid, sigmoid_prime)
    }
}
