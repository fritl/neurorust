use crate::gpu::{
    kernels::GpuKernels,
    utils::{GpuContext, init_gpu},
};

pub struct GpuState {
    pub gpu_context: GpuContext,
    pub kernels: GpuKernels,
}

impl GpuState {
    pub async fn default() -> GpuState {
        let gpu_context = init_gpu().await;
        let kernels = GpuKernels::new(&gpu_context);
        GpuState {
            gpu_context,
            kernels,
        }
    }
}
