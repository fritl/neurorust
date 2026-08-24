use crate::gpu::{buffer_pool::BufferPool, matrix::GpuMatrix};

pub struct SoftmaxCrossEntropy;

impl SoftmaxCrossEntropy {
    /// Calculates the last activation after the softmax function NOT the cost
    pub fn forward(
        logits: &GpuMatrix,
        output: &GpuMatrix,
        buffer_pool: &mut BufferPool,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(logits.shape(), output.shape());
        let column_max = buffer_pool.get(1, logits.columns());
        logits.column_max(&column_max, encoder);
        encoder.copy_buffer_to_buffer(
            logits.gpu_buffer(),
            0,
            output.gpu_buffer(),
            0,
            logits.gpu_buffer().size(),
        );
        output.exp_shifted(&column_max, encoder);

        let column_sum = column_max;
        output.column_sum(&column_sum, encoder);
        output.normalize(&column_sum, encoder);
        buffer_pool.recycle(column_sum);
    }

    pub fn backward(
        logits: &GpuMatrix,
        target_one_hot: &GpuMatrix,
        output: &GpuMatrix,
        buffer_pool: &mut BufferPool,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert_eq!(logits.shape(), output.shape());
        assert_eq!(target_one_hot.shape(), logits.shape());

        let column_max = buffer_pool.get(1, logits.columns());
        let scratch_buffer = buffer_pool.get(logits.rows(), logits.columns());
        logits.column_max(&column_max, encoder);
        encoder.copy_buffer_to_buffer(
            logits.gpu_buffer(),
            0,
            scratch_buffer.gpu_buffer(),
            0,
            logits.gpu_buffer().size(),
        );
        scratch_buffer.exp_shifted(&column_max, encoder);

        let column_sum = column_max;
        scratch_buffer.column_sum(&column_sum, encoder);
        scratch_buffer.normalize(&column_sum, encoder);
        scratch_buffer.cross_entropy_gradient(&output, target_one_hot, encoder);
        buffer_pool.recycle(column_sum);
        buffer_pool.recycle(scratch_buffer);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::state::GpuState;
    use std::rc::Rc;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_produces_valid_softmax_distribution() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 3 classes, batch 3
        let logits_data = vec![2.0, -1.0, 0.5, 1.0, 3.0, 0.5, 0.1, 0.2, 0.5];
        let logits = GpuMatrix::new(3, 3, &logits_data, gpu_state.clone());
        let output = GpuMatrix::new(3, 3, &vec![0.0; 9], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_forward"),
                });
        SoftmaxCrossEntropy::forward(&logits, &output, &mut buffer_pool, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = output.to_cpu().await.unwrap();

        for col in 0..3 {
            let sum: f32 = (0..3).map(|row| actual[row * 3 + col]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "column {col} softmax values sum to {sum}, expected 1.0"
            );
        }
        for value in &actual {
            assert!(
                *value >= 0.0 && *value <= 1.0,
                "softmax value {value} out of [0,1]"
            );
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_does_not_mutate_logits() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let logits_data = vec![1.0, 2.0, 3.0, 4.0];
        let logits = GpuMatrix::new(2, 2, &logits_data, gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_forward_preserves_logits"),
                });
        SoftmaxCrossEntropy::forward(&logits, &output, &mut buffer_pool, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let logits_after = logits.to_cpu().await.unwrap();
        assert_eq!(
            logits_after, logits_data,
            "logits buffer should be untouched — forward() writes into output, not logits"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_matches_cpu_reference() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 2 classes, batch 2
        let logits_data = vec![1.0, 2.0, 3.0, 0.5];
        let logits = GpuMatrix::new(2, 2, &logits_data, gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_forward_cpu_ref"),
                });
        SoftmaxCrossEntropy::forward(&logits, &output, &mut buffer_pool, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = output.to_cpu().await.unwrap();

        // CPU reference softmax, per column
        fn cpu_softmax_column(col: &[f32]) -> Vec<f32> {
            let max = col.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = col.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|v| v / sum).collect()
        }

        // col0: [1,3], col1: [2, 0.5]
        let col0 = cpu_softmax_column(&[logits_data[0], logits_data[2]]);
        let col1 = cpu_softmax_column(&[logits_data[1], logits_data[3]]);
        let expected = vec![col0[0], col1[0], col0[1], col1[1]];

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn backward_matches_cpu_reference() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let logits_data = vec![1.0, 2.0, 3.0, 0.5];
        let logits = GpuMatrix::new(2, 2, &logits_data, gpu_state.clone());
        let one_hot_data = vec![1.0, 0.0, 0.0, 1.0];
        let target_one_hot = GpuMatrix::new(2, 2, &one_hot_data, gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_backward"),
                });
        SoftmaxCrossEntropy::backward(
            &logits,
            &target_one_hot,
            &output,
            &mut buffer_pool,
            &mut encoder,
        );
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = output.to_cpu().await.unwrap();

        fn cpu_softmax_column(col: &[f32]) -> Vec<f32> {
            let max = col.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = col.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|v| v / sum).collect()
        }

        let col0 = cpu_softmax_column(&[logits_data[0], logits_data[2]]);
        let col1 = cpu_softmax_column(&[logits_data[1], logits_data[3]]);
        let softmax = vec![col0[0], col1[0], col0[1], col1[1]];

        // (softmax - one_hot) / columns
        let columns = 2.0;
        let expected: Vec<f32> = softmax
            .iter()
            .zip(one_hot_data.iter())
            .map(|(s, t)| (s - t) / columns)
            .collect();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn backward_does_not_mutate_logits() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let logits_data = vec![1.0, 2.0, 3.0, 0.5];
        let logits = GpuMatrix::new(2, 2, &logits_data, gpu_state.clone());
        let target_one_hot = GpuMatrix::new(2, 2, &vec![1.0, 0.0, 0.0, 1.0], gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_backward_preserves_logits"),
                });
        SoftmaxCrossEntropy::backward(
            &logits,
            &target_one_hot,
            &output,
            &mut buffer_pool,
            &mut encoder,
        );
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let logits_after = logits.to_cpu().await.unwrap();
        assert_eq!(
            logits_after, logits_data,
            "logits buffer should be untouched — backward() writes into output and a pool scratch buffer, not logits"
        );
    }

    // Perfect prediction (softmax matches one_hot exactly) should produce a
    // near-zero gradient — sanity check on the gradient formula itself.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn backward_near_zero_gradient_for_confident_correct_prediction() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // Large logit gap -> softmax is extremely close to [1, 0] for this column.
        let logits_data = vec![100.0, -100.0];
        let logits = GpuMatrix::new(2, 1, &logits_data, gpu_state.clone());
        let target_one_hot = GpuMatrix::new(2, 1, &vec![1.0, 0.0], gpu_state.clone());
        let output = GpuMatrix::new(2, 1, &vec![0.0; 2], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_backward_confident"),
                });
        SoftmaxCrossEntropy::backward(
            &logits,
            &target_one_hot,
            &output,
            &mut buffer_pool,
            &mut encoder,
        );
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual = output.to_cpu().await.unwrap();
        for value in actual {
            assert!(
                value.abs() < 1e-3,
                "expected near-zero gradient for confident correct prediction, got {value}"
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn forward_shape_mismatch_panics() {
        let gpu_state = Rc::new(GpuState::default().await);
        let logits = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let output = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone()); // wrong rows
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_forward_mismatch"),
                });
        SoftmaxCrossEntropy::forward(&logits, &output, &mut buffer_pool, &mut encoder);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn forward_shape_mismatch_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let logits = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let output = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone());
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_softmax_forward_mismatch"),
                });
        SoftmaxCrossEntropy::forward(&logits, &output, &mut buffer_pool, &mut encoder);
    }
}
