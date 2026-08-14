use rand::RngExt;
use rand::distr;
use rand::distr::Distribution;

use std::rc::Rc;

use crate::gpu::{matrix::GpuMatrix, state::GpuState};

struct Layer {
    weights: GpuMatrix,
    bias: GpuMatrix,
    cached_z: Option<GpuMatrix>,
    cached_input: Option<GpuMatrix>,
    gpu_state: Rc<GpuState>,
}

impl Layer {
    pub fn from_manual_weights(
        weights: GpuMatrix,
        bias: GpuMatrix,
        gpu_state: Rc<GpuState>,
    ) -> Layer {
        assert_eq!(weights.rows(), bias.rows());
        Layer {
            weights,
            bias,
            cached_z: None,
            cached_input: None,
            gpu_state,
        }
    }

    pub fn with_random_weights(
        input_neurons: usize,
        output_nurons: usize,
        rng: &mut impl rand::Rng,
        gpu_state: Rc<GpuState>,
    ) -> Layer {
        let weight_data = (0..input_neurons * output_nurons)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect::<Vec<_>>();
        let weights = GpuMatrix::new(
            output_nurons,
            input_neurons,
            &weight_data,
            Rc::clone(&gpu_state),
        );
        let bias = GpuMatrix::new(
            output_nurons,
            1,
            &vec![0.0 as f32; output_nurons],
            Rc::clone(&gpu_state),
        );

        Layer {
            weights,
            bias,
            cached_input: None,
            cached_z: None,
            gpu_state,
        }
    }

    pub fn with_uniform_xavier_weights(
        input_neurons: usize,
        output_nurons: usize,
        rng: &mut impl rand::Rng,
        gpu_state: Rc<GpuState>,
    ) -> Layer {
        let x = (6.0 / (input_neurons + output_nurons) as f32).sqrt();
        let dist = distr::Uniform::new(-x, x).unwrap();
        let weight_data = (0..input_neurons * output_nurons)
            .map(|_| dist.sample(rng))
            .collect::<Vec<_>>();
        let weights = GpuMatrix::new(
            output_nurons,
            input_neurons,
            &weight_data,
            Rc::clone(&gpu_state),
        );
        let bias = GpuMatrix::new(
            output_nurons,
            1,
            &vec![0.0 as f32; output_nurons],
            Rc::clone(&gpu_state),
        );

        Layer {
            weights,
            bias,
            cached_input: None,
            cached_z: None,
            gpu_state,
        }
    }

    pub fn forward(
        &mut self,
        x: &GpuMatrix,
        output: &GpuMatrix,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        match &self.cached_input {
            Some(i) if x.rows() == i.rows() && x.columns() == i.columns() => {}
            _ => {
                self.cached_input = Some(GpuMatrix::empty(
                    x.rows(),
                    x.columns(),
                    Rc::clone(&self.gpu_state),
                ))
            }
        }
        encoder.copy_buffer_to_buffer(
            x.gpu_buffer(),
            0,
            self.cached_input.as_ref().unwrap().gpu_buffer(),
            0,
            x.gpu_buffer().size(),
        );

        match &self.cached_z {
            Some(z) if z.rows() == self.weights.rows() && z.columns() == x.columns() => {}
            _ => {
                self.cached_z = Some(GpuMatrix::empty(
                    self.weights.rows(),
                    x.columns(),
                    Rc::clone(&self.gpu_state),
                ))
            }
        };
        GpuMatrix::matmul(&self.weights, x, &output, encoder);
        encoder.copy_buffer_to_buffer(
            output.gpu_buffer(),
            0,
            self.cached_z.as_ref().unwrap().gpu_buffer(),
            0,
            output.gpu_buffer().size(),
        );

        output.add_assign(&self.bias, encoder);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn from_manual_weights_stores_correct_shapes() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 3 input neurons, 2 output neurons.
        let weight_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let weights = GpuMatrix::new(2, 3, &weight_data, gpu_state.clone());
        let bias = GpuMatrix::new(2, 1, &vec![0.0, 0.0], gpu_state.clone());

        let layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        assert_eq!(layer.weights.rows(), 2);
        assert_eq!(layer.weights.columns(), 3);
        assert_eq!(layer.bias.rows(), 2);
        assert_eq!(layer.bias.columns(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    #[should_panic(expected = "assertion")]
    async fn from_manual_weights_mismatched_rows_panics() {
        let gpu_state = Rc::new(GpuState::default().await);

        // weights has 2 output rows, bias has 3 — mismatch.
        let weights = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());
        let bias = GpuMatrix::new(3, 1, &vec![0.0; 3], gpu_state.clone());

        Layer::from_manual_weights(weights, bias, gpu_state.clone());
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    #[should_panic]
    async fn from_manual_weights_mismatched_rows_panics() {
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let weights = GpuMatrix::new(2, 3, &vec![0.0; 6], gpu_state.clone());
        let bias = GpuMatrix::new(3, 1, &vec![0.0; 3], gpu_state.clone());

        Layer::from_manual_weights(weights, bias, gpu_state.clone());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn with_random_weights_has_correct_shape() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let mut rng = StdRng::seed_from_u64(42);

        let layer = Layer::with_random_weights(4, 3, &mut rng, gpu_state.clone());

        assert_eq!(layer.weights.rows(), 3);
        assert_eq!(layer.weights.columns(), 4);
        assert_eq!(layer.bias.rows(), 3);
        assert_eq!(layer.bias.columns(), 1);
    }

    // Weights must fall in the documented range, and bias must start at zero.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn with_random_weights_values_in_expected_range() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let mut rng = StdRng::seed_from_u64(42);

        let layer = Layer::with_random_weights(4, 3, &mut rng, gpu_state.clone());

        let weight_values = layer.weights.to_cpu().await.unwrap();
        for value in weight_values {
            assert!(
                (-0.5..0.5).contains(&value),
                "weight {value} out of expected range"
            );
        }

        let bias_values = layer.bias.to_cpu().await.unwrap();
        assert_eq!(bias_values, vec![0.0; 3]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn with_uniform_xavier_weights_has_correct_shape() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let mut rng = StdRng::seed_from_u64(42);

        let layer = Layer::with_uniform_xavier_weights(4, 3, &mut rng, gpu_state.clone());

        assert_eq!(layer.weights.rows(), 3);
        assert_eq!(layer.weights.columns(), 4);
        assert_eq!(layer.bias.rows(), 3);
        assert_eq!(layer.bias.columns(), 1);
    }

    // Xavier init bounds weights to [-x, x] where x = sqrt(6 / (fan_in + fan_out)).
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn with_uniform_xavier_weights_values_in_expected_range() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);
        let mut rng = StdRng::seed_from_u64(42);

        let input_neurons = 4;
        let output_neurons = 3;
        let layer = Layer::with_uniform_xavier_weights(
            input_neurons,
            output_neurons,
            &mut rng,
            gpu_state.clone(),
        );

        let expected_bound = (6.0 / (input_neurons + output_neurons) as f32).sqrt();

        let weight_values = layer.weights.to_cpu().await.unwrap();
        for value in weight_values {
            assert!(
                (-expected_bound..expected_bound).contains(&value),
                "weight {value} out of expected Xavier range [-{expected_bound}, {expected_bound}]"
            );
        }

        let bias_values = layer.bias.to_cpu().await.unwrap();
        assert_eq!(bias_values, vec![0.0; 3]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_computes_matmul_plus_bias() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 2 input neurons, 3 output neurons, batch size 2.
        let weight_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let weights = GpuMatrix::new(3, 2, &weight_data, gpu_state.clone());
        let bias_data = vec![1.0, 2.0, 3.0];
        let bias = GpuMatrix::new(3, 1, &bias_data, gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        // input: 2x2 (2 features, batch size 2)
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let x = GpuMatrix::new(2, 2, &x_data, gpu_state.clone());
        let output = GpuMatrix::new(3, 2, &vec![0.0; 6], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward"),
                });
        layer.forward(&x, &output, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let result_actual = output.to_cpu().await.unwrap();

        // matmul(weights, x): weights (3x2): [[1,0],[0,1],[1,1]], x (2x2): [[1,2],[3,4]]
        // matmul result (3x2): [[1,2],[3,4],[4,6]]
        // + bias (broadcast per row): [[2,3],[5,6],[7,9]]
        let expected = vec![2.0, 3.0, 5.0, 6.0, 7.0, 9.0];
        assert_eq!(result_actual, expected);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_caches_input() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let weights = GpuMatrix::new(2, 2, &vec![1.0, 0.0, 0.0, 1.0], gpu_state.clone());
        let bias = GpuMatrix::new(2, 1, &vec![0.0, 0.0], gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        let x_data = vec![5.0, 6.0, 7.0, 8.0];
        let x = GpuMatrix::new(2, 2, &x_data, gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_caches_input"),
                });
        layer.forward(&x, &output, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let cached_input = layer
            .cached_input
            .as_ref()
            .expect("cached_input should be set after forward");
        assert_eq!(cached_input.rows(), 2);
        assert_eq!(cached_input.columns(), 2);

        let cached_data = cached_input.to_cpu().await.unwrap();
        assert_eq!(cached_data, x_data);
    }

    // cached_z holds the pre-bias matmul result, not the final (post-bias) output.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_caches_pre_bias_z() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let weights = GpuMatrix::new(2, 2, &vec![1.0, 0.0, 0.0, 1.0], gpu_state.clone());
        let bias_data = vec![10.0, 20.0];
        let bias = GpuMatrix::new(2, 1, &bias_data, gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let x = GpuMatrix::new(2, 2, &x_data, gpu_state.clone());
        let output = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_caches_z"),
                });
        layer.forward(&x, &output, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        // weights is identity here, so matmul(weights, x) == x.
        let cached_z = layer
            .cached_z
            .as_ref()
            .expect("cached_z should be set after forward");
        let cached_z_data = cached_z.to_cpu().await.unwrap();
        assert_eq!(
            cached_z_data, x_data,
            "cached_z should be pre-bias, i.e. equal to matmul result"
        );

        // output should have bias added, so it must differ from cached_z.
        let output_data = output.to_cpu().await.unwrap();
        assert_ne!(
            output_data, cached_z_data,
            "output should include bias, unlike cached_z"
        );
    }

    // Changing batch size should trigger a resize (new buffer), not reuse the old one.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_resizes_cache_on_batch_size_change() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let weights = GpuMatrix::new(2, 2, &vec![1.0, 0.0, 0.0, 1.0], gpu_state.clone());
        let bias = GpuMatrix::new(2, 1, &vec![0.0, 0.0], gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        // first forward, batch size 2
        let x1 = GpuMatrix::new(2, 2, &vec![1.0, 2.0, 3.0, 4.0], gpu_state.clone());
        let output1 = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let mut encoder1 =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_resize_1"),
                });
        layer.forward(&x1, &output1, &mut encoder1);
        gpu_state.gpu_context.queue.submit([encoder1.finish()]);

        // second forward, batch size 4 — must resize
        let x2 = GpuMatrix::new(2, 4, &vec![1.0; 8], gpu_state.clone());
        let output2 = GpuMatrix::new(2, 4, &vec![0.0; 8], gpu_state.clone());
        let mut encoder2 =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_resize_2"),
                });
        layer.forward(&x2, &output2, &mut encoder2);
        gpu_state.gpu_context.queue.submit([encoder2.finish()]);

        let cached_input = layer.cached_input.as_ref().unwrap();
        let cached_z = layer.cached_z.as_ref().unwrap();
        assert_eq!(cached_input.rows(), 2);
        assert_eq!(cached_input.columns(), 4);
        assert_eq!(cached_z.rows(), 2);
        assert_eq!(cached_z.columns(), 4);
    }
}
