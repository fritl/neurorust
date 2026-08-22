use rand::RngExt;
use rand::distr;
use rand::distr::Distribution;

use std::rc::Rc;

use crate::gpu::buffer_pool::BufferPool;
use crate::gpu::{matrix::GpuMatrix, state::GpuState};

pub struct Layer {
    weights: GpuMatrix,
    bias: GpuMatrix,
    pub cached_a: Option<GpuMatrix>,
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
            cached_a: None,
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
            cached_a: None,
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
            cached_a: None,
            gpu_state,
        }
    }

    pub fn weights(&self) -> &GpuMatrix {
        &self.weights
    }

    fn ensure_buffer(
        buffer: &mut Option<GpuMatrix>,
        rows: usize,
        columns: usize,
        gpu_state: &Rc<GpuState>,
    ) {
        match &buffer {
            Some(i) if rows == i.rows() && columns == i.columns() => {}
            _ => *buffer = Some(GpuMatrix::empty(rows, columns, Rc::clone(gpu_state))),
        }
    }

    pub fn forward(&mut self, x: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        Self::ensure_buffer(
            &mut self.cached_input,
            x.rows(),
            x.columns(),
            &self.gpu_state,
        );
        encoder.copy_buffer_to_buffer(
            x.gpu_buffer(),
            0,
            self.cached_input.as_ref().unwrap().gpu_buffer(),
            0,
            x.gpu_buffer().size(),
        );

        Self::ensure_buffer(
            &mut self.cached_a,
            self.weights.rows(),
            x.columns(),
            &self.gpu_state,
        );
        let cached_a = self
            .cached_a
            .as_ref()
            .expect("Just ensured a buffer exists");
        GpuMatrix::matmul(&self.weights, x, &cached_a, encoder);
        cached_a.add_assign(&self.bias, encoder);
        cached_a.sigmoid(encoder);
    }

    pub fn backward(
        &mut self,
        delta: &GpuMatrix,
        learning_rate: f32,
        output: &GpuMatrix,
        buffer_pool: &mut BufferPool,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let sigmoid_prime = buffer_pool.get(self.weights.rows(), delta.columns());
        self.cached_a
            .as_ref()
            .expect("Run forward before backward")
            .sigmoid_prime(&sigmoid_prime, encoder);

        let weights_shape = self.weights.shape();
        let local_delta = buffer_pool.get(weights_shape.0, delta.columns());
        GpuMatrix::hadamard(delta, &sigmoid_prime, &local_delta, encoder);

        let row_sum = buffer_pool.get(delta.rows(), 1);
        GpuMatrix::row_sum(&local_delta, &row_sum, encoder);

        self.bias
            .subtract_assign_scaled(&row_sum, learning_rate, encoder);
        let cached_input_shape = self
            .cached_input
            .as_ref()
            .expect("Run forward pass before backward")
            .shape();
        let transposed_input = buffer_pool.get(cached_input_shape.1, cached_input_shape.0);
        self.cached_input
            .as_ref()
            .expect("Run forward beofre backward")
            .transpose(&transposed_input, encoder);

        let weight_update = buffer_pool.get(weights_shape.0, weights_shape.1);
        GpuMatrix::matmul(&local_delta, &transposed_input, &weight_update, encoder);

        let transposed_weights = buffer_pool.get(weights_shape.1, weights_shape.0);
        self.weights.transpose(&transposed_weights, encoder);
        GpuMatrix::matmul(&transposed_weights, &local_delta, output, encoder);
        self.weights
            .subtract_assign_scaled(&weight_update, learning_rate, encoder);
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
    async fn forward_computes_matmul_plus_bias_then_sigmoid() {
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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward"),
                });
        layer.forward(&x, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let cached_a = layer
            .cached_a
            .as_ref()
            .expect("cached_a should be set after forward");
        let result_actual = cached_a.to_cpu().await.unwrap();

        // matmul(weights, x): weights (3x2): [[1,0],[0,1],[1,1]], x (2x2): [[1,2],[3,4]]
        // matmul result (3x2): [[1,2],[3,4],[4,6]]
        // + bias (broadcast per row): [[2,3],[5,6],[7,9]]
        // sigmoid(z) = 1 / (1 + e^-z) applied elementwise
        let linear = [2.0f32, 3.0, 5.0, 6.0, 7.0, 9.0];
        let expected: Vec<f32> = linear.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();

        for (actual, expected) in result_actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
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

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_caches_input"),
                });
        layer.forward(&x, &mut encoder);
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

    // cached_a holds the layer's activation output (post-bias, post-sigmoid).
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn forward_caches_sigmoid_output() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let weights = GpuMatrix::new(2, 2, &vec![1.0, 0.0, 0.0, 1.0], gpu_state.clone());
        let bias_data = vec![10.0, 20.0];
        let bias = GpuMatrix::new(2, 1, &bias_data, gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let x = GpuMatrix::new(2, 2, &x_data, gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_caches_a"),
                });
        layer.forward(&x, &mut encoder);
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let cached_a = layer
            .cached_a
            .as_ref()
            .expect("cached_a should be set after forward");
        let cached_a_data = cached_a.to_cpu().await.unwrap();

        // Sanity check: sigmoid squashes every value into the closed interval [0, 1]
        // (closed rather than open due to f32 precision at extreme magnitudes).
        for value in cached_a_data {
            assert!(
                value >= 0.0 && value <= 1.0,
                "sigmoid output {value} outside [0, 1]"
            );
        }
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
        let mut encoder1 =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_resize_1"),
                });
        layer.forward(&x1, &mut encoder1);
        gpu_state.gpu_context.queue.submit([encoder1.finish()]);

        // second forward, batch size 4 — must resize
        let x2 = GpuMatrix::new(2, 4, &vec![1.0; 8], gpu_state.clone());
        let mut encoder2 =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_forward_resize_2"),
                });
        layer.forward(&x2, &mut encoder2);
        gpu_state.gpu_context.queue.submit([encoder2.finish()]);

        let cached_input = layer.cached_input.as_ref().unwrap();
        let cached_a = layer.cached_a.as_ref().unwrap();
        assert_eq!(cached_input.rows(), 2);
        assert_eq!(cached_input.columns(), 4);
        assert_eq!(cached_a.rows(), 2);
        assert_eq!(cached_a.columns(), 4);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn backward_matches_cpu_reference() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        // 2 input neurons, 2 output neurons, batch size 2.
        let weight_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2, row-major
        let bias_data = vec![0.5, -0.5]; // 2x1
        let weights = GpuMatrix::new(2, 2, &weight_data, gpu_state.clone());
        let bias = GpuMatrix::new(2, 1, &bias_data, gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        let x_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 (2 input neurons, batch 2)
        let x = GpuMatrix::new(2, 2, &x_data, gpu_state.clone());

        let delta_data = vec![0.1, 0.2, 0.3, 0.4]; // 2x2 (out neurons x batch)
        let delta = GpuMatrix::new(2, 2, &delta_data, gpu_state.clone());
        let backward_out = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone()); // in neurons x batch
        let learning_rate = 0.1;
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_backward"),
                });
        layer.forward(&x, &mut encoder);
        layer.backward(
            &delta,
            learning_rate,
            &backward_out,
            &mut buffer_pool,
            &mut encoder,
        );
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual_weights = layer.weights.to_cpu().await.unwrap();
        let actual_bias = layer.bias.to_cpu().await.unwrap();
        let actual_output = backward_out.to_cpu().await.unwrap();

        // ---- CPU reference, computed independently of the GPU kernels ----

        fn cpu_matmul(
            a: &[f32],
            a_rows: usize,
            a_cols: usize,
            b: &[f32],
            b_cols: usize,
        ) -> Vec<f32> {
            let mut out = vec![0.0f32; a_rows * b_cols];
            for r in 0..a_rows {
                for c in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a[r * a_cols + k] * b[k * b_cols + c];
                    }
                    out[r * b_cols + c] = sum;
                }
            }
            out
        }

        fn cpu_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = a[r * cols + c];
                }
            }
            out
        }

        fn cpu_sigmoid(z: &[f32]) -> Vec<f32> {
            z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
        }

        // z = weights * x + bias (broadcast)
        let z = cpu_matmul(&weight_data, 2, 2, &x_data, 2);
        let z_with_bias: Vec<f32> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v + bias_data[i / 2]) // row i/2, broadcast bias per row
            .collect();
        let a = cpu_sigmoid(&z_with_bias); // cached_a

        // derivative = a * (1 - a)
        let derivative: Vec<f32> = a.iter().map(|v| v * (1.0 - v)).collect();
        // local_delta = delta ⊙ derivative
        let local_delta: Vec<f32> = delta_data
            .iter()
            .zip(derivative.iter())
            .map(|(d, s)| d * s)
            .collect();

        // bias_update = row_sum(local_delta), 2 rows x 2 cols -> 2 values
        let bias_update = vec![
            local_delta[0] + local_delta[1],
            local_delta[2] + local_delta[3],
        ];
        let expected_bias: Vec<f32> = bias_data
            .iter()
            .zip(bias_update.iter())
            .map(|(b, u)| b - u * learning_rate)
            .collect();

        // weight_update = local_delta * x^T
        let x_t = cpu_transpose(&x_data, 2, 2);
        let weight_update = cpu_matmul(&local_delta, 2, 2, &x_t, 2);
        let expected_weights: Vec<f32> = weight_data
            .iter()
            .zip(weight_update.iter())
            .map(|(w, u)| w - u * learning_rate)
            .collect();

        // output = weights^T * local_delta, using the ORIGINAL (pre-update) weights
        let weights_t = cpu_transpose(&weight_data, 2, 2);
        let expected_output = cpu_matmul(&weights_t, 2, 2, &local_delta, 2);

        for (actual, expected) in actual_bias.iter().zip(expected_bias.iter()) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "bias mismatch: expected {expected}, got {actual}"
            );
        }
        for (actual, expected) in actual_weights.iter().zip(expected_weights.iter()) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "weights mismatch: expected {expected}, got {actual}"
            );
        }
        for (actual, expected) in actual_output.iter().zip(expected_output.iter()) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "output delta mismatch: expected {expected}, got {actual}"
            );
        }
    }

    // Regression test for the ordering bug: the returned delta must use the
    // ORIGINAL weights (as used during forward), not the weights already
    // updated earlier in the same backward() call.
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    async fn backward_uses_pre_update_weights_for_returned_delta() {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let gpu_state = Rc::new(GpuState::default().await);

        let weight_data = vec![10.0, -5.0, 3.0, 8.0]; // 2x2
        let bias_data = vec![0.0, 0.0];
        let weights = GpuMatrix::new(2, 2, &weight_data, gpu_state.clone());
        let bias = GpuMatrix::new(2, 1, &bias_data, gpu_state.clone());
        let mut layer = Layer::from_manual_weights(weights, bias, gpu_state.clone());

        let x_data = vec![1.0, 0.5, -0.5, 1.0];
        let x = GpuMatrix::new(2, 2, &x_data, gpu_state.clone());

        let delta_data = vec![1.0, 1.0, 1.0, 1.0];
        let delta = GpuMatrix::new(2, 2, &delta_data, gpu_state.clone());
        let backward_out = GpuMatrix::new(2, 2, &vec![0.0; 4], gpu_state.clone());
        let learning_rate = 1.0; // large, to make pre/post-update weights diverge sharply
        let mut buffer_pool = BufferPool::new(gpu_state.clone());

        let mut encoder =
            gpu_state
                .gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_backward_ordering"),
                });
        layer.forward(&x, &mut encoder);
        layer.backward(
            &delta,
            learning_rate,
            &backward_out,
            &mut buffer_pool,
            &mut encoder,
        );
        gpu_state.gpu_context.queue.submit([encoder.finish()]);

        let actual_output = backward_out.to_cpu().await.unwrap();

        fn cpu_matmul(
            a: &[f32],
            a_rows: usize,
            a_cols: usize,
            b: &[f32],
            b_cols: usize,
        ) -> Vec<f32> {
            let mut out = vec![0.0f32; a_rows * b_cols];
            for r in 0..a_rows {
                for c in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a[r * a_cols + k] * b[k * b_cols + c];
                    }
                    out[r * b_cols + c] = sum;
                }
            }
            out
        }
        fn cpu_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = a[r * cols + c];
                }
            }
            out
        }
        fn cpu_sigmoid(z: &[f32]) -> Vec<f32> {
            z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
        }

        let z = cpu_matmul(&weight_data, 2, 2, &x_data, 2);
        let z_with_bias: Vec<f32> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v + bias_data[i / 2])
            .collect();
        let a = cpu_sigmoid(&z_with_bias);
        let derivative: Vec<f32> = a.iter().map(|v| v * (1.0 - v)).collect();
        let local_delta: Vec<f32> = delta_data
            .iter()
            .zip(derivative.iter())
            .map(|(d, s)| d * s)
            .collect();

        // Correct: transpose of the ORIGINAL weights, before any update.
        let weights_t = cpu_transpose(&weight_data, 2, 2);
        let expected_output = cpu_matmul(&weights_t, 2, 2, &local_delta, 2);

        for (actual, expected) in actual_output.iter().zip(expected_output.iter()) {
            assert!(
                (actual - expected).abs() < 1e-3,
                "returned delta used the wrong weights (likely post-update instead of pre-update): expected {expected}, got {actual}"
            );
        }
    }
}
