use rand::distr;
use rand::distr::Distribution;
use rand::{RngExt, SeedableRng};

use crate::matrix::Matrix;

use super::activations;

struct Layer<A: activations::Activation> {
    activation: A,
    weights: Matrix,
    bias: Matrix,
    cached_input: Option<Matrix>,
    cached_z: Option<Matrix>,
}

impl<A: activations::Activation> Layer<A> {
    pub fn from_manual_weights(weights: Matrix, bias: Matrix, activation: A) -> Layer<A> {
        assert_eq!(weights.rows(), bias.rows());
        Layer {
            activation,
            weights,
            bias,
            cached_z: None,
            cached_input: None,
        }
    }

    pub fn with_random_weights(
        input_neurons: usize,
        output_nurons: usize,
        activation: A,
        seed: Option<u64>,
    ) -> Layer<A> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(3001));
        let weight_data = (0..input_neurons * output_nurons)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect::<Vec<_>>();
        let weights = Matrix::from_vec(output_nurons, input_neurons, weight_data);
        let bias = Matrix::zeros(output_nurons, 1);

        Layer {
            activation,
            weights,
            bias,
            cached_input: None,
            cached_z: None,
        }
    }

    pub fn with_uniform_xavier_weights(
        input_neurons: usize,
        output_nurons: usize,
        activation: A,
        seed: Option<u64>,
    ) -> Layer<A> {
        let x = (6.0 / (input_neurons + output_nurons) as f32).sqrt();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(3001));
        let dist = distr::Uniform::new(-x, x).unwrap();
        let weight_data = (0..input_neurons * output_nurons)
            .map(|_| dist.sample(&mut rng))
            .collect::<Vec<_>>();
        let weights = Matrix::from_vec(output_nurons, input_neurons, weight_data);
        let bias = Matrix::zeros(output_nurons, 1);

        Layer {
            activation,
            weights,
            bias,
            cached_input: None,
            cached_z: None,
        }
    }

    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let z = &(&self.weights * x) + &self.bias;
        let a = self.activation.forward(&z);
        self.cached_input = Some(x.clone());
        self.cached_z = Some(z);
        a
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::activations::{ReLU, Sigmoid};

    fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tolerance: f32) {
        assert_eq!(a.shape(), b.shape(), "Shape mismatch");
        for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
            assert!(
                (x - y).abs() < tolerance,
                "Values differ: {} vs {} (tolerance {})",
                x,
                y,
                tolerance
            );
        }
    }

    #[test]
    fn single_layer_forward_xor_input_hand_calculated() {
        let input = Matrix::from_vec(2, 1, vec![1.0, 0.0]);
        let weights = Matrix::from_vec(2, 2, vec![0.5, -0.5, 1.0, 0.5]);
        let bias = Matrix::from_vec(2, 1, vec![0.1, -0.2]);
        let expected = Matrix::from_vec(2, 1, vec![0.6, 0.8]);

        let mut layer = Layer::from_manual_weights(weights, bias, ReLU);
        let result = layer.forward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn chained_two_layer_forward_xor_input_hand_calculated() {
        let input = Matrix::from_vec(2, 1, vec![1.0, 0.0]);

        let hidden_weights = Matrix::from_vec(2, 2, vec![0.5, -0.5, 1.0, 0.5]);
        let hidden_bias = Matrix::from_vec(2, 1, vec![0.1, -0.2]);
        let mut hidden_layer = Layer::from_manual_weights(hidden_weights, hidden_bias, ReLU);

        let output_weights = Matrix::from_vec(1, 2, vec![0.3, -0.7]);
        let output_bias = Matrix::from_vec(1, 1, vec![0.05]);
        let mut output_layer = Layer::from_manual_weights(output_weights, output_bias, Sigmoid);

        let expected = Matrix::from_vec(1, 1, vec![0.418240]);

        let hidden_output = hidden_layer.forward(&input);
        let result = output_layer.forward(&hidden_output);

        assert_matrix_approx_eq(&result, &expected, 1e-5);
    }
    #[test]
    fn with_random_weights_has_correct_shape() {
        let layer = Layer::with_random_weights(3, 2, ReLU, Some(1));

        assert_eq!(layer.weights.shape(), (2, 3));
        assert_eq!(layer.bias.shape(), (2, 1));
    }

    #[test]
    fn with_random_weights_values_within_range() {
        let layer = Layer::with_random_weights(10, 10, ReLU, Some(1));

        for &value in layer.weights.as_slice() {
            assert!(value >= -0.5 && value < 0.5, "value {} out of range", value);
        }
    }

    #[test]
    fn with_random_weights_bias_initialized_to_zero() {
        let layer = Layer::with_random_weights(3, 2, ReLU, Some(1));

        for &value in layer.bias.as_slice() {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn with_random_weights_same_seed_produces_same_weights() {
        let layer_a = Layer::with_random_weights(5, 4, ReLU, Some(42));
        let layer_b = Layer::with_random_weights(5, 4, ReLU, Some(42));

        assert_eq!(layer_a.weights.as_slice(), layer_b.weights.as_slice());
    }

    #[test]
    fn with_random_weights_different_seed_produces_different_weights() {
        let layer_a = Layer::with_random_weights(5, 4, ReLU, Some(1));
        let layer_b = Layer::with_random_weights(5, 4, ReLU, Some(2));

        assert_ne!(layer_a.weights.as_slice(), layer_b.weights.as_slice());
    }

    #[test]
    fn with_random_weights_default_seed_is_reproducible() {
        // no seed given -> falls back to the same default seed both times
        let layer_a = Layer::with_random_weights(5, 4, ReLU, None);
        let layer_b = Layer::with_random_weights(5, 4, ReLU, None);

        assert_eq!(layer_a.weights.as_slice(), layer_b.weights.as_slice());
    }

    #[test]
    fn with_uniform_xavier_weights_has_correct_shape() {
        let layer = Layer::with_uniform_xavier_weights(3, 2, ReLU, Some(1));

        assert_eq!(layer.weights.shape(), (2, 3));
        assert_eq!(layer.bias.shape(), (2, 1));
    }

    #[test]
    fn with_uniform_xavier_weights_values_within_expected_range() {
        let input_neurons = 10;
        let output_neurons = 10;
        let expected_limit = (6.0 / (input_neurons + output_neurons) as f32).sqrt();

        let layer =
            Layer::with_uniform_xavier_weights(input_neurons, output_neurons, ReLU, Some(1));

        for &value in layer.weights.as_slice() {
            assert!(
                value >= -expected_limit && value < expected_limit,
                "value {} outside expected xavier range [{}, {})",
                value,
                -expected_limit,
                expected_limit
            );
        }
    }

    #[test]
    fn with_uniform_xavier_weights_bias_initialized_to_zero() {
        let layer = Layer::with_uniform_xavier_weights(3, 2, ReLU, Some(1));

        for &value in layer.bias.as_slice() {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn with_uniform_xavier_weights_same_seed_produces_same_weights() {
        let layer_a = Layer::with_uniform_xavier_weights(5, 4, ReLU, Some(42));
        let layer_b = Layer::with_uniform_xavier_weights(5, 4, ReLU, Some(42));

        assert_eq!(layer_a.weights.as_slice(), layer_b.weights.as_slice());
    }

    #[test]
    fn with_uniform_xavier_weights_range_shrinks_with_larger_layer() {
        // larger n_in + n_out should produce a smaller xavier limit
        let small_layer = Layer::with_uniform_xavier_weights(2, 2, ReLU, Some(1));
        let large_layer = Layer::with_uniform_xavier_weights(1000, 1000, ReLU, Some(1));

        let small_max = small_layer
            .weights
            .as_slice()
            .iter()
            .cloned()
            .fold(0.0_f32, |a, b| a.max(b.abs()));
        let large_max = large_layer
            .weights
            .as_slice()
            .iter()
            .cloned()
            .fold(0.0_f32, |a, b| a.max(b.abs()));

        assert!(
            large_max < small_max,
            "expected xavier range to shrink for larger layers"
        );
    }
}
