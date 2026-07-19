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
}
