use crate::matrix::Matrix;

pub trait Activation {
    fn forward(&self, mat: &Matrix) -> Matrix;
    fn backward(&self, mat: &Matrix) -> Matrix;
}

#[derive(Clone)]
pub struct ReLU;
#[derive(Clone)]
pub struct Sigmoid;
#[derive(Clone)]
pub struct Tanh;

impl Activation for ReLU {
    fn forward(&self, mat: &Matrix) -> Matrix {
        let new_data = mat.as_slice().into_iter().map(|x| x.max(0.0)).collect();
        Matrix::from_vec(mat.rows(), mat.columns(), new_data)
    }
    fn backward(&self, mat: &Matrix) -> Matrix {
        let new_data = mat
            .as_slice()
            .into_iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        Matrix::from_vec(mat.rows(), mat.columns(), new_data)
    }
}

impl Activation for Sigmoid {
    fn forward(&self, mat: &Matrix) -> Matrix {
        let new_data = mat
            .as_slice()
            .into_iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Matrix::from_vec(mat.rows(), mat.columns(), new_data)
    }
    fn backward(&self, mat: &Matrix) -> Matrix {
        let new_data = mat
            .as_slice()
            .into_iter()
            .map(|x| (-x).exp() / (1.0 + (-x).exp()).powf(2.0))
            .collect();
        Matrix::from_vec(mat.rows(), mat.columns(), new_data)
    }
}

impl Activation for Tanh {
    fn forward(&self, mat: &Matrix) -> Matrix {
        let new_data = mat.as_slice().into_iter().map(|x| x.tanh()).collect();
        Matrix::from_vec(mat.rows(), mat.columns(), new_data)
    }

    fn backward(&self, mat: &Matrix) -> Matrix {
        let new_data = mat
            .as_slice()
            .into_iter()
            .map(|x| 1.0 - x.tanh().powf(2.0))
            .collect();
        Matrix::from_vec(mat.rows(), mat.columns(), new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn relu_forward_hand_calculated() {
        let input = Matrix::from_vec(1, 3, vec![-2.0, 0.0, 3.0]);
        let expected = Matrix::from_vec(1, 3, vec![0.0, 0.0, 3.0]);

        let result = ReLU.forward(&input);

        assert_eq!(result, expected);
    }

    #[test]
    fn relu_backward_hand_calculated() {
        let input = Matrix::from_vec(1, 3, vec![-2.0, 0.0, 3.0]);
        let expected = Matrix::from_vec(1, 3, vec![0.0, 0.0, 1.0]);

        let result = ReLU.backward(&input);

        assert_eq!(result, expected);
    }

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
    fn sigmoid_forward_zero() {
        // sigmoid(0) = 1 / (1 + e^0) = 1 / 2 = 0.5, exactly representable
        let input = Matrix::from_vec(1, 1, vec![0.0]);
        let expected = Matrix::from_vec(1, 1, vec![0.5]);

        let result = Sigmoid.forward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn sigmoid_forward_hand_calculated() {
        // sigmoid(1)  ≈ 0.7310586
        // sigmoid(-1) ≈ 0.2689414   (symmetric to sigmoid(1): 1 - 0.7310586)
        // sigmoid(2)  ≈ 0.8807971
        let input = Matrix::from_vec(1, 3, vec![1.0, -1.0, 2.0]);
        let expected = Matrix::from_vec(1, 3, vec![0.7310586, 0.2689414, 0.8807971]);

        let result = Sigmoid.forward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn sigmoid_forward_output_bounded_between_zero_and_one() {
        // for very large/small inputs, output should approach 1.0 / 0.0
        // but never reach or exceed those bounds
        let input = Matrix::from_vec(1, 2, vec![100.0, -100.0]);

        let result = Sigmoid.forward(&input);

        assert!(result.as_slice()[0] > 0.999);
        assert!(result.as_slice()[0] <= 1.0);
        assert!(result.as_slice()[1] < 0.001);
        assert!(result.as_slice()[1] >= 0.0);
    }

    #[test]
    fn sigmoid_backward_hand_calculated() {
        // derivative: sigmoid(x) * (1 - sigmoid(x))
        // at x=0: 0.5 * 0.5 = 0.25
        // at x=1: 0.7310586 * (1 - 0.7310586) ≈ 0.1966119
        let input = Matrix::from_vec(1, 2, vec![0.0, 1.0]);
        let expected = Matrix::from_vec(1, 2, vec![0.25, 0.1966119]);

        let result = Sigmoid.backward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn tanh_forward_zero() {
        // tanh(0) = 0, exactly representable
        let input = Matrix::from_vec(1, 1, vec![0.0]);
        let expected = Matrix::from_vec(1, 1, vec![0.0]);

        let result = Tanh.forward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn tanh_forward_hand_calculated() {
        // tanh(1)  ≈ 0.7615942
        // tanh(-1) ≈ -0.7615942 (odd function: tanh(-x) = -tanh(x))
        // tanh(2)  ≈ 0.9640276
        let input = Matrix::from_vec(1, 3, vec![1.0, -1.0, 2.0]);
        let expected = Matrix::from_vec(1, 3, vec![0.7615942, -0.7615942, 0.9640276]);

        let result = Tanh.forward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn tanh_forward_output_bounded_between_minus_one_and_one() {
        // for very large/small inputs, output should approach 1.0 / -1.0
        // but never reach or exceed those bounds
        let input = Matrix::from_vec(1, 2, vec![100.0, -100.0]);

        let result = Tanh.forward(&input);

        assert!(result.as_slice()[0] > 0.999);
        assert!(result.as_slice()[0] <= 1.0);
        assert!(result.as_slice()[1] < -0.999);
        assert!(result.as_slice()[1] >= -1.0);
    }

    #[test]
    fn tanh_backward_hand_calculated() {
        // derivative: 1 - tanh(x)^2
        // at x=0: 1 - 0^2 = 1.0 (maximum slope, at the origin)
        // at x=1: 1 - 0.7615942^2 ≈ 0.4199743
        let input = Matrix::from_vec(1, 2, vec![0.0, 1.0]);
        let expected = Matrix::from_vec(1, 2, vec![1.0, 0.4199743]);

        let result = Tanh.backward(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn tanh_backward_is_even_function() {
        // derivative is even: f'(x) == f'(-x), since it depends on tanh(x)^2
        let positive = Matrix::from_vec(1, 1, vec![1.0]);
        let negative = Matrix::from_vec(1, 1, vec![-1.0]);

        let result_pos = Tanh.backward(&positive);
        let result_neg = Tanh.backward(&negative);

        assert_matrix_approx_eq(&result_pos, &result_neg, 1e-6);
    }
}
