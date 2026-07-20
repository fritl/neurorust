use crate::matrix::Matrix;

pub trait Loss {
    fn forward(&self, pred: &Matrix, target: &Matrix) -> f32;
    fn backward(&self, pred: &Matrix, target: &Matrix) -> Matrix;
}

struct MSE;
struct SoftmaxCrossEntropy;

impl Loss for MSE {
    fn forward(&self, pred: &Matrix, target: &Matrix) -> f32 {
        assert_eq!(pred.shape(), target.shape());
        let diff = pred - target;
        let squared = &diff.hadamard(&diff);
        squared.as_slice().iter().sum::<f32>() / (pred.rows() * pred.columns()) as f32
    }
    fn backward(&self, pred: &Matrix, target: &Matrix) -> Matrix {
        assert_eq!(pred.shape(), target.shape());
        let diff = pred - target;
        &diff * (2.0 / (diff.rows() * diff.columns()) as f32)
    }
}

impl SoftmaxCrossEntropy {
    fn softmax(&self, mat: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(mat.rows(), mat.columns());
        for col in 0..mat.columns() {
            let mut max_val = f32::NEG_INFINITY;
            for row in 0..mat.rows() {
                max_val = mat.get(row, col).max(max_val);
            }

            let mut sum = 0.0;
            let mut exponents = vec![0.0; mat.rows()];
            for row in 0..mat.rows() {
                let e = (mat.get(row, col) - max_val).exp();
                sum += e;
                exponents[row] = e;
            }

            for row in 0..mat.rows() {
                result.set(row, col, exponents[row] / sum);
            }
        }
        result
    }
}

impl Loss for SoftmaxCrossEntropy {
    fn forward(&self, pred: &Matrix, target: &Matrix) -> f32 {
        let mut pred = self.softmax(pred);
        pred.as_mut_slice().iter_mut().for_each(|x| *x = x.ln());
        let mut sum = 0.0;
        pred.as_slice()
            .iter()
            .zip(target.as_slice())
            .for_each(|(x, y)| sum += x * y);
        -sum / pred.columns() as f32
    }

    fn backward(&self, pred: &Matrix, target: &Matrix) -> Matrix {
        &self.softmax(pred) - target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

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
    fn mse_forward_identical_values_is_zero() {
        // prediction == target for every element -> loss must be exactly 0
        // shape: 2 outputs, batch of 2
        let prediction = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let target = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let loss = MSE.forward(&prediction, &target);

        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mse_forward_hand_calculated_single_example() {
        // 3 output neurons, batch of 1
        // errors: 3-1=2, 5-4=1, 0-2=-2
        // squared: 4, 1, 4 -> mean = 9/3 = 3.0
        let prediction = Matrix::from_vec(3, 1, vec![3.0, 5.0, 0.0]);
        let target = Matrix::from_vec(3, 1, vec![1.0, 4.0, 2.0]);

        let loss = MSE.forward(&prediction, &target);

        assert!((loss - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mse_forward_hand_calculated_batch() {
        // 2 output neurons, batch of 2 examples
        // column 1 (example 1): predictions [3.0, 0.0], targets [1.0, 2.0]
        // column 2 (example 2): predictions [5.0, 1.0], targets [4.0, 1.0]
        // errors: 2, -2, 1, 0
        // squared: 4, 4, 1, 0 -> mean = 9/4 = 2.25
        let prediction = Matrix::from_vec(2, 2, vec![3.0, 5.0, 0.0, 1.0]);
        let target = Matrix::from_vec(2, 2, vec![1.0, 4.0, 2.0, 1.0]);

        let loss = MSE.forward(&prediction, &target);

        assert!((loss - 2.25).abs() < 1e-6);
    }

    #[test]
    fn mse_backward_identical_values_is_zero() {
        // gradient should be zero everywhere when prediction matches target exactly
        let prediction = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let target = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let gradient = MSE.backward(&prediction, &target);

        for &value in gradient.as_slice() {
            assert!((value - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn mse_backward_hand_calculated_single_example() {
        // derivative: 2 * (prediction - target) / n, n = 3 (total element count)
        // errors: 3-1=2, 5-4=1, 0-2=-2
        // gradient: 2*2/3 ≈ 1.333333, 2*1/3 ≈ 0.666667, 2*(-2)/3 ≈ -1.333333
        let prediction = Matrix::from_vec(3, 1, vec![3.0, 5.0, 0.0]);
        let target = Matrix::from_vec(3, 1, vec![1.0, 4.0, 2.0]);
        let expected = Matrix::from_vec(3, 1, vec![1.333333, 0.666667, -1.333333]);

        let gradient = MSE.backward(&prediction, &target);

        assert_matrix_approx_eq(&gradient, &expected, 1e-5);
    }

    #[test]
    fn mse_backward_hand_calculated_batch() {
        // 2 output neurons, batch of 2 examples, n = 4 total elements
        // errors: 2, -2, 1, 0
        // gradient: 2*2/4=1.0, 2*(-2)/4=-1.0, 2*1/4=0.5, 2*0/4=0.0
        let prediction = Matrix::from_vec(2, 2, vec![3.0, 5.0, 0.0, 1.0]);
        let target = Matrix::from_vec(2, 2, vec![1.0, 4.0, 2.0, 1.0]);
        let expected = Matrix::from_vec(2, 2, vec![1.0, 0.5, -1.0, 0.0]);

        let gradient = MSE.backward(&prediction, &target);
        println!("{gradient:?}");

        assert_matrix_approx_eq(&gradient, &expected, 1e-6);
    }

    #[test]
    fn mse_backward_output_has_same_shape_as_input() {
        // gradient shape must match prediction/target shape, regardless of batch size
        let prediction = Matrix::from_vec(3, 4, vec![0.0; 12]);
        let target = Matrix::from_vec(3, 4, vec![0.0; 12]);

        let gradient = MSE.backward(&prediction, &target);

        assert_eq!(gradient.shape(), (3, 4));
    }

    #[test]
    fn mse_forward_is_never_negative() {
        // squared errors can never produce a negative loss, regardless of input signs
        let prediction = Matrix::from_vec(1, 2, vec![-5.0, 100.0]);
        let target = Matrix::from_vec(1, 2, vec![10.0, -50.0]);

        let loss = MSE.forward(&prediction, &target);

        assert!(loss >= 0.0);
    }

    #[test]
    fn softmax_hand_calculated_single_column() {
        // input: [1.0, 2.0, 3.0], single example (1 column)
        // exp(1) ≈ 2.718282, exp(2) ≈ 7.389056, exp(3) ≈ 20.085537
        // sum ≈ 30.192875
        // softmax ≈ [0.090031, 0.244728, 0.665241]
        let input = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        let expected = Matrix::from_vec(3, 1, vec![0.090031, 0.244728, 0.665241]);

        let result = SoftmaxCrossEntropy.softmax(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        // regardless of input values, each column must sum to (approximately) 1.0
        let input = Matrix::from_vec(4, 1, vec![-3.0, 0.5, 2.2, 10.0]);

        let result = SoftmaxCrossEntropy.softmax(&input);

        let sum: f32 = result.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_all_equal_inputs_are_uniform() {
        // if all inputs are the same, softmax must distribute probability equally
        let input = Matrix::from_vec(4, 1, vec![5.0, 5.0, 5.0, 5.0]);
        let expected = Matrix::from_vec(4, 1, vec![0.25, 0.25, 0.25, 0.25]);

        let result = SoftmaxCrossEntropy.softmax(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn softmax_preserves_order() {
        // the largest input should map to the largest output probability
        let input = Matrix::from_vec(3, 1, vec![1.0, 5.0, 2.0]);

        let result = SoftmaxCrossEntropy.softmax(&input);
        let values = result.as_slice();

        assert!(values[1] > values[0]);
        assert!(values[1] > values[2]);
    }

    #[test]
    fn softmax_is_shift_invariant() {
        // softmax(x) == softmax(x + c) for any constant c
        // this is the property the max-subtraction trick relies on for numerical stability
        let input = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        let shifted = Matrix::from_vec(3, 1, vec![101.0, 102.0, 103.0]);

        let result = SoftmaxCrossEntropy.softmax(&input);
        let result_shifted = SoftmaxCrossEntropy.softmax(&shifted);

        assert_matrix_approx_eq(&result, &result_shifted, 1e-5);
    }

    #[test]
    fn softmax_handles_large_values_without_overflow() {
        // without the max-subtraction trick, exp(1000) would overflow to infinity
        let input = Matrix::from_vec(2, 1, vec![1000.0, 1001.0]);

        let result = SoftmaxCrossEntropy.softmax(&input);

        for &value in result.as_slice() {
            assert!(value.is_finite(), "softmax produced a non-finite value");
        }
        let sum: f32 = result.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_multiple_columns_independent() {
        // each column (batch example) should be normalized independently
        let input = Matrix::from_vec(2, 2, vec![1.0, 10.0, 1.0, 10.0]);
        // column 0: [1.0, 1.0] -> equal -> [0.5, 0.5]
        // column 1: [10.0, 10.0] -> equal -> [0.5, 0.5]
        let expected = Matrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);

        let result = SoftmaxCrossEntropy.softmax(&input);

        assert_matrix_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn cross_entropy_forward_hand_calculated_single_example() {
        // logits: [1.0, 2.0, 3.0] -> softmax ≈ [0.090031, 0.244728, 0.665241]
        // one-hot target: class 2 is correct
        // loss = -ln(0.665241) ≈ 0.407606
        let prediction = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        let target = Matrix::from_vec(3, 1, vec![0.0, 0.0, 1.0]);

        let loss = SoftmaxCrossEntropy.forward(&prediction, &target);

        assert!((loss - 0.407606).abs() < 1e-5);
    }

    #[test]
    fn cross_entropy_forward_low_when_confident_and_correct() {
        // a very large logit for the correct class should push loss close to 0
        let prediction = Matrix::from_vec(3, 1, vec![0.0, 0.0, 20.0]);
        let target = Matrix::from_vec(3, 1, vec![0.0, 0.0, 1.0]);

        let loss = SoftmaxCrossEntropy.forward(&prediction, &target);

        assert!(
            loss < 0.001,
            "expected near-zero loss for confident correct prediction, got {}",
            loss
        );
    }

    #[test]
    fn cross_entropy_forward_high_when_confident_and_wrong() {
        // a very large logit for the WRONG class should produce a large loss
        let prediction = Matrix::from_vec(3, 1, vec![0.0, 0.0, 20.0]);
        let target = Matrix::from_vec(3, 1, vec![1.0, 0.0, 0.0]);

        let loss = SoftmaxCrossEntropy.forward(&prediction, &target);

        assert!(
            loss > 10.0,
            "expected large loss for confident wrong prediction, got {}",
            loss
        );
    }

    #[test]
    fn cross_entropy_forward_is_never_negative() {
        // cross-entropy loss can never be negative, regardless of input
        let prediction = Matrix::from_vec(3, 1, vec![-5.0, 2.0, 0.3]);
        let target = Matrix::from_vec(3, 1, vec![1.0, 0.0, 0.0]);

        let loss = SoftmaxCrossEntropy.forward(&prediction, &target);

        assert!(loss >= 0.0);
    }

    #[test]
    fn cross_entropy_backward_hand_calculated() {
        // gradient simplifies to: softmax(prediction) - target
        // logits: [1.0, 2.0, 3.0] -> softmax ≈ [0.090031, 0.244728, 0.665241]
        // target (one-hot, class 2): [0.0, 0.0, 1.0]
        // gradient ≈ [0.090031, 0.244728, -0.334759]
        let prediction = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        let target = Matrix::from_vec(3, 1, vec![0.0, 0.0, 1.0]);
        let expected = Matrix::from_vec(3, 1, vec![0.090031, 0.244728, -0.334759]);

        let gradient = SoftmaxCrossEntropy.backward(&prediction, &target);

        assert_matrix_approx_eq(&gradient, &expected, 1e-5);
    }

    #[test]
    fn cross_entropy_backward_zero_when_prediction_matches_target_confidently() {
        // near-perfect confident correct prediction -> gradient should be near zero
        let prediction = Matrix::from_vec(3, 1, vec![0.0, 0.0, 20.0]);
        let target = Matrix::from_vec(3, 1, vec![0.0, 0.0, 1.0]);

        let gradient = SoftmaxCrossEntropy.backward(&prediction, &target);

        for &value in gradient.as_slice() {
            assert!(
                value.abs() < 0.001,
                "expected near-zero gradient, got {}",
                value
            );
        }
    }

    #[test]
    fn cross_entropy_backward_output_has_same_shape_as_input() {
        let prediction = Matrix::from_vec(4, 3, vec![0.0; 12]);
        let target = Matrix::from_vec(4, 3, vec![0.0; 12]);

        let gradient = SoftmaxCrossEntropy.backward(&prediction, &target);

        assert_eq!(gradient.shape(), (4, 3));
    }

    #[test]
    fn cross_entropy_forward_batch_hand_calculated() {
        // 2 columns (batch of 2), same logits per example for simplicity
        // each example: loss ≈ 0.407606 (from single-example test above)
        // averaged over batch: still ≈ 0.407606
        let prediction = Matrix::from_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let target = Matrix::from_vec(3, 2, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let loss = SoftmaxCrossEntropy.forward(&prediction, &target);

        println!("{loss}");
        assert!((loss - 0.407606).abs() < 1e-5);
    }
}
