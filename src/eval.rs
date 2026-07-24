use crate::matrix::Matrix;

pub fn accuracy(y_pred: &Matrix, y_true: &Matrix) -> f32 {
    assert_eq!(y_pred.shape(), y_true.shape());
    let mut num_correct = 0.0;
    for i in 0..y_pred.columns() {
        let pred = (0..y_pred.rows())
            .map(|j| y_pred.get(j, i))
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let correct = (0..y_true.rows())
            .map(|j| y_true.get(j, i))
            .enumerate()
            .filter(|(_, a)| *a == 1.0)
            .map(|(idx, _)| idx)
            .next()
            .unwrap();
        if correct == pred {
            num_correct += 1.0;
        }
    }
    num_correct / y_pred.columns() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // All predictions correct
    #[test]
    fn accuracy_all_correct() {
        let y_true = Matrix::from_vec(
            3,
            4,
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        );
        let y_pred = Matrix::from_vec(
            3,
            4,
            vec![0.1, 0.7, 0.05, 0.2, 0.8, 0.2, 0.15, 0.6, 0.1, 0.1, 0.8, 0.2],
        );
        assert_eq!(accuracy(&y_pred, &y_true), 1.0);
    }

    // All predictions wrong
    #[test]
    fn accuracy_all_wrong() {
        let y_true = Matrix::from_vec(
            3,
            4,
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        );
        let y_pred = Matrix::from_vec(
            3,
            4,
            vec![
                0.9, 0.1, 0.9, 0.9, 0.05, 0.8, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05,
            ],
        );
        assert_eq!(accuracy(&y_pred, &y_true), 0.0);
    }

    // 2 of 4 correct -> 0.5
    #[test]
    fn accuracy_partial() {
        let y_true = Matrix::from_vec(
            3,
            4,
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        );
        let y_pred = Matrix::from_vec(
            3,
            4,
            vec![
                0.1, 0.7, 0.9, 0.9, 0.8, 0.2, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05,
            ],
        );
        assert_eq!(accuracy(&y_pred, &y_true), 0.5);
    }

    // Single sample
    #[test]
    fn accuracy_single_sample() {
        let y_true = Matrix::from_vec(3, 1, vec![0.0, 1.0, 0.0]);
        let y_pred = Matrix::from_vec(3, 1, vec![0.2, 0.7, 0.1]);
        assert_eq!(accuracy(&y_pred, &y_true), 1.0);
    }
}
