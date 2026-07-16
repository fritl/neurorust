use std::ops;

pub struct Matrix {
    rows: usize,
    columns: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Matrix {
            rows,
            columns,
            data: vec![0 as f32; rows * columns],
        }
    }

    pub fn ones(rows: usize, columns: usize) -> Self {
        Matrix {
            rows,
            columns,
            data: vec![1 as f32; rows * columns],
        }
    }

    pub fn from_vec(rows: usize, columns: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * columns, "Insufficient data for vector");
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn from_array<const N: usize, const M: usize>(data: [[f32; N]; M]) -> Self {
        Matrix {
            rows: M,
            columns: N,
            data: Vec::from(data.into_iter().flatten().collect::<Vec<_>>()),
        }
    }

    pub fn get(&self, row: usize, column: usize) -> f32 {
        assert!(row < self.rows, "Row out of bounds");
        assert!(column < self.columns, "Column out of bounds");
        self.data[row * self.columns + column]
    }

    pub fn set(&mut self, row: usize, column: usize, value: f32) {
        assert!(row < self.rows, "Row out of bounds");
        assert!(column < self.columns, "Column out of bounds");
        self.data[row * self.columns + column] = value
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut new_mat = Matrix::zeros(self.columns, self.rows);
        for i in 0..self.rows {
            for j in 0..self.columns {
                new_mat.set(j, i, self.get(i, j));
            }
        }
        new_mat
    }

    pub fn optimized_matmul(&self, other: &Matrix) -> Matrix {
        let other = other.transpose();
        assert_eq!(self.columns, other.rows, "Shape mismatch");
        let mut new_mat = Matrix::zeros(self.rows, other.columns);
        for i in 0..self.rows {
            for j in 0..other.rows {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.data[i * self.columns + k] * other.data[j * other.columns + k];
                }
                new_mat.set(i, j, sum);
                new_mat.data[i * new_mat.columns + j] = sum;
            }
        }
        new_mat
    }
}

impl ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        let mut new_vec = Vec::new();
        for i in 0..self.data.len() {
            new_vec.push(self.data[i] + other.data[i]);
        }
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: new_vec,
        }
    }
}

impl ops::Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        let mut new_vec = Vec::new();
        for i in 0..self.data.len() {
            new_vec.push(self.data[i] - other.data[i]);
        }
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: new_vec,
        }
    }
}

impl ops::Mul<f32> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: f32) -> Matrix {
        let new_vec = self
            .as_slice()
            .into_iter()
            .map(|v| v * other)
            .collect::<Vec<f32>>();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: new_vec,
        }
    }
}

impl ops::Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Matrix {
        assert_eq!(self.columns, other.rows, "Shape mismatch");
        let mut new_mat = Matrix::zeros(self.rows, other.columns);
        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.get(i, k) * other.get(k, j);
                }
                new_mat.set(i, j, sum);
            }
        }
        new_mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(2, 3);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let m = Matrix::ones(2, 2);
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_from_vec() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    #[should_panic(expected = "Insufficient data for vector")]
    fn test_from_vec_panics_on_wrong_length() {
        Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_array() {
        let m = Matrix::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), 5.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    #[should_panic(expected = "Row out of bounds")]
    fn test_get_row_out_of_bounds_panics() {
        let m = Matrix::zeros(2, 2);
        m.get(2, 0);
    }

    #[test]
    #[should_panic(expected = "Column out of bounds")]
    fn test_get_column_out_of_bounds_panics() {
        let m = Matrix::zeros(2, 2);
        m.get(0, 2);
    }

    #[test]
    #[should_panic(expected = "Row out of bounds")]
    fn test_set_row_out_of_bounds_panics() {
        let mut m = Matrix::zeros(2, 2);
        m.set(2, 0, 1.0);
    }

    #[test]
    fn test_shape() {
        let m = Matrix::zeros(3, 5);
        assert_eq!(m.shape(), (3, 5));
    }

    #[test]
    fn test_as_slice() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = &a + &b;
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_add_shape_mismatch_panics() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(3, 3);
        let _ = &a + &b;
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = &a - &b;
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.as_slice(), &[4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn test_sub_shape_mismatch_panics() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(3, 3);
        let _ = &a - &b;
    }

    #[test]
    fn test_matmul() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = &a * &b;
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.get(0, 0), 1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0);
        assert_eq!(result.get(0, 1), 1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0);
        assert_eq!(result.get(1, 0), 4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0);
        assert_eq!(result.get(1, 1), 4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Matrix::from_vec(2, 2, vec![3.0, 5.0, 7.0, 9.0]);
        let identity = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let result = &a * &identity;
        assert_eq!(result.as_slice(), a.as_slice());
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_matmul_dimension_mismatch_panics() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 2);
        let _ = &a * &b;
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_transpose_twice_is_identity() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = m.transpose().transpose();
        assert_eq!(tt.shape(), m.shape());
        assert_eq!(tt.as_slice(), m.as_slice());
    }

    #[test]
    fn test_hadamard() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
        let result = a.hadamard(&b);
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    #[should_panic]
    fn test_hadamard_shape_mismatch_panics() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(3, 3);
        let _ = a.hadamard(&b);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = &a * 2.0;
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_mul_by_zero() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = &a * 0.0;
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scalar_mul_by_negative() {
        let a = Matrix::from_vec(2, 2, vec![1.0, -2.0, 3.0, -4.0]);
        let result = &a * -1.0;
        assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0, 4.0]);
    }
}
