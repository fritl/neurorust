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

    pub fn from_vec<T>(rows: usize, columns: usize, data: Vec<f32>) -> Self {
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn from_array<const N: usize, const M: usize>(data: [[f32; N]; M]) -> Self {
        Matrix {
            rows: N,
            columns: M,
            data: Vec::from(data.into_iter().flatten().collect::<Vec<_>>()),
        }
    }

    pub fn get(&self, row: usize, column: usize) -> f32 {
        self.data[row * self.columns + column]
    }

    pub fn set(&mut self, row: usize, column: usize, value: f32) {
        self.data[row * self.columns + column] = value
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
}
