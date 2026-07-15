use crate::matrix::Matrix;

mod matrix;

fn main() {
    println!("Hello, world!");
    let new_matrix = Matrix::from_array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]);
    println!("{:?}", new_matrix.as_slice());
}
