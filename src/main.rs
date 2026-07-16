use crate::matrix::Matrix;

mod matrix;

fn main() {
    println!("Hello, world!");
    let m1 = Matrix::from_array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]);
    let m2 = Matrix::ones(2, 4);
    println!("{:?}", (&m1 + &m2).as_slice());
    println!("{:?}", m2.as_slice());
}
