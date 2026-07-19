use crate::matrix::Matrix;
use crate::nn::activations;
use std::hint;
use std::time::Instant;

mod matrix;
mod nn;

fn main() {
    let shape = (1024, 1024);
    let mut mat_a = Matrix::zeros(shape.0, shape.1);
    let shape = (1024, 1024);
    let mut mat_b = Matrix::zeros(shape.0, shape.1);

    for i in 0..mat_a.shape().0 {
        for j in 0..mat_a.shape().1 {
            mat_a.set(i, j, rand::random());
        }
    }
    for i in 0..mat_b.shape().0 {
        for j in 0..mat_b.shape().1 {
            mat_b.set(i, j, rand::random());
        }
    }

    println!("Start standard");
    let start_standard = Instant::now();
    let _res_standard = &mat_a * &mat_b;
    // Removing this will optimze the calculation away when building with optimization
    hint::black_box(_res_standard);
    let time = start_standard.elapsed();
    println!("Standard time: {time:?}");
}
