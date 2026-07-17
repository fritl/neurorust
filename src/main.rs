use crate::matrix::Matrix;
use std::hint;
use std::time::Instant;

mod matrix;

fn main() {
    let shape = (1024, 512);
    let mut mat_a = Matrix::zeros(shape.0, shape.1);
    let shape = (512, 4096);
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

    println!("Start optimized");
    let start_optimized = Instant::now();
    let _res_optimized = &mat_a.optimized_matmul(&mat_b);
    hint::black_box(_res_optimized);
    let time = start_optimized.elapsed();
    println!("Optimized time: {time:?}");
}
