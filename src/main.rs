use crate::matrix::Matrix;
use std::time::Instant;

mod matrix;

fn main() {
    let shape = (250, 250);
    let mut mat_a = Matrix::zeros(shape.0, shape.1);
    let mut mat_b = Matrix::zeros(shape.0, shape.1);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            mat_a.set(i, j, rand::random());
            mat_b.set(i, j, rand::random());
        }
    }

    println!("Start standard");
    let start_standard = Instant::now();
    let _res_standard = &mat_a * &mat_b;
    let time = start_standard.elapsed();
    println!("Standard time: {time:?}");

    println!("Start optimized");
    let start_optimized = Instant::now();
    let _res_standard = &mat_a.optimized_matmul(&mat_b);
    let time = start_optimized.elapsed();
    println!("Optimized time: {time:?}");
}
