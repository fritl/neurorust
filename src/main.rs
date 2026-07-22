use crate::{matrix::Matrix, nn::network::Network};
use std::fs::File;

mod matrix;
mod mnist_parser;
mod nn;

fn main() {
    let train_images_file =
        File::open("./data/train-images-idx3-ubyte.gz").expect("Failed to open train images file");
    let train_labels_file =
        File::open("./data/train-labels-idx1-ubyte.gz").expect("Failed to open train labels file");
    let train_images =
        mnist_parser::MnistData::new(&train_images_file).expect("Failed to parse training images");
    let train_labels =
        mnist_parser::MnistData::new(&train_labels_file).expect("Failed to parse training labels");
    let num_samples = 10000;
    let train_images_matrix = Matrix::from_vec(
        num_samples as usize,
        (train_images.sizes[1] * train_images.sizes[2]) as usize,
        train_images
            .data
            .iter()
            .take(num_samples * 784)
            .map(|&x| x as f32 / 255.0)
            .collect(),
    )
    .transpose();
    let train_labels_matrix = Matrix::from_vec(
        num_samples as usize,
        10,
        train_labels
            .data
            .iter()
            .take(num_samples)
            .flat_map(|&x| {
                let mut values = vec![0.0; 10];
                values[x as usize] = 1.0;
                values
            })
            .collect(),
    )
    .transpose();

    let mut network = nn::network::Network::from_vec(
        &vec![784, 100, 10],
        nn::activations::ReLU,
        Some(1221),
        nn::loss::SoftmaxCrossEntropy,
        0.1,
    );
    println!("Begin Training");
    network.train(&train_images_matrix, &train_labels_matrix, 10000);
}

fn train_xor() -> Network<nn::loss::MSE> {
    let mut network = nn::network::Network::from_vec(
        &vec![2, 2, 1],
        nn::activations::Sigmoid,
        Some(1221),
        nn::loss::MSE,
        1.5,
    );
    let xor_data = matrix::Matrix::from_array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
    let xor_result = matrix::Matrix::from_array([[0.0, 1.0, 1.0, 0.0]]);

    network.train(&xor_data, &xor_result, 10000);
    network
}
