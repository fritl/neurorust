use crate::{eval::accuracy, matrix::Matrix, nn::loss::SoftmaxCrossEntropy};
use wasm_bindgen::prelude::*;
mod eval;
mod matrix;
mod mnist_parser;
mod nn;
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
// Import the `window.alert` function from the Web.
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

// Export a `greet` function from Rust to JavaScript, that alerts a
// hello message.
#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}

#[wasm_bindgen]
pub fn train_net() -> f32 {
    web_sys::console::log_1(&"Hello World".into());
    let train_images_file = include_bytes!("../data/train-images-idx3-ubyte.gz");
    let train_labels_file = include_bytes!("../data/train-labels-idx1-ubyte.gz");

    let train_images = mnist_parser::MnistData::from_bytes(train_images_file)
        .expect("Failed to parse training images");
    let train_labels = mnist_parser::MnistData::from_bytes(train_labels_file)
        .expect("Failed to parse training labels");
    let train_images_matrix = Matrix::from_vec(
        train_labels.sizes[0] as usize,
        (train_images.sizes[1] * train_images.sizes[2]) as usize,
        train_images
            .data
            .iter()
            .map(|&x| x as f32 / 255.0)
            .collect(),
    )
    .transpose();
    let train_labels_matrix = Matrix::from_vec(
        train_labels.sizes[0] as usize,
        10,
        train_labels
            .data
            .iter()
            .flat_map(|&x| {
                let mut values = vec![0.0; 10];
                values[x as usize] = 1.0;
                values
            })
            .collect(),
    )
    .transpose();

    let test_images_file = include_bytes!("../data/t10k-images-idx3-ubyte.gz");
    let test_labels_file = include_bytes!("../data/t10k-labels-idx1-ubyte.gz");
    let test_images =
        mnist_parser::MnistData::from_bytes(test_images_file).expect("Failed to parse test images");
    let test_labels =
        mnist_parser::MnistData::from_bytes(test_labels_file).expect("Failed to parse test labels");
    let test_images_matrix = Matrix::from_vec(
        test_labels.sizes[0] as usize,
        (test_images.sizes[1] * test_images.sizes[2]) as usize,
        test_images.data.iter().map(|&x| x as f32 / 255.0).collect(),
    )
    .transpose();
    let test_labels_matrix = Matrix::from_vec(
        test_labels.sizes[0] as usize,
        10,
        test_labels
            .data
            .iter()
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
    web_sys::console::log_1(&"Start training".into());
    network.train(&train_images_matrix, &train_labels_matrix, 2, 128);
    let pred = network.predict(&train_images_matrix);
    let train_accuracy = accuracy(&pred, &train_labels_matrix);
    println!("Training accuracy: {train_accuracy}");
    let pred = network.predict(&test_images_matrix);
    let test_accuracy = accuracy(&pred, &test_labels_matrix);
    println!("Test accuracy: {test_accuracy}");
    test_accuracy
}
#[wasm_bindgen]
pub struct WasmNet {
    network: nn::network::Network<SoftmaxCrossEntropy>,
}

#[wasm_bindgen]
impl WasmNet {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmNet {
        let network = nn::network::Network::from_vec(
            &vec![784, 100, 10],
            nn::activations::ReLU,
            Some(1221),
            nn::loss::SoftmaxCrossEntropy,
            0.1,
        );
        WasmNet { network }
    }

    pub fn train(&mut self) -> f32 {
        web_sys::console::log_1(&"Hello World".into());
        let train_images_file = include_bytes!("../data/train-images-idx3-ubyte.gz");
        let train_labels_file = include_bytes!("../data/train-labels-idx1-ubyte.gz");

        let train_images = mnist_parser::MnistData::from_bytes(train_images_file)
            .expect("Failed to parse training images");
        let train_labels = mnist_parser::MnistData::from_bytes(train_labels_file)
            .expect("Failed to parse training labels");
        let train_images_matrix = Matrix::from_vec(
            train_labels.sizes[0] as usize,
            (train_images.sizes[1] * train_images.sizes[2]) as usize,
            train_images
                .data
                .iter()
                .map(|&x| x as f32 / 255.0)
                .collect(),
        )
        .transpose();
        let train_labels_matrix = Matrix::from_vec(
            train_labels.sizes[0] as usize,
            10,
            train_labels
                .data
                .iter()
                .flat_map(|&x| {
                    let mut values = vec![0.0; 10];
                    values[x as usize] = 1.0;
                    values
                })
                .collect(),
        )
        .transpose();

        let test_images_file = include_bytes!("../data/t10k-images-idx3-ubyte.gz");
        let test_labels_file = include_bytes!("../data/t10k-labels-idx1-ubyte.gz");
        let test_images = mnist_parser::MnistData::from_bytes(test_images_file)
            .expect("Failed to parse test images");
        let test_labels = mnist_parser::MnistData::from_bytes(test_labels_file)
            .expect("Failed to parse test labels");
        let test_images_matrix = Matrix::from_vec(
            test_labels.sizes[0] as usize,
            (test_images.sizes[1] * test_images.sizes[2]) as usize,
            test_images.data.iter().map(|&x| x as f32 / 255.0).collect(),
        )
        .transpose();
        let test_labels_matrix = Matrix::from_vec(
            test_labels.sizes[0] as usize,
            10,
            test_labels
                .data
                .iter()
                .flat_map(|&x| {
                    let mut values = vec![0.0; 10];
                    values[x as usize] = 1.0;
                    values
                })
                .collect(),
        )
        .transpose();

        web_sys::console::log_1(&"Start training".into());
        self.network
            .train(&train_images_matrix, &train_labels_matrix, 10, 128);

        let pred = self.network.predict(&test_images_matrix);
        accuracy(&pred, &test_labels_matrix)
    }

    pub fn predict(&mut self, input: Vec<f32>) -> Vec<f32> {
        let input_matrix = Matrix::from_vec(784, 1, input);
        let pred = self.network.predict(&input_matrix);
        pred.as_slice().to_vec()
    }
}
