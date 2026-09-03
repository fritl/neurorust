use std::{fs::File, rc::Rc};

use wgpu::wgt::CommandEncoderDescriptor;

use crate::gpu::{matrix::GpuMatrix, network::Network, state::GpuState};

mod eval;
mod gpu;
mod matrix;
mod mnist_parser;

/// Returns: (train_X, train_y, test_X, test_y)
fn load_data(gpu_state: Rc<GpuState>) -> (GpuMatrix, GpuMatrix, GpuMatrix, GpuMatrix) {
    let train_images_file =
        File::open("./data/train-images-idx3-ubyte.gz").expect("Failed to open train images file");
    let train_labels_file =
        File::open("./data/train-labels-idx1-ubyte.gz").expect("Failed to open train labels file");

    let train_images =
        mnist_parser::MnistData::new(&train_images_file).expect("Failed to parse training images");
    let train_labels =
        mnist_parser::MnistData::new(&train_labels_file).expect("Failed to parse training labels");

    let train_images_matrix = GpuMatrix::new(
        train_images.sizes[0] as usize,
        (train_images.sizes[1] * train_images.sizes[2]) as usize,
        &train_images
            .data
            .iter()
            .map(|&x| x as f32 / 255.0)
            .collect(),
        Rc::clone(&gpu_state),
    );
    let train_images_matrix_transposed = GpuMatrix::empty(
        train_images_matrix.columns(),
        train_images_matrix.rows(),
        Rc::clone(&gpu_state),
    );
    let mut encoder =
        gpu_state
            .gpu_context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("command_encoder_load_data"),
            });
    train_images_matrix.transpose(&train_images_matrix_transposed, &mut encoder);

    let train_labels_matrix = GpuMatrix::new(
        train_labels.sizes[0] as usize,
        10,
        &train_labels
            .data
            .iter()
            .flat_map(|&x| {
                let mut values = vec![0.0; 10];
                values[x as usize] = 1.0;
                values
            })
            .collect(),
        Rc::clone(&gpu_state),
    );
    let train_labels_matrix_transposed = GpuMatrix::empty(
        train_labels_matrix.columns(),
        train_labels_matrix.rows(),
        Rc::clone(&gpu_state),
    );
    train_labels_matrix.transpose(&train_labels_matrix_transposed, &mut encoder);

    let test_images_file =
        File::open("./data/t10k-images-idx3-ubyte.gz").expect("Failed to open test images file");
    let test_labels_file =
        File::open("./data/t10k-labels-idx1-ubyte.gz").expect("Failed to open test labels file");
    let test_images =
        mnist_parser::MnistData::new(&test_images_file).expect("Failed to parse test images");
    let test_labels =
        mnist_parser::MnistData::new(&test_labels_file).expect("Failed to parse test labels");

    let test_images_matrix = GpuMatrix::new(
        test_labels.sizes[0] as usize,
        (test_images.sizes[1] * test_images.sizes[2]) as usize,
        &test_images.data.iter().map(|&x| x as f32 / 255.0).collect(),
        Rc::clone(&gpu_state),
    );
    let test_images_matrix_transposed = GpuMatrix::empty(
        test_images_matrix.columns(),
        test_images_matrix.rows(),
        Rc::clone(&gpu_state),
    );
    test_images_matrix.transpose(&test_images_matrix_transposed, &mut encoder);

    let test_labels_matrix = GpuMatrix::new(
        test_labels.sizes[0] as usize,
        10,
        &test_labels
            .data
            .iter()
            .flat_map(|&x| {
                let mut values = vec![0.0; 10];
                values[x as usize] = 1.0;
                values
            })
            .collect(),
        Rc::clone(&gpu_state),
    );
    let test_labels_matrix_transposed = GpuMatrix::empty(
        test_labels_matrix.columns(),
        test_labels_matrix.rows(),
        Rc::clone(&gpu_state),
    );
    test_labels_matrix.transpose(&test_labels_matrix_transposed, &mut encoder);
    let command_buffer = encoder.finish();
    gpu_state.gpu_context.queue.submit([command_buffer]);
    (
        train_images_matrix_transposed,
        train_labels_matrix_transposed,
        test_images_matrix_transposed,
        test_labels_matrix_transposed,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use std::time::Instant;

    use tokio::runtime;

    use crate::eval::accuracy;

    let rt = runtime::Runtime::new().unwrap();
    let gpu_state = Rc::new(rt.block_on(GpuState::default()));

    let mut network =
        Network::from_vec(&vec![784, 100, 10], Some(1221), 0.1, Rc::clone(&gpu_state));

    let (train_x, train_y, test_x, test_y) = load_data(Rc::clone(&gpu_state));
    println!("Begin Training");
    let start = Instant::now();
    network.train(&train_x, &train_y, 100, 128);
    let time = start.elapsed();
    println!("Training time: {time:?}");
    let pred = network.predict(&train_x);
    let pred_cpu = matrix::Matrix::from_vec(
        pred.rows(),
        pred.columns(),
        rt.block_on(pred.to_cpu()).unwrap(),
    );
    let true_cpu = matrix::Matrix::from_vec(
        train_y.rows(),
        train_y.columns(),
        rt.block_on(train_y.to_cpu()).unwrap(),
    );
    let train_accuracy = accuracy(&pred_cpu, &true_cpu);
    println!("Training accuracy: {train_accuracy}");
    let pred = network.predict(&test_x);
    let pred_cpu = matrix::Matrix::from_vec(
        pred.rows(),
        pred.columns(),
        rt.block_on(pred.to_cpu()).unwrap(),
    );
    let true_cpu = matrix::Matrix::from_vec(
        test_y.rows(),
        test_y.columns(),
        rt.block_on(test_y.to_cpu()).unwrap(),
    );
    let test_accuracy = accuracy(&pred_cpu, &true_cpu);
    println!("Test accuracy: {test_accuracy}");
}
