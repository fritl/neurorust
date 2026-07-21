mod matrix;
mod nn;

fn main() {
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
}
