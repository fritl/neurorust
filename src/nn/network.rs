use crate::{
    matrix::Matrix,
    nn::{
        activations,
        layer::{self, Layer},
        loss,
    },
};
use rand::SeedableRng;
use std::time::Instant;

pub struct Network<L: loss::Loss> {
    layers: Vec<layer::Layer>,
    loss_function: L,
    learning_rate: f32,
}

impl<L: loss::Loss> Network<L> {
    pub fn from_vec<A>(
        arch: &[usize],
        activation: A,
        seed: Option<u64>,
        loss_function: L,
        learning_rate: f32,
    ) -> Self
    where
        A: activations::Activation + Clone + 'static,
    {
        if arch.len() < 2 {
            panic!("Network requires at least 2 layers (input and output)");
        }
        let mut layers = Vec::with_capacity(arch.len());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(3001));
        for i in 0..arch.len() - 1 {
            layers.push(Layer::with_uniform_xavier_weights(
                arch[i],
                arch[i + 1],
                Box::new(activation.clone()),
                &mut rng,
            ))
        }
        Network {
            layers,
            loss_function,
            learning_rate,
        }
    }

    fn forward(&mut self, x: &Matrix, y: &Matrix) -> (Matrix, f32) {
        assert_eq!(x.columns(), y.columns());
        let mut x = x.clone();
        for l in &mut self.layers {
            x = l.forward(&x);
        }
        let loss = self.loss_function.forward(&x, y);
        (x, loss)
    }

    fn backward(&mut self, x: &Matrix, y: &Matrix) {
        let mut delta = self.loss_function.backward(x, y);
        for l in self.layers.iter_mut().rev() {
            delta = l.backward(&delta, self.learning_rate);
        }
    }

    pub fn train(&mut self, x: &Matrix, y: &Matrix, epochs: u32, batch_size: usize) {
        assert_eq!(x.columns(), y.columns());
        let start = Instant::now();
        let num_batches = x.columns().div_ceil(batch_size);
        for i in 0..epochs {
            let mut loss_sum = 0.0;
            let epoch_start = Instant::now();
            //TODO: Shuffle input
            for j in 0..num_batches {
                let start_col = j * batch_size;
                let end_col = ((j + 1) * batch_size).min(x.columns());
                let cur_batch_size = end_col - start_col;

                let mut batch_data = Vec::with_capacity(x.rows() * cur_batch_size);
                for r in 0..x.rows() {
                    let start_idx = r * x.columns() + start_col;
                    batch_data
                        .extend_from_slice(&x.as_slice()[start_idx..start_idx + cur_batch_size]);
                }

                let mut label_data = Vec::with_capacity(y.rows() * cur_batch_size);
                for r in 0..y.rows() {
                    let start_idx = r * y.columns() + start_col;
                    label_data
                        .extend_from_slice(&y.as_slice()[start_idx..start_idx + cur_batch_size]);
                }
                let batch = Matrix::from_vec(x.rows(), cur_batch_size, batch_data);
                let labels = Matrix::from_vec(y.rows(), cur_batch_size, label_data);
                let (pred, loss) = self.forward(&batch, &labels);
                loss_sum += loss;
                self.backward(&pred, &labels);
            }
            let epoch_duration = epoch_start.elapsed();
            let avg_epoch_duration = start.elapsed() / (i + 1);
            println!(
                "Epoch {i} / {epochs} Average Loss: {} Time: {epoch_duration:?} eta: {:?}",
                loss_sum / num_batches as f32,
                avg_epoch_duration * (epochs - i)
            );
        }
    }
    pub fn predict(&mut self, x: &Matrix) -> Matrix {
        let mut x = x.clone();
        for l in &mut self.layers {
            x = l.forward(&x);
        }
        x
    }
}
