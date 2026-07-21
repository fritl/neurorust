use crate::{
    matrix::Matrix,
    nn::{
        activations,
        layer::{self, Layer},
        loss,
    },
};
use rand::SeedableRng;

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

    pub fn train(&mut self, x: &Matrix, y: &Matrix, epochs: u32) {
        for i in 0..epochs {
            let (pred, loss) = self.forward(x, y);
            println!("Epoch {i} / {epochs} Loss: {loss}");
            self.backward(&pred, y);
        }
    }
}
