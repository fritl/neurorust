use std::rc::Rc;

use rand::SeedableRng;

use crate::gpu::{
    buffer_pool::BufferPool, layer::Layer, loss::SoftmaxCrossEntropy, matrix::GpuMatrix,
    state::GpuState,
};

pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f32,
    buffer_pool: BufferPool,
    gpu_state: Rc<GpuState>,
}

impl Network {
    pub fn from_vec(
        arch: &[usize],
        seed: Option<u64>,
        learning_rate: f32,
        gpu_state: Rc<GpuState>,
    ) -> Self {
        if arch.len() < 2 {
            panic!("Network requires at least 2 layers (input and output)");
        }
        let mut layers = Vec::with_capacity(arch.len());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(3001));
        for i in 0..arch.len() - 2 {
            layers.push(Layer::with_uniform_xavier_weights(
                arch[i],
                arch[i + 1],
                &mut rng,
                false,
                Rc::clone(&gpu_state),
            ));
        }
        layers.push(Layer::with_uniform_xavier_weights(
            arch[arch.len() - 2],
            arch[arch.len() - 1],
            &mut rng,
            true,
            Rc::clone(&gpu_state),
        ));
        Network {
            layers,
            learning_rate,
            buffer_pool: BufferPool::new(Rc::clone(&gpu_state)),
            gpu_state,
        }
    }

    pub fn fowrard(&mut self, x: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) -> GpuMatrix {
        let mut current_input = x;
        for l in self.layers.iter_mut() {
            l.forward(current_input, encoder);
            current_input = l.cached_a.as_ref().unwrap();
        }
        current_input.clone()
    }

    pub fn backward(&mut self, x: &GpuMatrix, y: &GpuMatrix, encoder: &mut wgpu::CommandEncoder) {
        // x = Logits , y = one-hot Targets
        let initial_delta = self.buffer_pool.get(x.rows(), x.columns());
        SoftmaxCrossEntropy::backward(x, y, &initial_delta, &mut self.buffer_pool, encoder);

        let mut current_delta = initial_delta;
        for l in self.layers.iter_mut().rev() {
            let input_shape = l.cached_input_shape();
            let next_delta = self.buffer_pool.get(input_shape.0, input_shape.1);

            l.backward(
                &current_delta,
                self.learning_rate,
                &next_delta,
                &mut self.buffer_pool,
                encoder,
            );

            self.buffer_pool.recycle(current_delta);
            current_delta = next_delta;
        }
        self.buffer_pool.recycle(current_delta);
    }

    pub fn train(&mut self, x: &GpuMatrix, y: &GpuMatrix, epochs: u32, batch_size: usize) {
        assert_eq!(x.columns(), y.columns());

        let num_batches = x.columns().div_ceil(batch_size);
        let device = &Rc::clone(&self.gpu_state).gpu_context.device;

        for i in 0..epochs {
            println!("Epoch: {i} / {epochs}");
            for j in 0..num_batches {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("command_encoder_epoch_{i}_batch{j}")),
                });
                let start_col = j * batch_size;
                let end_col = ((j + 1) * batch_size).min(x.columns());
                let cur_batch_size = end_col - start_col;

                let scratch_x = self.buffer_pool.get(x.rows(), cur_batch_size);
                let scratch_y = self.buffer_pool.get(y.rows(), cur_batch_size);

                x.column_slice(
                    &scratch_x,
                    start_col as u32,
                    cur_batch_size as u32,
                    &mut encoder,
                );

                y.column_slice(
                    &scratch_y,
                    start_col as u32,
                    cur_batch_size as u32,
                    &mut encoder,
                );

                let output = self.fowrard(&scratch_x, &mut encoder);
                self.backward(&output, &scratch_y, &mut encoder);

                self.buffer_pool.recycle(scratch_y);
                self.buffer_pool.recycle(scratch_x);
                let command_buffer = encoder.finish();
                self.gpu_state.gpu_context.queue.submit([command_buffer]);
            }
        }
    }

    pub fn train_one(&mut self, x: &GpuMatrix, y: &GpuMatrix, batch_size: usize) {
        assert_eq!(x.columns(), y.columns());

        let num_batches = x.columns().div_ceil(batch_size);
        let device = &Rc::clone(&self.gpu_state).gpu_context.device;

        for i in 0..num_batches {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("command_encoder__batch{i}")),
            });
            let start_col = i * batch_size;
            let end_col = ((i + 1) * batch_size).min(x.columns());
            let cur_batch_size = end_col - start_col;

            let scratch_x = self.buffer_pool.get(x.rows(), cur_batch_size);
            let scratch_y = self.buffer_pool.get(y.rows(), cur_batch_size);

            x.column_slice(
                &scratch_x,
                start_col as u32,
                cur_batch_size as u32,
                &mut encoder,
            );

            y.column_slice(
                &scratch_y,
                start_col as u32,
                cur_batch_size as u32,
                &mut encoder,
            );

            let output = self.fowrard(&scratch_x, &mut encoder);
            self.backward(&output, &scratch_y, &mut encoder);

            self.buffer_pool.recycle(scratch_y);
            self.buffer_pool.recycle(scratch_x);
            let command_buffer = encoder.finish();
            self.gpu_state.gpu_context.queue.submit([command_buffer]);
        }
    }

    pub fn predict(&mut self, x: &GpuMatrix) -> GpuMatrix {
        let mut encoder = self.gpu_state.gpu_context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("command_encoder_predict"),
            },
        );
        let logits = self.fowrard(x, &mut encoder);
        let output = self.buffer_pool.get(logits.rows(), logits.columns());
        SoftmaxCrossEntropy::forward(&logits, &output, &mut self.buffer_pool, &mut encoder);
        let command_buffer = encoder.finish();
        self.gpu_state.gpu_context.queue.submit([command_buffer]);
        output
    }
}
