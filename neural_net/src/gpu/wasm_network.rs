use std::cell::Cell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys;
use wgpu::wgt::CommandEncoderDescriptor;

use crate::eval::accuracy;
use crate::gpu::{matrix::GpuMatrix, network::Network, state::GpuState};
use crate::matrix;
use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;
use std::io::Read;

#[derive(Debug)]
pub struct MnistData {
    pub sizes: Vec<i32>,
    pub data: Vec<u8>,
}

impl MnistData {
    pub fn from_bytes(bytes: &[u8]) -> Result<MnistData, std::io::Error> {
        let mut r = Cursor::new(bytes);
        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        let magic_number = r.read_i32::<BigEndian>()?;

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!("Invalid MNIST magic number: {magic_number}. Expected 2049 or 2051"),
        };

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

#[wasm_bindgen]
pub struct WasmNetwork {
    network: Network,
    train_x: GpuMatrix,
    train_y: GpuMatrix,
    test_x: GpuMatrix,
    test_y: GpuMatrix,
    training: Rc<Cell<bool>>,
    input_neurons: usize,
    gpu_state: Rc<GpuState>,
}

#[wasm_bindgen]
impl WasmNetwork {
    #[wasm_bindgen]
    pub async fn network(
        arch: &[usize],
        learning_rate: f32,
        seed: Option<u64>,
        train_images: &[u8],
        train_labels: &[u8],
        test_images: &[u8],
        test_labels: &[u8],
    ) -> Result<WasmNetwork, JsError> {
        if arch.len() < 2 {
            return Err(JsError::new(
                "Network requires at leas 2 layer (input and output)",
            ));
        }

        let train_x_raw =
            MnistData::from_bytes(train_images).expect("Failed to load training images");
        let train_y_raw =
            MnistData::from_bytes(train_labels).expect("Failed to load training labels");

        let gpu_state = Rc::new(GpuState::default().await);

        let train_x = GpuMatrix::new(
            train_x_raw.sizes[0] as usize,
            (train_x_raw.sizes[1] * train_x_raw.sizes[2]) as usize,
            &train_x_raw.data.iter().map(|&x| x as f32 / 255.0).collect(),
            Rc::clone(&gpu_state),
        );
        let train_x = Self::transposed(&train_x, Rc::clone(&gpu_state));

        let train_y = GpuMatrix::new(
            train_y_raw.sizes[0] as usize,
            10,
            &train_y_raw
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
        let train_y = Self::transposed(&train_y, Rc::clone(&gpu_state));

        if train_x.columns() != train_y.columns() {
            return Err(JsError::new(
                "Train and test data must have the same amount of columns",
            ));
        }
        if train_x.rows() != arch[0] {
            return Err(JsError::new(&format!(
                "First layer must match train matrix rows ({})",
                train_x.rows()
            )));
        }
        if train_y.rows() != arch[arch.len() - 1] {
            return Err(JsError::new(&format!(
                "Last layer must match train matrix rows ({})",
                train_y.rows()
            )));
        }

        let test_x_raw = MnistData::from_bytes(test_images).expect("Failed to load testing images");
        let test_y_raw = MnistData::from_bytes(test_labels).expect("Failed to load testing labels");

        let test_x = GpuMatrix::new(
            test_x_raw.sizes[0] as usize,
            (test_x_raw.sizes[1] * test_x_raw.sizes[2]) as usize,
            &test_x_raw.data.iter().map(|&x| x as f32 / 255.0).collect(),
            Rc::clone(&gpu_state),
        );
        let test_x = Self::transposed(&test_x, Rc::clone(&gpu_state));

        let test_y = GpuMatrix::new(
            test_y_raw.sizes[0] as usize,
            10,
            &test_y_raw
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
        let test_y = Self::transposed(&test_y, Rc::clone(&gpu_state));

        if test_x.columns() != test_y.columns() {
            return Err(JsError::new(
                "Train and test data must have the same amount of columns",
            ));
        }
        if test_x.rows() != arch[0] {
            return Err(JsError::new(&format!(
                "First layer must match test matrix rows ({})",
                test_x.rows()
            )));
        }
        if test_y.rows() != arch[arch.len() - 1] {
            return Err(JsError::new(&format!(
                "Last layer must match test matrix rows ({})",
                test_y.rows()
            )));
        }

        let network = Network::from_vec(arch, seed, learning_rate, Rc::clone(&gpu_state));

        Ok(WasmNetwork {
            network,
            input_neurons: arch[0],
            gpu_state: Rc::clone(&gpu_state),
            training: Rc::new(Cell::new(false)),
            train_x,
            train_y,
            test_x,
            test_y,
        })
    }

    fn transposed(input: &GpuMatrix, gpu_state: Rc<GpuState>) -> GpuMatrix {
        let result = GpuMatrix::empty(input.columns(), input.rows(), Rc::clone(&gpu_state));
        let device = &gpu_state.gpu_context.device;
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("command_encoder_transpose"),
        });

        input.transpose(&result, &mut encoder);

        let command_buffer = encoder.finish();
        gpu_state.gpu_context.queue.submit([command_buffer]);

        result
    }

    #[wasm_bindgen]
    pub async fn train(
        &mut self,
        epochs: u32,
        batch_size: usize,
        on_progress: js_sys::Function,
    ) -> Vec<f32> {
        self.training.set(true);
        let _ = on_progress.call1(&JsValue::NULL, &JsValue::from(0));
        for i in 0..epochs {
            if !self.training.get() {
                break;
            }
            self.network
                .train_one(&self.train_x, &self.train_y, batch_size);

            let (tx, rx) = tokio::sync::oneshot::channel();
            self.gpu_state
                .gpu_context
                .queue
                .on_submitted_work_done(move || {
                    let _ = tx.send(());
                });
            let _ = rx.await;
            let _ = on_progress.call1(&JsValue::NULL, &JsValue::from(i + 1));
        }
        self.training.set(false);
        let pred = self.network.predict(&self.train_x);
        let pred_cpu =
            matrix::Matrix::from_vec(pred.rows(), pred.columns(), pred.to_cpu().await.unwrap());
        let true_cpu = matrix::Matrix::from_vec(
            self.train_y.rows(),
            self.train_y.columns(),
            self.train_y.to_cpu().await.unwrap(),
        );
        let train_accuracy = accuracy(&pred_cpu, &true_cpu);
        let pred = self.network.predict(&self.test_x);
        let pred_cpu =
            matrix::Matrix::from_vec(pred.rows(), pred.columns(), pred.to_cpu().await.unwrap());
        let true_cpu = matrix::Matrix::from_vec(
            self.test_y.rows(),
            self.test_y.columns(),
            self.test_y.to_cpu().await.unwrap(),
        );
        let test_accuracy = accuracy(&pred_cpu, &true_cpu);
        vec![train_accuracy, test_accuracy]
    }

    #[wasm_bindgen]
    pub async fn predict(&mut self, x: &[f32]) -> Vec<f32> {
        let x = GpuMatrix::new(
            self.input_neurons,
            1,
            &x.to_vec(),
            Rc::clone(&self.gpu_state),
        );
        self.network.predict(&x).to_cpu().await.unwrap()
    }

    #[wasm_bindgen]
    pub fn abort_training(&self) {
        self.training.set(false);
    }
}
