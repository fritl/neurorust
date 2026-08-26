use std::{collections::HashMap, rc::Rc};

use crate::gpu::{matrix::GpuMatrix, state::GpuState};

pub struct BufferPool {
    buffer: HashMap<(usize, usize), Vec<GpuMatrix>>,
    gpu_state: Rc<GpuState>,
}

impl BufferPool {
    pub fn new(gpu_state: Rc<GpuState>) -> BufferPool {
        BufferPool {
            buffer: HashMap::new(),
            gpu_state,
        }
    }

    pub fn get(&mut self, rows: usize, columns: usize) -> GpuMatrix {
        let list = self.buffer.entry((rows, columns)).or_default();
        list.pop()
            .unwrap_or_else(|| GpuMatrix::empty(rows, columns, Rc::clone(&self.gpu_state)))
    }

    pub fn recycle(&mut self, gpu_matrix: GpuMatrix) {
        self.buffer
            .entry(gpu_matrix.shape())
            .or_default()
            .push(gpu_matrix);
    }
}
