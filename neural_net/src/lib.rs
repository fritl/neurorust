use wasm_bindgen::prelude::*;
mod eval;
mod gpu;
mod matrix;

pub use gpu::wasm_network;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    // Leitet Rust-Panics direkt an die Browser console.error weiter (sehr hilfreich zum Debuggen)
    console_error_panic_hook::set_once();
}
