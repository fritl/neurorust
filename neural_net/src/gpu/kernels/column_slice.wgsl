struct Dims {
    rows: u32,
    columns: u32,
    start_col: u32,
    batch_width: u32};
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;   // full dataset (rows x columns)
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // batch (rows x batch_width)

@compute @workgroup_size(16, 16)
fn column_slice(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= dims.batch_width || id.y >= dims.rows { return; }
    let src_col = dims.start_col + id.x;
    output[id.y * dims.batch_width + id.x] = input[id.y * dims.columns + src_col];
}
