struct Uniforms {
    rows: u32,
    columns: u32,
};

@group(0) @binding(0) var<uniform> dims: Uniforms;

@group(0) @binding(1) var<storage, read_write> writable_input: array<f32>; // (rows, columns)
@group(0) @binding(1) var<storage, read> input: array<f32>; // (rows, columns)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Input shape: (rows, cols)
// Output shape: (1, cols)
@compute @workgroup_size(256)
fn column_max(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= dims.columns { return; }
    var max = input[id.x];
    for (var r = 1u; r < dims.rows; r++) {
        let v = input[r * dims.columns + id.x];
        if v > max { max = v; }
    }
    output[id.x] = max;
}

@group(0) @binding(2) var<storage, read> col_max: array<f32>;
// Input shape: (rows, cols)
// Result saved in input
@compute @workgroup_size(16, 16)
fn exp_shifted(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= dims.columns || id.y >= dims.rows { return; }
    let idx = id.y * dims.columns + id.x;
    writable_input[idx] = exp(writable_input[idx] - col_max[id.x]);
}

// Input shape: (rows, cols)
// Output shape: (1, cols)
@compute @workgroup_size(256)
fn column_sum(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= dims.columns { return; }
    var sum = 0.0;
    for (var r = 0u; r < dims.rows; r++) {
        let idx = r * dims.columns + id.x;
        sum += input[idx];
    }
    output[id.x] = sum;
}

@group(0) @binding(2) var<storage, read> col_sum: array<f32>;
// Input shape: (rows, cols)
// Result saved in input
@compute @workgroup_size(16, 16)
fn normalize(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= dims.columns || id.y >= dims.rows { return; }
    let idx = id.y * dims.columns + id.x;
    writable_input[idx] = writable_input[idx] / col_sum[id.x];
}

@group(0) @binding(3) var<storage, read> target_one_hot: array<f32>;
// Input shape: (rows, cols)
// Output shape: (rows, cols)
@compute @workgroup_size(16, 16)
fn cross_entropy_gradient(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= dims.columns || id.y >= dims.rows { return; }
    let idx = id.y * dims.columns + id.x;
    output[idx] = (input[idx] - target_one_hot[idx]) / f32(dims.columns);
}
