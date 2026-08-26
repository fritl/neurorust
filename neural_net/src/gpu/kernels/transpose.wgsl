struct Uniforms {
    rows: u32,    // rows of the INPUT
    columns: u32, // columns of the INPUT
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn transpose(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= uniforms.columns || id.y >= uniforms.rows { return; }
    output[id.x * uniforms.rows + id.y] = input[id.y * uniforms.columns + id.x];
}
