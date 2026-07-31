struct Uniforms {
    columns_a: u32,
    is_broadcast: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;

// This limits the maximum matrix size or 256*65535 elements. Because you can only dispatch 65535 work groups at max
@compute @workgroup_size(256)
fn inplace_add(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&mat_a) { return; }
    var b_index: u32 = id.x;
    if uniforms.is_broadcast != 0u {
        b_index = id.x / uniforms.columns_a;
    }
    mat_a[id.x] += mat_b[b_index];
}
