@group(0) @binding(0) var<storage, read> mat_a: array<f32>;
@group(0) @binding(1) var<storage, read> mat_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mat_result: array<f32>;

@compute @workgroup_size(256)
fn hadamard(@builtin(global_invocation_id) id: vec3u) {
    if id.x > arrayLength(&mat_result) { return; }
    mat_result[id.x] = mat_a[id.x] * mat_b[id.x];
}
