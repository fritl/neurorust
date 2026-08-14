@group(0) @binding(0) var<storage, read> mat_a: array<f32>;
@group(0) @binding(0) var<storage, read> sigmoid_output: array<f32>;
@group(0) @binding(1) var<storage, read_write> mat_result: array<f32>;

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) id: vec3u) {
    if id.x > arrayLength(&mat_result) { return; }
    mat_result[id.x] = 1.0 / (1.0 + exp(-mat_a[id.x]));
}

@compute @workgroup_size(256)
fn sigmoid_prime(@builtin(global_invocation_id) id: vec3u) {
    if id.x > arrayLength(&mat_result) { return; }
    mat_result[id.x] = sigmoid_output[id.x] * (1.0 - sigmoid_output[id.x]);
}
