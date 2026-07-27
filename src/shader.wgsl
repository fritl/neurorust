@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn double(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x;
    let arraySize = arrayLength(&input);
    if i >= arraySize { return; }
    output[i] = input[i] * 2.0;
}
