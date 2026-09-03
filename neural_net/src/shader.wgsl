@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn double(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
) {
    let index = (workgroup_id.y * num_workgroups.x + workgroup_id.x) * 256u + local_invocation_id.x;
    let arraySize = arrayLength(&input);
    if index >= arraySize { return; }
    output[index] = input[index] * 2.0;
}
