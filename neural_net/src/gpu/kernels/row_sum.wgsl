@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn row_sum(@builtin(global_invocation_id) id: vec3u) {
    let rows = arrayLength(&output);
    if id.x >= rows { return; }
    let columns = arrayLength(&input) / rows;
    var sum = 0f;
    let row_start = id.x * columns;
    for (var c = 0u; c < columns; c++) {
        sum += input[row_start + c];
    }
    output[id.x] = sum;
}
