struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
};

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> mat_result: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) id: vec3u) {
    let col = id.x;
    let row = id.y;

    if row >= dimensions.M || col >= dimensions.N { return; }

    mat_result[row * dimensions.N + col] = 0.0;
    var sum: f32 = 0.0;
    for (var i = 0u; i < dimensions.K; i++) {
        sum += mat_a[row * dimensions.K + i] * mat_b[i * dimensions.N + col];
    }
    mat_result[row * dimensions.N + col] += sum;
}
