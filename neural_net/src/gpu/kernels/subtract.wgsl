struct Uniforms {
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;

@compute @workgroup_size(256)
fn subtract_assign(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&mat_a) { return; }
    mat_a[id.x] -= mat_b[id.x] * uniforms.scale;
}
