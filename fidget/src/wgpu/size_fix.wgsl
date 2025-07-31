@group(0) @binding(0) var<storage, read_write> count: array<u32, 4>;

@compute @workgroup_size(64)
fn size_fix(@builtin(local_invocation_id) local_id: vec3u) {
    if (local_id.x == 0u) {
        if (count[0] > 32768u) {
            count[1] = 32768u;
            count[2] = (count[0] + 32767) / 32768;
        } else {
            count[1] = count[0];
            count[2] = 1u;
        }
        count[3] = 1u;
    }
}
