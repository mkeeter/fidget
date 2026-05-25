@group(2) @binding(0) var<storage, read_write> tile16_count: array<u32, 4>;
@group(2) @binding(1) var<storage, read_write> tile16_sort: array<u32, 4>;
@group(2) @binding(2) var<storage, read_write> tile4_count: array<u32, 4>;
@group(2) @binding(3) var<storage, read_write> tile4_sort: array<u32, 4>;

// zhist_buf is 4x u32, padding to 256 bytes, then 16x u32.  We only care about
// clearing the u32 ranges, i.e indices 0..4 and 64..80.
@group(2) @binding(4) var<storage, read_write> zhist_buf: array<u32, 80>;

// Dispatched as a single workgroup
@compute @workgroup_size(64)
fn clear_main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let i = global_id.x;

    if i < 4 {
        tile16_count[i] = 0u;
        tile16_sort[i] = 0u;
        tile4_count[i] = 0u;
        tile4_sort[i] = 0u;
        zhist_buf[i] = 0u;
    }

    if i < 16 {
        zhist_buf[64 + i] = 0u;
    }

    // Reset the global tape position
    if i == 0u {
        atomicStore(&config.tape_data_offset, atomicLoad(&config.root_tape_len));
    }

}
