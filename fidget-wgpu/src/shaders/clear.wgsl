@group(1) @binding(0) var<storage, read_write> tile16_count: array<u32, 4>;
@group(1) @binding(1) var<storage, read_write> tile16_sort: array<u32, 4>;
@group(1) @binding(2) var<storage, read_write> tile16_zhist: array<u32, 4>;
@group(1) @binding(3) var<storage, read_write> tile4_count: array<u32, 4>;
@group(1) @binding(4) var<storage, read_write> tile4_sort: array<u32, 4>;
@group(1) @binding(5) var<storage, read_write> tile4_zhist: array<u32, 16>;

// Dispatched as a single workgroup
@compute @workgroup_size(64)
fn clear_main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let i = global_id.x;

    if i < 4 {
        tile16_count[i] = 0u;
        tile16_sort[i] = 0u;
        tile16_zhist[i] = 0u;
        tile4_count[i] = 0u;
        tile4_sort[i] = 0u;
    }

    if i < 16 {
        tile4_zhist[i] = 0u;
    }

    // Reset the global tape position
    if i == 0u {
        atomicStore(&config.tape_data_offset, atomicLoad(&config.root_tape_len));
    }

}
