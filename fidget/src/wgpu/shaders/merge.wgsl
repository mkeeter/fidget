/// Backfill tile_zmin from subtile_zmin
@group(1) @binding(0) var<storage, read> tile64_zmin: array<u32>;
@group(1) @binding(1) var<storage, read> tile16_zmin: array<u32>;
@group(1) @binding(2) var<storage, read> tile4_zmin: array<u32>;
@group(1) @binding(3) var<storage, read> result: array<u32>;
@group(1) @binding(4) var<storage, read_write> merged: array<u32>;

@compute @workgroup_size(8, 8)
fn merge_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Out of bounds, return
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }
    let pixel_index = global_id.x + global_id.y * config.image_size.x;
    if merged[pixel_index] != 0 {
        return;
    }

    let size64 = config.render_size / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;

    var out = 0u;
    let index64 = global_id.x / 64 + global_id.y / 64 * size64.x;
    out = max(out, tile64_zmin[index64]);

    let index16 = global_id.x / 16 + global_id.y / 16 * size16.x;
    out = max(out, tile16_zmin[index16]);

    let index4 = global_id.x / 4 + global_id.y / 4 * size4.x;
    out = max(out, tile4_zmin[index4]);

    out = max(out, result[global_id.x + global_id.y * config.render_size.x]);

    merged[pixel_index] = out;
}
