/// Merge all tile stages into a pixel array
@group(1) @binding(0) var<storage, read_write> tile64_zmin: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> tile16_zmin: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> tile4_zmin: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> voxels: array<u32>;
@group(1) @binding(4) var<storage, read_write> heightmap: array<u32>;

// Dispatched as an 2D workgroup across render_size pixels
@compute @workgroup_size(8, 8)
fn merge_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Precompute useful sizes
    let size64 = config.render_size / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;

    // Compute indices within tile data (using render size bounds)
    let index64 = global_id.x / 64 + global_id.y / 64 * size64.x;
    let index16 = global_id.x / 16 + global_id.y / 16 * size16.x;
    let index4 = global_id.x / 4 + global_id.y / 4 * size4.x;
    let index1 = global_id.x + global_id.y * config.render_size.x;

    // Merge operations are limited to image bounds
    if global_id.x < config.image_size.x &&
       global_id.y < config.image_size.y
    {
        // Note that this index uses image size, not (rounded-up) render size
        let pixel_index = global_id.x + global_id.y * config.image_size.x;

        var out = heightmap[pixel_index];
        out = max(out, atomicLoad(&tile64_zmin[index64]));
        out = max(out, atomicLoad(&tile16_zmin[index16]));
        out = max(out, atomicLoad(&tile4_zmin[index4]));
        out = max(out, voxels[index1]);

        // Clamp to image z size and write to the heightmap
        heightmap[pixel_index] = min(out, config.image_size.z);
    }

    // Backfill operations are limited to render bounds
    //
    // Copying from high-res to low-res tiles is deliberately racey, because
    // it's to improve the odds of raymarching early-exit (but is not required
    // for correctness)
    if global_id.x < config.render_size.x &&
       global_id.y < config.render_size.y
    {
        // Copy from voxels to tile4_zmin
        if (global_id.x % 4) == 0 && (global_id.y % 4) == 0 {
            var all_present = true;
            var new_zmin = 0xFFFFFFFu;
            let corner = global_id.xy;
            for (var i=0u; i < 4u && all_present; i++) {
                for (var j=0u; j < 4u && all_present; j++) {
                    let pos = corner + vec2u(i, j);
                    let index = pos.x + pos.y * config.render_size.x;
                    let v = voxels[index];
                    if v != 0 {
                        new_zmin = min(new_zmin, v);
                    } else {
                        all_present = false;
                    }
                }
            }
            if all_present {
                atomicMax(&tile4_zmin[index4], new_zmin);
            }
        }

        // Copy from tile4_zmin to tile16_zmin
        if (global_id.x % 16) == 0 && (global_id.y % 16) == 0 {
            var all_present = true;
            var new_zmin = 0xFFFFFFFu;
            let corner = global_id.xy / 4;
            for (var i=0u; i < 4u && all_present; i++) {
                for (var j=0u; j < 4u && all_present; j++) {
                    let pos = corner + vec2u(i, j);
                    let index = pos.x + pos.y * size4.x;
                    let v = tile4_zmin[index];
                    if v != 0 {
                        new_zmin = min(new_zmin, v);
                    } else {
                        all_present = false;
                    }
                }
            }
            if all_present {
                atomicMax(&tile16_zmin[index16], new_zmin);
            }
        }

        // Copy from tile16_zmin to tile64_zmin
        if (global_id.x % 64) == 0 && (global_id.y % 64) == 0 {
            var all_present = true;
            var new_zmin = 0xFFFFFFFu;
            let corner = global_id.xy / 16;
            for (var i=0u; i < 4u && all_present; i++) {
                for (var j=0u; j < 4u && all_present; j++) {
                    let pos = corner + vec2u(i, j);
                    let index = pos.x + pos.y * size16.x;
                    let v = tile16_zmin[index];
                    if v != 0 {
                        new_zmin = min(new_zmin, v);
                    } else {
                        all_present = false;
                    }
                }
            }
            if all_present {
                atomicMax(&tile64_zmin[index64], new_zmin);
            }
        }
    }
}
