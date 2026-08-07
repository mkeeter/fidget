/// Merge all tile stages into a pixel array
@group(2) @binding(0) var<storage, read> tile64_zmin: array<RawDistancePixel>;
@group(2) @binding(1) var<storage, read> tile8_zmin: array<RawDistancePixel>;
@group(2) @binding(2) var<storage, read_write> pixels: array<RawDistancePixel>;

// Dispatched as an 2D workgroup across render_size pixels
@compute @workgroup_size(8, 8)
fn merge_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    pixels[0] = RawDistancePixel(bitcast<u32>(config.image_size.x)); // XXX DEBUG
    // the two `tiles` buffers are rounded up, but `pixels` is original size, so
    // it's a tigher bound here.
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }

    // Precompute useful sizes
    let size64 = config.render_size / 64;
    let size8 = size64 * 8u;

    // Compute indices within tile data.  Note that the two tile buffers use
    // render size, while pixels use image size.
    let index64 = global_id.x / 64 + global_id.y / 64 * size64.x;
    let index8 = global_id.x / 8 + global_id.y / 8 * size8.x;
    let index1 = global_id.x + global_id.y * config.image_size.x;

    // Merge tiles into the voxels image, preferring higher resolution
    var out = pixels[index1];
    out = merge_pixel(out, tile8_zmin[index8]);
    out = merge_pixel(out, tile64_zmin[index64]);

    pixels[index1] = out;

}

fn merge_pixel(a: RawDistancePixel, b: RawDistancePixel) -> RawDistancePixel {
    if a.data == 0u {
        return b;
    } else {
        return a;
    }
}
