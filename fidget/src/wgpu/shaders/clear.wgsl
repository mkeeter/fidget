@group(1) @binding(0) var<storage, read_write> voxel_result: array<u32>;
@group(1) @binding(1) var<storage, read_write> voxel_merged: array<u32>;
@group(1) @binding(2) var<storage, read_write> geometry: array<vec4u>;

@compute @workgroup_size(8, 8)
fn clear_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Out of bounds, return
    if global_id.x < config.image_size.x &&
       global_id.y < config.image_size.y
    {
        let index = global_id.x + global_id.y * config.image_size.x;
        voxel_merged[index] = 0u;
        geometry[index] = vec4u(0u);
    }

    if global_id.x < config.render_size.x &&
       global_id.y < config.render_size.y
   {
        let index = global_id.x + global_id.y * config.render_size.x;
        voxel_result[index] = 0u;
   }
}
