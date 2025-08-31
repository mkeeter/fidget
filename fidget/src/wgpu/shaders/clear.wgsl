@group(1) @binding(0) var<storage, read_write> tile64_zmin: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile16_zmin: array<u32>;
@group(1) @binding(2) var<storage, read_write> tile4_zmin: array<u32>;
@group(1) @binding(3) var<storage, read_write> voxel_result: array<u32>;
@group(1) @binding(4) var<storage, read_write> voxel_merged: array<u32>;
@group(1) @binding(5) var<storage, read_write> geometry: array<vec4u>;

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

   let size64 = config.render_size / 64;
   if global_id.x < size64.x &&
      global_id.y < size64.y
   {
        let index = global_id.x + global_id.y * size64.x;
        tile64_zmin[index] = 0u;
   }

   let size16 = config.render_size / 16;
   if global_id.x < size16.x &&
      global_id.y < size16.y
   {
        let index = global_id.x + global_id.y * size16.x;
        tile16_zmin[index] = 0u;
   }

   let size4 = config.render_size / 4;
   if global_id.x < size4.x &&
      global_id.y < size4.y
   {
        let index = global_id.x + global_id.y * size4.x;
        tile4_zmin[index] = 0u;
   }
}
