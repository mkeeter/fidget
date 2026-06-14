struct PackedVoxel {
    /// Bit-packed values for normal and index
    ///
    /// Normal is stored as an `[i8; 2]`, normalized to a length of 127 with the
    /// Z component implied and positive.
    norm_index: u32,

    /// Depth of the voxel
    z: u32,
}

/// Duplicated from `voxel/shaders/common.wgsl`
struct GeometryPixel {
    normal: vec3f,
    depth: u32,
}

/// Geometry pixel tagged with an image index
struct TaggedGeometryPixel {
    pixel: GeometryPixel,
    index: u32,
}

fn pack(p: TaggedGeometryPixel) -> PackedVoxel {
    let index = p.index + config.index_base;

    var packed_normal_index = min(index, 0xFFFF) << 16;

    // Return a special bit pattern for pixels with invalid normals
    let normal_length = length(p.pixel.normal);
    if normal_length == 0.0
        || normal_length != normal_length
        || p.pixel.normal.z < 0.0
    {
        packed_normal_index |= 0x0008080; // [-128, -128]
    } else {
        let norm_norm = p.pixel.normal / normal_length;
        let nx = bitcast<u32>(i32(norm_norm.x * 127.0)) & 0xFF;
        let ny = bitcast<u32>(i32(norm_norm.y * 127.0)) & 0xFF;
        packed_normal_index |= nx | (ny << 8);
    }
    return PackedVoxel(packed_normal_index, p.pixel.depth);
}

fn unpack(p: PackedVoxel) -> TaggedGeometryPixel {
    let depth = p.z;
    let index = p.norm_index >> 16;
    let signed = bitcast<i32>(p.norm_index);
    let dx_i = extractBits(signed, 0, 8);
    let dy_i = extractBits(signed, 8, 8);

    // Start with an invalid normal, then populate it if this isn't the tagged
    // case of [-128, -128].
    var normal = vec3f(0.0);
    if !(dx_i == -128 && dy_i == -128) {
        let dx = f32(dx_i) / 127.0;
        let dy = f32(dy_i) / 127.0;
        let dz = sqrt(max(1.0 - dx*dx - dy*dy, 0.0));
        normal = vec3(dx, dy, dz);
    }

    return TaggedGeometryPixel(
        GeometryPixel(normal, depth),
        index,
    );
}
