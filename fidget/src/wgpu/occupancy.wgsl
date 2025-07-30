// Occupancy data for 64 children
struct Occupancy {
    // Each member contains 2-bit values for 16 subregions
    data: array<u32, 4>,
}

// Finds the number of voxel children in an occupancy mask
fn occupancy_size(o: Occupancy) -> u32 {
    var out = 0u;
    for (var i=0u; i < 4u; i += 1u) {
        out += countOneBits(o.data[i] & (o.data[i] >> 1u) & 0x55555555);
    }
    return out;
}

// Finds the offset of a particular voxel child (in the 0-64 range)
fn occupancy_offset(o: Occupancy, i: u32) -> u32 {
    var out = 0u;
    for (var j=0u; j < i / 16u; j += 1u) {
        out += countOneBits(o.data[j] & (o.data[j] >> 1u) & 0x55555555);
    }
    // Mask off high bits with a shift
    let d = o.data[i / 16u] << (2u * (16u - (i % 16u)));
    return out + countOneBits(d & (d >> 1u) & 0x55555555);
}

