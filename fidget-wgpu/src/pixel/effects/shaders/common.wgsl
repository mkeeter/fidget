/// Distance pixel with an associated shape tag
struct TaggedRawDistancePixel {
    /// Distance associated with this pixel
    distance: RawDistancePixel,

    /// Shape index
    index: u32,
}
