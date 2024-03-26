use crate::eval::types::Interval;

/// A bounded region in space, typically used as a render region
struct Bounds {
    x: Interval,
    y: Interval,
    z: Interval,
}
