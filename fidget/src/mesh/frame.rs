//! Coordinate frames
use super::types::{Axis, X, Y, Z};

/// Marker trait for a right-handed coordinate frame
pub trait Frame {
    /// Next frame, i.e. a left rotation of [`Self::frame()`]
    type Next: Frame;

    /// Returns the right-handed frame
    fn frame() -> (Axis<3>, Axis<3>, Axis<3>);
}

/// The X-Y-Z coordinate frame
#[allow(clippy::upper_case_acronyms)]
pub struct XYZ;

/// The Y-Z-X coordinate frame
#[allow(clippy::upper_case_acronyms)]
pub struct YZX;

/// The Z-X-Y coordinate frame
#[allow(clippy::upper_case_acronyms)]
pub struct ZXY;

impl Frame for XYZ {
    type Next = YZX;
    fn frame() -> (Axis<3>, Axis<3>, Axis<3>) {
        (X, Y, Z)
    }
}

impl Frame for YZX {
    type Next = ZXY;
    fn frame() -> (Axis<3>, Axis<3>, Axis<3>) {
        (Y, Z, X)
    }
}
impl Frame for ZXY {
    type Next = XYZ;
    fn frame() -> (Axis<3>, Axis<3>, Axis<3>) {
        (Z, X, Y)
    }
}
