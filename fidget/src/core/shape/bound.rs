/// A bounded region in space, typically used as a render region
///
/// Right now, all spatial operations take place in a cubical region, so we
/// specify bounds as a lower corner and region size.
pub struct Bounds {
    /// Lower corner of the bounds
    pub corner: nalgebra::Vector3<f32>,
    /// Size of the bounds (on each axis)
    pub size: f32,
}

impl Default for Bounds {
    /// By default, the bounds are the `[-1, +1]` cube
    fn default() -> Self {
        Self {
            corner: nalgebra::Vector3::new(-1.0, -1.0, -1.0),
            size: 2.0,
        }
    }
}
