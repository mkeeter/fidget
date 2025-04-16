//! GLSL-style vector types
//!
//! We use dedicated types (instead of `nalgebra` types) because we must derive
//! `Facet` on them, so are limited by the orphan rule.
use facet::Facet;

/// 2D position
///
/// This is a separate type (instead of [`nalgebra::Vector2`]) because we want
/// to implement the `Facet` trait on it.
#[derive(Copy, Clone, Facet)]
#[allow(missing_docs)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl From<nalgebra::Vector2<f64>> for Vec2 {
    fn from(value: nalgebra::Vector2<f64>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<Vec2> for nalgebra::Vector2<f64> {
    fn from(value: Vec2) -> Self {
        Self::new(value.x, value.y)
    }
}

/// 3D position
#[derive(Copy, Clone, Facet)]
#[allow(missing_docs)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<nalgebra::Vector3<f64>> for Vec3 {
    fn from(value: nalgebra::Vector3<f64>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Vec3> for nalgebra::Vector3<f64> {
    fn from(value: Vec3) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}
