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

/// 3D position
#[derive(Copy, Clone, Facet)]
#[allow(missing_docs)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
