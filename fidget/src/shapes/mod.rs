//! Standard library of shapes and transforms
use crate::context::Tree;
use facet::Facet;

mod vec;
pub use vec::{Vec2, Vec3, Vec4};

////////////////////////////////////////////////////////////////////////////////
// 2D shapes

/// 2D circle
#[derive(Clone, Facet)]
pub struct Circle {
    /// Center of the circle (in XY)
    pub center: Vec2,
    /// Circle radius
    pub radius: f64,
}

impl From<Circle> for Tree {
    fn from(v: Circle) -> Self {
        let (x, y, _) = Tree::axes();
        ((x - v.center.x).square() + (y - v.center.y).square()).sqrt()
            - v.radius
    }
}

////////////////////////////////////////////////////////////////////////////////
// 3D shapes

/// 3D sphere
#[derive(Clone, Facet)]
pub struct Sphere {
    /// Center of the circle (in XYZ)
    pub center: Vec3,
    /// Sphere radius
    pub radius: f64,
}

impl From<Sphere> for Tree {
    fn from(v: Sphere) -> Self {
        let (x, y, z) = Tree::axes();
        ((x - v.center.x).square()
            + (y - v.center.y).square()
            + (z - v.center.z).square())
        .sqrt()
            - v.radius
    }
}

////////////////////////////////////////////////////////////////////////////////
// CSG operations

/// Take the union of a set of shapes
///
/// If the input is empty, returns an constant empty tree (at +∞)
#[derive(Clone, Facet)]
pub struct Union {
    /// List of shapes to merge
    pub input: Vec<Tree>,
}

impl From<Union> for Tree {
    fn from(v: Union) -> Self {
        if v.input.is_empty() {
            // XXX should this be an error instead?
            Tree::constant(f64::INFINITY)
        } else {
            fn recurse(s: &[Tree]) -> Tree {
                match s.len() {
                    1 => s[0].clone(),
                    n => recurse(&s[..n / 2]).min(recurse(&s[n / 2..])),
                }
            }
            recurse(&v.input)
        }
    }
}

/// Take the intersection of a set of shapes
///
/// If the input is empty, returns a constant full tree (at -∞)
#[derive(Clone, Facet)]
pub struct Intersection {
    /// List of shapes to intersect
    pub input: Vec<Tree>,
}

impl From<Intersection> for Tree {
    fn from(v: Intersection) -> Self {
        if v.input.is_empty() {
            // XXX should this be an error instead?
            Tree::constant(-f64::INFINITY)
        } else {
            fn recurse(s: &[Tree]) -> Tree {
                match s.len() {
                    1 => s[0].clone(),
                    n => recurse(&s[..n / 2]).max(recurse(&s[n / 2..])),
                }
            }
            recurse(&v.input)
        }
    }
}

/// Computes the inverse of a shape
#[derive(Clone, Facet)]
pub struct Inverse {
    /// Shape to invert
    pub shape: Tree,
}

impl From<Inverse> for Tree {
    fn from(v: Inverse) -> Self {
        -v.shape
    }
}

/// Take the difference of two shapes
#[derive(Clone, Facet)]
pub struct Difference {
    /// Original shape
    pub shape: Tree,
    /// Shape to be subtracted from the original
    pub cutout: Tree,
}

impl From<Difference> for Tree {
    fn from(v: Difference) -> Self {
        v.shape.max(-v.cutout)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Transforms

/// Move a shape
#[derive(Clone, Facet)]
pub struct Move {
    /// Shape to move
    pub shape: Tree,
    /// Position offset
    pub offset: Vec3,
}

impl From<Move> for Tree {
    fn from(v: Move) -> Self {
        v.shape.remap_affine(nalgebra::convert(
            nalgebra::Translation3::<f64>::new(
                v.offset.x, v.offset.y, v.offset.z,
            ),
        ))
    }
}

/// Non-uniform scaling
#[derive(Clone, Facet)]
pub struct Scale {
    /// Shape to scale
    pub shape: Tree,
    /// Scale to apply on each axis
    pub scale: Vec3,
}

impl From<Scale> for Tree {
    fn from(v: Scale) -> Self {
        v.shape
            .remap_affine(nalgebra::convert(nalgebra::Scale3::<f64>::new(
                v.scale.x, v.scale.y, v.scale.z,
            )))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn circle_docstring() {
        assert_eq!(Circle::SHAPE.doc, &[" 2D circle"]);
    }
}
