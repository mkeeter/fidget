//! Standard library of shapes and transforms
use crate::context::Tree;
use facet::Facet;

mod vec;
pub use vec::{Vec2, Vec3};

/// 2D circle
#[derive(Clone, Facet)]
#[allow(missing_docs)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f64,
}

impl From<Circle> for Tree {
    fn from(v: Circle) -> Self {
        let (x, y, _) = Tree::axes();
        ((x - v.center.x).square() + (y - v.center.y).square()).sqrt()
            - v.radius
    }
}

/// 3D circle
#[derive(Clone, Facet)]
#[allow(missing_docs)]
pub struct Sphere {
    pub center: Vec3,
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

/// Take the union of a set of shapes
///
/// If the input is empty, returns an constant empty tree (at +∞)
#[derive(Clone, Facet)]
#[allow(missing_docs)]
pub struct Union {
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
#[allow(missing_docs)]
pub struct Intersection {
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

////////////////////////////////////////////////////////////////////////////////

/// Move a shape
#[derive(Clone, Facet)]
#[allow(missing_docs)]
pub struct Move {
    pub shape: Tree,
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

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn circle_docstring() {
        assert_eq!(Circle::SHAPE.doc, &[" 2D circle"]);
    }
}
