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

/// Move a shape
#[derive(Clone)]
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
