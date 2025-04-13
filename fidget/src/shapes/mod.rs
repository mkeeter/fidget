/// Standard library of shapes and transforms
use crate::context::Tree;
use facet::Facet;

/// 2D position
#[derive(Copy, Clone, Facet)]
#[allow(missing_docs)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn circle_docstring() {
        assert_eq!(Circle::SHAPE.doc, &[" 2D circle"]);
    }
}
