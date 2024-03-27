use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimNameAdd, DimNameSub,
    DimNameSum, OVector, Transform, U1,
};

/// A bounded region in space, typically used as a render region
///
/// Right now, all spatial operations take place in a cubical region, so we
/// specify bounds as a center point and region size.
#[derive(Copy, Clone, Debug)]
pub struct Bounds<const N: usize> {
    /// Center of the bounds
    pub center: OVector<f32, Const<N>>,

    /// Size of the bounds in each direction
    ///
    /// The full bounds are given by `[center - size, center + size]` on each
    /// axis.
    pub size: f32,
}

impl<const N: usize> Default for Bounds<N> {
    /// By default, the bounds are the `[-1, +1]` region
    fn default() -> Self {
        let center = OVector::<f32, Const<N>>::zeros();
        Self { center, size: 1.0 }
    }
}

impl<const N: usize> Bounds<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<U1>,
{
    /// Returns a homogeneous transform matrix for these bounds
    ///
    /// When this matrix is applied, the `[-1, +1]` region (used for all
    /// rendering operations) will be remapped to the original bounds.
    pub fn transform(&self) -> Transform<f32, nalgebra::TGeneral, N> {
        let mut t = nalgebra::Translation::<f32, N>::identity();
        t.vector = self.center / self.size;

        let mut out = Transform::<f32, nalgebra::TGeneral, N>::default();
        out.matrix_mut().append_scaling_mut(self.size);
        out *= t;

        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{Point2, Vector2};

    #[test]
    fn bounds_default() {
        let b = Bounds::default();
        let t = b.transform();
        assert_eq!(
            t.transform_point(&Point2::new(-1.0, -1.0)),
            Point2::new(-1.0, -1.0)
        );
        assert_eq!(
            t.transform_point(&Point2::new(0.5, 0.0)),
            Point2::new(0.5, 0.0)
        );
    }

    #[test]
    fn bounds_scale() {
        let b = Bounds {
            center: Vector2::zeros(),
            size: 0.5,
        };
        let t = b.transform();
        assert_eq!(
            t.transform_point(&Point2::new(-1.0, -1.0)),
            Point2::new(-0.5, -0.5)
        );
        assert_eq!(
            t.transform_point(&Point2::new(1.0, 0.0)),
            Point2::new(0.5, 0.0)
        );
    }

    #[test]
    fn bounds_translate() {
        let b = Bounds {
            center: Vector2::new(1.0, 2.0),
            size: 1.0,
        };
        let t = b.transform();
        assert_eq!(
            t.transform_point(&Point2::new(-1.0, -1.0)),
            Point2::new(0.0, 1.0)
        );
        assert_eq!(
            t.transform_point(&Point2::new(1.0, 0.0)),
            Point2::new(2.0, 2.0)
        );
    }

    #[test]
    fn bounds_translate_scale() {
        let b = Bounds {
            center: Vector2::new(0.5, 0.5),
            size: 0.5,
        };
        let t = b.transform();
        assert_eq!(
            t.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(0.5, 0.5)
        );
        assert_eq!(
            t.transform_point(&Point2::new(-1.0, -1.0)),
            Point2::new(0.0, 0.0)
        );
        assert_eq!(
            t.transform_point(&Point2::new(1.0, 1.0)),
            Point2::new(1.0, 1.0)
        );
    }
}
