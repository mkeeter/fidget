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
    /// When the matrix is applied, the given bounds will be mapped into the
    /// `[-1, +1]` region (which is used for all rendering operations).
    pub fn to_transform_matrix(&self) -> Transform<f32, nalgebra::TGeneral, N> {
        let mut t = nalgebra::Translation::<f32, N>::identity();
        for (t, c) in t.vector.iter_mut().zip(&self.center) {
            *t = *c;
        }

        let mut out = Transform::<f32, nalgebra::TGeneral, N>::default();
        out *= t;

        out.matrix_mut().append_scaling_mut(self.size);
        out
    }
}
