//! Helper types and functions for camera and viewport manipulation
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimNameAdd, DimNameSub,
    DimNameSum, OMatrix, OPoint, OVector, Point2, Point3, U1,
};

/// Helper object for a camera in 2D or 3D space
///
/// Rendering and meshing happen in the ±1 square or cube; these are referred to
/// as _world_ coordinates.  A `Camera` generates a homogeneous transform matrix
/// that maps from positions in world coordinates to _model_ coordinates, which
/// can be whatever you want.
///
/// Here's an example of using a `Camera` to focus on the region `[4, 6]`:
///
/// ```
/// # use nalgebra::{Vector2, Point2};
/// # use fidget::render::{Camera, TransformPoint};
/// let camera = Camera::<2>::from_center_and_scale(
///     Vector2::new(5.0, 5.0), 1.0
/// );
///
/// //   -------d-------
/// //   |             |
/// //   |             |
/// //   c      a      b
/// //   |             |
/// //   |             |
/// //   -------e-------
/// let a = camera.transform_point(&Point2::new(0.0, 0.0));
/// assert_eq!(a, Point2::new(5.0, 5.0));
///
/// let b = camera.transform_point(&Point2::new(1.0, 0.0));
/// assert_eq!(b, Point2::new(6.0, 5.0));
///
/// let c = camera.transform_point(&Point2::new(-1.0, 0.0));
/// assert_eq!(c, Point2::new(4.0, 5.0));
///
/// let d = camera.transform_point(&Point2::new(0.0, 1.0));
/// assert_eq!(d, Point2::new(5.0, 6.0));
///
/// let e = camera.transform_point(&Point2::new(0.0, -1.0));
/// assert_eq!(e, Point2::new(5.0, 4.0));
/// ```
///
/// See also
/// [`RegionSize::screen_to_world`](crate::render::RegionSize::screen_to_world),
/// which converts from screen to world coordinates.
///
/// Apologies for the terrible trait bounds; they're necessary to persuade the
/// internals to type-check, but shouldn't be noticeable to library users.
#[derive(Copy, Clone, Debug)]
pub struct Camera<const N: usize>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator:
        Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator:
        Allocator<
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >: Copy,
    Self: TransformPoint<N>,
{
    mat: OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >,
}

impl<const N: usize> Camera<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator:
        Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator:
        Allocator<
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >: Copy,
    Self: TransformPoint<N>,
{
    /// Builds a camera from a center (in world coordinates) and a scale
    ///
    /// The resulting camera will point at the center, and the viewport will be
    /// ± `scale` in size.
    pub fn from_center_and_scale(
        center: OVector<
            f32,
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
        scale: f32,
    ) -> Self {
        let mut mat = OMatrix::<
            f32,
            <Const<N> as DimNameAdd<Const<1>>>::Output,
            <Const<N> as DimNameAdd<Const<1>>>::Output,
        >::identity();
        mat.append_scaling_mut(scale);
        mat.append_translation_mut(&center);
        Self { mat }
    }

    /// Returns the world-to-model transform matrix
    pub fn world_to_model(
        &self,
    ) -> OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    > {
        self.mat
    }

    /// Applies a translation (in model units) to the current camera position
    pub fn translate(
        &mut self,
        dt: OVector<
            f32,
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    ) {
        self.mat.append_translation_mut(&dt);
    }

    /// Zooms the camera about a particular position in world space
    pub fn zoom(&mut self, amount: f32, pos: Option<OPoint<f32, Const<N>>>) {
        match pos {
            Some(p) => {
                let pos_before = self.transform_point(&p);
                self.mat.append_scaling_mut(amount);
                let pos_after = self.transform_point(&p);
                self.mat.append_translation_mut(&(pos_before - pos_after));
            }
            None => {
                self.mat.append_scaling_mut(amount);
            }
        }
    }

    fn transform_point(
        &self,
        pt: &OPoint<f32, Const<N>>,
    ) -> OPoint<f32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>{
        TransformPoint::<N>::transform_point(self, pt)
    }
}

/// Helper trait for being able to transform a point
///
/// `transform_point` is only implemented for specific matrix sizes in
/// `nalgebra`, so we can't make it a function on every `Camera<N>`.
pub trait TransformPoint<const N: usize>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator:
        Allocator<
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
{
    /// Transforms a point from one coordinate system to another
    fn transform_point(
        &self,
        pt: &OPoint<f32, Const<N>>,
    ) -> OPoint<f32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>;
}

impl TransformPoint<2> for Camera<2> {
    fn transform_point(&self, pt: &Point2<f32>) -> Point2<f32> {
        self.mat.transform_point(pt)
    }
}

impl TransformPoint<3> for Camera<3> {
    fn transform_point(&self, pt: &Point3<f32>) -> Point3<f32> {
        self.mat.transform_point(pt)
    }
}

impl<const N: usize> Default for Camera<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator:
        Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator:
        Allocator<
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >: Copy,
    Self: TransformPoint<N>,
{
    fn default() -> Self {
        Self {
            mat: OMatrix::<
                f32,
                <Const<N> as DimNameAdd<Const<1>>>::Output,
                <Const<N> as DimNameAdd<Const<1>>>::Output,
            >::identity(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_camera_from_center_and_scale() {
        let c = Camera::<3>::from_center_and_scale(
            Vector3::new(1.0, 2.0, 3.0),
            5.0,
        );
        let mat = c.world_to_model();

        let pt = mat.transform_point(&Point3::new(0.0, 0.0, 0.0));
        assert_eq!(pt.x, 1.0);
        assert_eq!(pt.y, 2.0);
        assert_eq!(pt.z, 3.0);

        let pt = mat.transform_point(&Point3::new(1.0, 0.0, 0.0));
        assert_eq!(pt.x, 1.0 + 5.0);
        assert_eq!(pt.y, 2.0);
        assert_eq!(pt.z, 3.0);

        let pt = mat.transform_point(&Point3::new(-1.0, 0.0, 0.0));
        assert_eq!(pt.x, 1.0 - 5.0);
        assert_eq!(pt.y, 2.0);
        assert_eq!(pt.z, 3.0);

        let pt = mat.transform_point(&Point3::new(0.0, 1.0, 0.0));
        assert_eq!(pt.x, 1.0);
        assert_eq!(pt.y, 2.0 + 5.0);
        assert_eq!(pt.z, 3.0);

        let pt = mat.transform_point(&Point3::new(0.0, -1.0, 0.0));
        assert_eq!(pt.x, 1.0);
        assert_eq!(pt.y, 2.0 - 5.0);
        assert_eq!(pt.z, 3.0);

        let pt = mat.transform_point(&Point3::new(0.0, 0.0, 1.0));
        assert_eq!(pt.x, 1.0);
        assert_eq!(pt.y, 2.0);
        assert_eq!(pt.z, 3.0 + 5.0);

        let pt = mat.transform_point(&Point3::new(0.0, 0.0, -1.0));
        assert_eq!(pt.x, 1.0);
        assert_eq!(pt.y, 2.0);
        assert_eq!(pt.z, 3.0 - 5.0);
    }
}
