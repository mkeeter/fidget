//! Helper types and functions for camera and viewport manipulation
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimNameAdd, DimNameSub,
    DimNameSum, OMatrix, OVector, U1,
};

/// Helper object for a camera in 3D space
///
/// Rendering and meshing happen in the ±1 square or cube; these are referred to
/// as "world" coordinates.  A camera generates a homogeneous transform matrix
/// that maps from positions in world coordinates to "model" coordinates.
///
/// When rendering, screen pixels are converted into the ±1 region by
/// [`RegionSize::screen_to_world`].  If the render region is not square, then
/// the shorter axis is clamped to ±1 and the longer axis will exceed that
/// value.
///
/// The camera converts points in world coordinates (±1) into model coordinates
/// for actual evaluation against a function.  For example, if the region of
/// interest for function evaluation spans `0 < (x, y, z) < 10`, the camera's
/// role is to translate `x_world = 0 → x_model = 5`:
///
/// ```
/// # use nalgebra::{Vector3, Point3};
/// # use fidget::render::Camera;
/// let camera = Camera::<3>::from_center_and_scale(
///     Vector3::new(5.0, 5.0, 5.0), 5.0
/// );
/// let mat = camera.world_to_model();
/// let pos = mat.transform_point(&Point3::new(0.0, 0.0, 0.0));
/// assert_eq!(pos, Point3::new(5.0, 5.0, 5.0));
/// ```
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
        let mut mat = nalgebra::OMatrix::<
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
    ) -> nalgebra::OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    > {
        self.mat
    }
}

impl<const N: usize> Default for Camera<N>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    <nalgebra::Const<N> as DimNameAdd<nalgebra::Const<1>>>::Output:
        DimNameSub<nalgebra::Const<1>>,
    OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >: Copy,
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
