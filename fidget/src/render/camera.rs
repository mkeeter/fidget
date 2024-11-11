//! Helper types and functions for camera and viewport manipulation
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimNameAdd, DimNameSub,
    DimNameSum, OMatrix, OVector, Vector2, Vector3, U1,
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

/// Image size in pixels
///
/// The screen coordinate space is the following:
///
/// ```text
///        0 ------------> width
///        |             |
///        |             |
///        |             |
///        V--------------
///   height
/// ```
#[derive(Copy, Clone, Debug)]
pub struct RegionSize<const N: usize>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OVector<u32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>: Copy,
{
    size: OVector<u32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
}

impl<const N: usize> RegionSize<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    <DefaultAllocator as nalgebra::allocator::Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>>::Buffer<u32>: std::marker::Copy,
{
    /// Builds a matrix that converts from screen to world coordinates
    ///
    /// The map from screen to world coordinates is as following:
    /// ```text
    ///       -1           y = +1
    ///        0-------------^-------------> width
    ///        |             |             |
    ///        |             |             |
    ///        |             |             |
    ///   x = -1 <-----------0-------------> x = +1
    ///        |             |             |
    ///        |             |             |
    ///        |             V             |
    ///        V---------- y = -1 ---------
    ///   height
    /// ```
    ///
    /// (with `+z` pointing out of the screen)
    ///
    /// Note that Y axis is flipped: screen coordinates have `+y` pointing down,
    /// but world coordinates have it pointing up.  For both X and Y
    /// coordinates, the `+1` value is located one pixel beyond the edge of the
    /// screen region (off the right edge for X, and off the top edge for Y).
    pub fn screen_to_world(
        &self,
    ) -> OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    > {
        let mut center = self.size.cast::<f32>() / 2.0;
        center[1] -= 1.0;
        let scale = 2.0 / self.size.min() as f32;

        let mut out = OMatrix::<
            f32,
            <Const<N> as DimNameAdd<Const<1>>>::Output,
            <Const<N> as DimNameAdd<Const<1>>>::Output,
        >::identity();
        out.append_translation_mut(&(-center));
        let mut scale = OVector::<f32, _>::from_element(scale);
        scale[1] *= -1.0;
        out.append_nonuniform_scaling_mut(&scale);
        out
    }

    /// Returns the width of the image (in pixels or voxels)
    pub fn width(&self) -> u32 {
        self.size[0]
    }

    /// Returns the height of the image (in pixels or voxels)
    pub fn height(&self) -> u32 {
        self.size[1]
    }
}

/// Builds a `RegionSize` with the same dimension on all axes
impl<const N: usize> From<u32> for RegionSize<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    <DefaultAllocator as nalgebra::allocator::Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>>::Buffer<u32>: std::marker::Copy,
{
    fn from(v: u32) -> Self {
        Self {
            size: OVector::<
                u32,
                <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                    Const<1>,
                >>::Output,
            >::from_element(v)
        }
    }
}

/// Size for 2D rendering of an image
pub type ImageSize = RegionSize<2>;
impl ImageSize {
    /// Builds a new `ImageSize` object from width and height in pixels
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            size: Vector2::new(width, height),
        }
    }
}

/// Size for 3D rendering of an image
pub type VoxelSize = RegionSize<3>;
impl VoxelSize {
    /// Builds a new `VoxelSize` object from width, height, and depth in voxels
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            size: Vector3::new(width, height, depth),
        }
    }

    /// Returns the depth of the image (in voxels)
    pub fn depth(&self) -> u32 {
        self.size.z
    }
}

impl<const N: usize> std::ops::Index<usize> for RegionSize<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OVector<u32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>: Copy,
{
    type Output = u32;
    fn index(&self, i: usize) -> &Self::Output {
        &self.size[i]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{Point2, Point3, Vector3};

    #[test]
    fn test_screen_size() {
        let image_size = ImageSize::new(1000, 500);
        let mat = image_size.screen_to_world();

        let pt = mat.transform_point(&Point2::new(500.0, 249.0));
        assert_eq!(pt.x, 0.0);
        assert_eq!(pt.y, 0.0);

        let pt = mat.transform_point(&Point2::new(500.0, -1.0));
        assert_eq!(pt.x, 0.0);
        assert_eq!(pt.y, 1.0);

        let pt = mat.transform_point(&Point2::new(500.0, 499.0));
        assert_eq!(pt.x, 0.0);
        assert_eq!(pt.y, -1.0);

        let pt = mat.transform_point(&Point2::new(0.0, 249.0));
        assert_eq!(pt.x, -2.0);
        assert_eq!(pt.y, 0.0);

        let pt = mat.transform_point(&Point2::new(1000.0, 249.0));
        assert_eq!(pt.x, 2.0);
        assert_eq!(pt.y, 0.0);
    }

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
