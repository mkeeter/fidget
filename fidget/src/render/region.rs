use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimNameAdd, DimNameSub,
    DimNameSum, OMatrix, OVector, Vector2, Vector3, U1,
};

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
    use nalgebra::Point2;

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
}
