use crate::{
    eval::Function,
    render::{Camera, RegionSize, RenderMode},
    shape::{Shape, TileSizes},
    Error,
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimNameAdd, DimNameSub,
    DimNameSum, OMatrix, OPoint, OVector, U1,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Container to store render configuration (resolution, etc)
pub struct RenderConfig<const N: usize>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OVector<u32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>: Copy,
    OMatrix<f32, <Const<N> as DimNameAdd<Const<1>>>::Output, <Const<N> as DimNameAdd<Const<1>>>::Output>: Copy
{
    /// Image size (provides screen-to-world transform)
    pub image_size: RegionSize<N>,

    /// Camera (provides world-to-model transform)
    pub camera: Camera<N>,

    /// Tile sizes to use during evaluation.
    ///
    /// You'll likely want to use
    /// [`RenderHints::tile_sizes_2d`](crate::shape::RenderHints::tile_sizes_2d)
    /// or
    /// [`RenderHints::tile_sizes_3d`](crate::shape::RenderHints::tile_sizes_3d)
    /// to select this based on evaluator type.
    pub tile_sizes: TileSizes,

    /// Number of threads to use; 8 by default
    #[cfg(not(target_arch = "wasm32"))]
    pub threads: std::num::NonZeroUsize,
}

impl<const N: usize> Default for RenderConfig<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OVector<u32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>: Copy,
    OMatrix<f32, <Const<N> as DimNameAdd<Const<1>>>::Output, <Const<N> as DimNameAdd<Const<1>>>::Output>: Copy,
    <DefaultAllocator as Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>>::Buffer<u32>: std::marker::Copy,
{
    fn default() -> Self {
        Self {
            image_size: RegionSize::from(512),
            tile_sizes: match N {
                2 => TileSizes::new(&[128, 32, 8]).unwrap(),
                _ => TileSizes::new(&[128, 64, 32, 16, 8]).unwrap(),
            },
            camera: Camera::default(),

            #[cfg(not(target_arch = "wasm32"))]
            threads: std::num::NonZeroUsize::new(8).unwrap(),
        }
    }
}

impl<const N: usize> RenderConfig<N>
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    OVector<u32, <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>: Copy,
    OMatrix<f32, <Const<N> as DimNameAdd<Const<1>>>::Output, <Const<N> as DimNameAdd<Const<1>>>::Output>: Copy,
    <DefaultAllocator as Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>>::Buffer<u32>: Copy
{
    /// Returns the number of threads to use when rendering
    ///
    /// This is always 1 for WebAssembly builds
    pub fn threads(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        let out = 1;

        #[cfg(not(target_arch = "wasm32"))]
        let out = self.threads.get();

        out
    }

    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output
    > {
        self.camera.world_to_model() * self.image_size.screen_to_world()
    }

    /// Returns the data offset of a position within a subtile
    ///
    /// The position within the subtile is given by `x` and `y`, which are
    /// relative coordinates (in the range `0..self.tile_sizes[n]`).
    ///
    /// The root tile is assumed to be of size `self.tile_sizes[0]` and aligned.
    #[inline]
    pub(crate) fn tile_to_offset(&self, tile: Tile<N>, x: usize, y: usize) -> usize {
        // Find the relative position within the root tile
        let tx = tile.corner[0] % self.tile_sizes[0];
        let ty = tile.corner[1] % self.tile_sizes[0];

        // Apply the relative offset and find the data index
        tx + x + (ty + y) * self.tile_sizes[0]
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug)]
pub(crate) struct Tile<const N: usize> {
    /// Corner of this tile, in global screen (pixel) coordinates
    pub corner: OPoint<usize, Const<N>>,
}

impl<const N: usize> Tile<N> {
    /// Build a new tile from its global coordinates
    #[inline]
    pub(crate) fn new(corner: OPoint<usize, Const<N>>) -> Tile<N> {
        Tile { corner }
    }
}

/// Worker queue
pub(crate) struct Queue<const N: usize> {
    index: AtomicUsize,
    tiles: Vec<Tile<N>>,
}

impl<const N: usize> Queue<N> {
    pub fn new(tiles: Vec<Tile<N>>) -> Self {
        Self {
            index: AtomicUsize::new(0),
            tiles,
        }
    }
    pub fn next(&self) -> Option<Tile<N>> {
        let index = self.index.fetch_add(1, Ordering::Relaxed);
        self.tiles.get(index).cloned()
    }
}

impl RenderConfig<2> {
    /// High-level API for rendering shapes in 2D
    ///
    /// Under the hood, this delegates to
    /// [`fidget::render::render2d`](crate::render::render2d())
    pub fn run<F: Function, M: RenderMode + Sync>(
        &self,
        shape: Shape<F>,
    ) -> Result<Vec<<M as RenderMode>::Output>, Error> {
        Ok(crate::render::render2d::<F, M>(shape, self))
    }
}

impl RenderConfig<3> {
    /// High-level API for rendering shapes in 2D
    ///
    /// Under the hood, this delegates to
    /// [`fidget::render::render3d`](crate::render::render3d())
    ///
    /// Returns a tuple of heightmap, RGB image.
    pub fn run<F: Function>(
        &self,
        shape: Shape<F>,
    ) -> Result<(Vec<u32>, Vec<[u8; 3]>), Error> {
        Ok(crate::render::render3d::<F>(shape, self))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::render::ImageSize;
    use nalgebra::Point2;

    #[test]
    fn test_default_render_config() {
        let config = RenderConfig::<2> {
            image_size: ImageSize::from(512),
            ..Default::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(-1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 0.0)),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 512.0)),
            Point2::new(1.0, -1.0)
        );

        let config: RenderConfig<2> = RenderConfig {
            image_size: ImageSize::from(575),
            ..Default::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(-1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(
                config.image_size.width() as f32,
                0.0
            )),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(
                config.image_size.width() as f32,
                config.image_size.height() as f32,
            )),
            Point2::new(1.0, -1.0)
        );
    }

    #[test]
    fn test_camera_render_config() {
        let config = RenderConfig::<2> {
            image_size: ImageSize::from(512),
            camera: Camera::from_center_and_scale(
                nalgebra::Vector2::new(0.5, 0.5),
                0.5,
            ),
            ..RenderConfig::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(0.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 0.0)),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 512.0)),
            Point2::new(1.0, 0.0)
        );

        let config: RenderConfig<2> = RenderConfig {
            image_size: ImageSize::from(575),
            camera: Camera::from_center_and_scale(
                nalgebra::Vector2::new(0.5, 0.5),
                0.5,
            ),
            ..RenderConfig::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(0.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(
                config.image_size.width() as f32,
                0.0
            )),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(
                config.image_size.width() as f32,
                config.image_size.height() as f32,
            )),
            Point2::new(1.0, 0.0)
        );

        let config = RenderConfig::<2> {
            image_size: ImageSize::from(512),
            camera: Camera::from_center_and_scale(
                nalgebra::Vector2::new(0.5, 0.5),
                0.25,
            ),
            ..RenderConfig::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(0.25, 0.75)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 0.0)),
            Point2::new(0.75, 0.75)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 512.0)),
            Point2::new(0.75, 0.25)
        );
    }
}
