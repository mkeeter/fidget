use crate::{
    context::{Context, Node},
    eval::Family,
    render::RenderMode,
    Error,
};
use nalgebra::{
    allocator::Allocator, geometry::Transform, Const, DefaultAllocator,
    DimNameAdd, DimNameSub, DimNameSum, U1,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Container to store render configuration (resolution, etc)
pub struct RenderConfig<const N: usize>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
{
    /// Image size (for a square output image)
    pub image_size: usize,

    /// Tile sizes to use during evaluation.
    ///
    /// You'll likely want to use
    /// [`Family::tile_sizes_2d`] or [`Family::tile_sizes_3d`] to select this
    /// based on evaluator type.
    pub tile_sizes: Vec<usize>,

    /// Number of threads to use; 8 by default
    pub threads: usize,

    /// Transform matrix to apply to the input coordinates
    ///
    /// By default, we render a cube spanning ±1 on all axes; `mat` allows for
    /// rotation, scaling, transformation, and even perspective.
    pub mat: Transform<f32, nalgebra::TGeneral, N>,
}

impl<const N: usize> Default for RenderConfig<N>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
{
    fn default() -> Self {
        Self {
            image_size: 512,
            tile_sizes: match N {
                2 => vec![128, 32, 8],
                _ => vec![128, 64, 32, 16, 8],
            },
            threads: 8,
            mat: Transform::identity(),
        }
    }
}

impl<const N: usize> RenderConfig<N>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator:
        nalgebra::allocator::Allocator<
            f32,
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
        >,
    <nalgebra::Const<N> as DimNameAdd<nalgebra::Const<1>>>::Output:
        DimNameSub<nalgebra::Const<1>>,
{
    /// Returns a modified `RenderConfig` where `mat` is adjusted based on image
    /// size, and the image size is padded to an even multiple of `tile_size`.
    pub(crate) fn align(&self) -> AlignedRenderConfig<N> {
        let mut tile_sizes: Vec<usize> = self
            .tile_sizes
            .iter()
            .skip_while(|t| **t > self.image_size)
            .cloned()
            .collect();
        if tile_sizes.is_empty() {
            tile_sizes.push(8);
        }
        // Pad image size to an even multiple of tile size.
        let image_size = (self.image_size + tile_sizes[0] - 1) / tile_sizes[0]
            * tile_sizes[0];

        // Compensate for the image size change
        let scale = image_size as f32 / self.image_size as f32;

        // Look, I'm not any happier about this than you are.
        let v = nalgebra::Vector::<
            f32,
            <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                Const<1>,
            >>::Output,
            <DefaultAllocator as nalgebra::allocator::Allocator<
                f32,
                <<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<
                    Const<1>,
                >>::Output,
                U1,
            >>::Buffer,
        >::from_element(-1.0);
        let mat = self.mat.matrix()
            * nalgebra::Transform::<f32, nalgebra::TGeneral, N>::identity()
                .matrix()
                .append_scaling(2.0 / image_size as f32)
                .append_scaling(scale)
                .append_translation(&v);

        AlignedRenderConfig {
            image_size,
            orig_image_size: self.image_size,
            tile_sizes,
            threads: self.threads,
            mat,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub(crate) struct AlignedRenderConfig<const N: usize>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
{
    pub image_size: usize,
    pub orig_image_size: usize,

    pub tile_sizes: Vec<usize>,
    pub threads: usize,

    pub mat: NPlusOneMatrix<N>,
}

/// Type for a static `f32` matrix of size `N + 1`
type NPlusOneMatrix<const N: usize> = nalgebra::Matrix<
    f32,
    <Const<N> as DimNameAdd<Const<1>>>::Output,
    <Const<N> as DimNameAdd<Const<1>>>::Output,
    <DefaultAllocator as nalgebra::allocator::Allocator<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >>::Buffer,
>;

impl<const N: usize> AlignedRenderConfig<N>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
{
    #[inline]
    pub fn tile_to_offset(&self, tile: Tile<N>, x: usize, y: usize) -> usize {
        tile.offset + x + y * self.tile_sizes[0]
    }

    #[inline]
    pub fn new_tile(&self, corner: [usize; N]) -> Tile<N> {
        let x = corner[0] % self.tile_sizes[0];
        let y = corner[1] % self.tile_sizes[0];
        Tile {
            corner,
            offset: x + y * self.tile_sizes[0],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Tile<const N: usize> {
    pub corner: [usize; N],
    offset: usize,
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
    pub fn run<I: Family, M: RenderMode + Sync>(
        &self,
        root: Node,
        context: Context,
        mode: &M,
    ) -> Result<Vec<<M as RenderMode>::Output>, Error> {
        let tape = context.get_tape(root)?;
        Ok(crate::render::render2d::<I, M>(tape, self, mode))
    }
}

impl RenderConfig<3> {
    /// High-level API for rendering shapes in 2D
    ///
    /// Under the hood, this delegates to
    /// [`fidget::render::render3d`](crate::render::render3d())
    ///
    /// Returns a tuple of heightmap, RGB image.
    pub fn run<I: Family>(
        &self,
        root: Node,
        context: Context,
    ) -> Result<(Vec<u32>, Vec<[u8; 3]>), Error> {
        let tape = context.get_tape(root)?;
        Ok(crate::render::render3d::<I>(tape, self))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::Point2;

    #[test]
    fn test_aligned_config() {
        // Simple alignment
        let config: RenderConfig<2> = RenderConfig {
            image_size: 512,
            tile_sizes: vec![64, 32],
            threads: 8,
            mat: Transform::identity(),
        };
        let aligned = config.align();
        assert_eq!(aligned.image_size, config.image_size);
        assert_eq!(aligned.tile_sizes, config.tile_sizes);
        assert_eq!(aligned.threads, config.threads);
        assert_eq!(
            aligned.mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(-1.0, -1.0)
        );
        assert_eq!(
            aligned.mat.transform_point(&Point2::new(512.0, 0.0)),
            Point2::new(1.0, -1.0)
        );
        assert_eq!(
            aligned.mat.transform_point(&Point2::new(512.0, 512.0)),
            Point2::new(1.0, 1.0)
        );

        let config: RenderConfig<2> = RenderConfig {
            image_size: 575,
            tile_sizes: vec![64, 32],
            threads: 8,
            mat: Transform::identity(),
        };
        let aligned = config.align();
        assert_eq!(aligned.orig_image_size, 575);
        assert_eq!(aligned.image_size, 576);
        assert_eq!(aligned.tile_sizes, config.tile_sizes);
        assert_eq!(aligned.threads, config.threads);
        assert_eq!(
            aligned.mat.transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(-1.0, -1.0)
        );
        assert_eq!(
            aligned
                .mat
                .transform_point(&Point2::new(config.image_size as f32, 0.0)),
            Point2::new(1.0, -1.0)
        );
        assert_eq!(
            aligned.mat.transform_point(&Point2::new(
                config.image_size as f32,
                config.image_size as f32
            )),
            Point2::new(1.0, 1.0)
        );
    }
}
