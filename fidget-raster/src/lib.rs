//! 2D and 3D rendering
//!
//! To render something, build a configuration object then call its `run`
//! function, e.g. [`ImageRenderConfig::run`] and [`VoxelRenderConfig::run`].
use crate::config::Tile;
use fidget_core::{
    eval::Function,
    render::{ImageSize, RenderHandle, ThreadPool, TileSizes, VoxelSize},
    shape::{Shape, ShapeVars},
};
use nalgebra::Point2;
use rayon::prelude::*;
use zerocopy::{FromBytes, Immutable, IntoBytes};

mod config;
mod render2d;
mod render3d;

pub mod effects;
pub use config::{ImageRenderConfig, VoxelRenderConfig};
pub use render2d::DistancePixel;

use render2d::render as render2d;
use render3d::render as render3d;

/// Helper struct to borrow from [`TileSizes`]
///
/// This object has the same guarantees as `TileSizes`, but trims items off the
/// front of the `Vec<usize>` based on the image size.
pub(crate) struct TileSizesRef<'a>(&'a [usize]);

impl<'a> std::ops::Index<usize> for TileSizesRef<'a> {
    type Output = usize;

    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl TileSizesRef<'_> {
    /// Builds a new `TileSizesRef` based on the maximum tile size
    fn new(tiles: &TileSizes, max_size: usize) -> TileSizesRef<'_> {
        let i = tiles
            .iter()
            .position(|t| *t < max_size)
            .unwrap_or(tiles.len())
            .saturating_sub(1);
        TileSizesRef(&tiles[i..])
    }

    /// Returns the last (smallest) tile size
    pub fn last(&self) -> usize {
        *self.0.last().unwrap()
    }

    /// Gets a tile size by index
    pub fn get(&self, i: usize) -> Option<usize> {
        self.0.get(i).copied()
    }

    /// Returns the data offset of a global pixel position within a root tile
    ///
    /// The root tile is implicit: it's set by the largest tile size and aligned
    /// to multiples of that size.
    #[inline]
    pub(crate) fn pixel_offset(&self, pos: Point2<usize>) -> usize {
        // Find the relative position within the root tile
        let x = pos.x % self.0[0];
        let y = pos.y % self.0[0];

        // Apply the relative offset and find the data index
        x + y * self.0[0]
    }
}

/// Grand unified render function
///
/// This handles tile generation and building + calling render workers in
/// parallel (using [`rayon`] for parallelism at the tile level).
///
/// It returns a set of output tiles, or `None` if rendering has been cancelled
pub(crate) fn render_tiles<'a, F: Function, W: RenderWorker<'a, F, T>, T>(
    shape: Shape<F, T>,
    vars: &ShapeVars<f32>,
    config: &'a W::Config,
) -> Option<Vec<(Tile<2>, W::Output)>>
where
    W::Config: Send + Sync,
    T: Sync,
{
    use rayon::prelude::*;

    let tile_sizes = config.tile_sizes();

    let mut tiles = vec![];
    let t = tile_sizes[0];
    let width = config.width() as usize;
    let height = config.height() as usize;
    for i in 0..width.div_ceil(t) {
        for j in 0..height.div_ceil(t) {
            tiles.push(Tile::new(Point2::new(
                i * tile_sizes[0],
                j * tile_sizes[0],
            )));
        }
    }

    let mut rh = RenderHandle::new(shape);

    let _ = rh.i_tape(&mut vec![]); // populate i_tape before cloning
    let init = || {
        let rh = rh.clone();
        let worker = W::new(config);
        (worker, rh)
    };

    match config.threads() {
        None => {
            let mut worker = W::new(config);
            tiles
                .into_iter()
                .map(|tile| {
                    if config.is_cancelled() {
                        Err(())
                    } else {
                        let pixels = worker.render_tile(&mut rh, vars, tile);
                        Ok((tile, pixels))
                    }
                })
                .collect::<Result<Vec<_>, ()>>()
                .ok()
        }

        Some(p) => p.run(|| {
            tiles
                .into_par_iter()
                .map_init(init, |(w, rh), tile| {
                    if config.is_cancelled() {
                        Err(())
                    } else {
                        let pixels = w.render_tile(rh, vars, tile);
                        Ok((tile, pixels))
                    }
                })
                .collect::<Result<Vec<_>, ()>>()
                .ok()
        }),
    }
}

/// Helper trait for tiled rendering configuration
pub(crate) trait RenderConfig {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn tile_sizes(&self) -> TileSizesRef<'_>;
    fn threads(&self) -> Option<&ThreadPool>;
    fn is_cancelled(&self) -> bool;
}

/// Helper trait for a tiled renderer worker
pub(crate) trait RenderWorker<'a, F: Function, T> {
    type Config: RenderConfig;
    type Output: Send;

    /// Build a new worker
    ///
    /// Workers are typically built on a per-thread basis
    fn new(cfg: &'a Self::Config) -> Self;

    /// Render a single tile, returning a worker-dependent output
    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        tile: config::Tile<2>,
    ) -> Self::Output;
}

/// Generic image type
///
/// The image is laid out in row-major order, and can be indexed either by a
/// `usize` index or a `(row, column)` tuple.
///
/// ```text
///        0 ------------> width (columns)
///        |             |
///        |             |
///        |             |
///        V--------------
///   height (rows)
/// ```
#[derive(Clone)]
pub struct Image<P, S = ImageSize> {
    data: Vec<P>,
    size: S,
}

/// Helper trait to make images generic across [`ImageSize`] and [`VoxelSize`]
pub trait ImageSizeLike {
    /// Returns the width of the region, in pixels / voxels
    fn width(&self) -> u32;
    /// Returns the height of the region, in pixels / voxels
    fn height(&self) -> u32;
}

impl ImageSizeLike for ImageSize {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
}

impl ImageSizeLike for VoxelSize {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
}

impl<P: Send, S: ImageSizeLike + Sync> Image<P, S> {
    /// Generates an image by computing a per-pixel function
    ///
    /// This should be called on the _output_ image; the closure takes `(x, y)`
    /// tuples and is expected to capture one or more source images.
    pub fn apply_effect<F: Fn(usize, usize) -> P + Send + Sync>(
        &mut self,
        f: F,
        threads: Option<&ThreadPool>,
    ) {
        let r = |(y, row): (usize, &mut [P])| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = f(x, y);
            }
        };

        if let Some(threads) = threads {
            threads.run(|| {
                self.data
                    .par_chunks_mut(self.size.width() as usize)
                    .enumerate()
                    .for_each(r)
            })
        } else {
            self.data
                .chunks_mut(self.size.width() as usize)
                .enumerate()
                .for_each(r)
        }
    }
}

impl<P, S: Default> Default for Image<P, S> {
    fn default() -> Self {
        Image {
            data: vec![],
            size: S::default(),
        }
    }
}

impl<P: Default + Clone, S: ImageSizeLike> Image<P, S> {
    /// Builds a new image filled with `P::default()`
    pub fn new(size: S) -> Self {
        Self {
            data: vec![
                P::default();
                size.width() as usize * size.height() as usize
            ],
            size,
        }
    }
}

impl<P, S: Clone> Image<P, S> {
    /// Returns the image size
    pub fn size(&self) -> S {
        self.size.clone()
    }

    /// Generates an image by mapping a simple function over each pixel
    pub fn map<T, F: Fn(&P) -> T>(&self, f: F) -> Image<T, S> {
        let data = self.data.iter().map(f).collect();
        Image {
            data,
            size: self.size.clone(),
        }
    }

    /// Decomposes the image into its components
    pub fn take(self) -> (Vec<P>, S) {
        (self.data, self.size)
    }
}

impl<P, S: ImageSizeLike> Image<P, S> {
    /// Returns the image width
    pub fn width(&self) -> usize {
        self.size.width() as usize
    }

    /// Returns the image height
    pub fn height(&self) -> usize {
        self.size.height() as usize
    }

    /// Checks a `(row, column)` position
    ///
    /// Returns the input position in the 1D array if valid; panics otherwise
    fn decode_position(&self, pos: (usize, usize)) -> usize {
        let (row, col) = pos;
        assert!(
            row < self.height(),
            "row ({row}) must be less than image height ({})",
            self.height()
        );
        assert!(
            col < self.width(),
            "column ({col}) must be less than image width ({})",
            self.width()
        );
        row * self.width() + col
    }
}

impl<P, S> Image<P, S> {
    /// Iterates over pixel values
    pub fn iter(&self) -> impl Iterator<Item = &P> + '_ {
        self.data.iter()
    }

    /// Returns the number of pixels in the image
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks whether the image is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, P: 'a, S> IntoIterator for &'a Image<P, S> {
    type Item = &'a P;
    type IntoIter = std::slice::Iter<'a, P>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<P, S> IntoIterator for Image<P, S> {
    type Item = P;
    type IntoIter = std::vec::IntoIter<P>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<P, S> std::ops::Index<usize> for Image<P, S> {
    type Output = P;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<P, S> std::ops::IndexMut<usize> for Image<P, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

macro_rules! define_image_index {
    ($ty:ty) => {
        impl<P, S> std::ops::Index<$ty> for Image<P, S> {
            type Output = [P];
            fn index(&self, index: $ty) -> &Self::Output {
                &self.data[index]
            }
        }

        impl<P, S> std::ops::IndexMut<$ty> for Image<P, S> {
            fn index_mut(&mut self, index: $ty) -> &mut Self::Output {
                &mut self.data[index]
            }
        }
    };
}

define_image_index!(std::ops::Range<usize>);
define_image_index!(std::ops::RangeTo<usize>);
define_image_index!(std::ops::RangeFrom<usize>);
define_image_index!(std::ops::RangeInclusive<usize>);
define_image_index!(std::ops::RangeToInclusive<usize>);
define_image_index!(std::ops::RangeFull);

/// Indexes an image with `(row, col)`
impl<P, S: ImageSizeLike> std::ops::Index<(usize, usize)> for Image<P, S> {
    type Output = P;
    fn index(&self, pos: (usize, usize)) -> &Self::Output {
        let index = self.decode_position(pos);
        &self.data[index]
    }
}

impl<P, S: ImageSizeLike> std::ops::IndexMut<(usize, usize)> for Image<P, S> {
    fn index_mut(&mut self, pos: (usize, usize)) -> &mut Self::Output {
        let index = self.decode_position(pos);
        &mut self.data[index]
    }
}

/// Pixel type for a [`GeometryBuffer`]
///
/// This type can be passed directly in a buffer to the GPU.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, IntoBytes, FromBytes, Immutable)]
pub struct GeometryPixel {
    /// Z position of this pixel, in voxel units
    pub depth: u32, // TODO should this be `f32`?
    /// Function gradients at this pixel
    pub normal: [f32; 3],
}

impl GeometryPixel {
    /// Converts the normal into a normalized RGB value
    pub fn to_color(&self) -> [u8; 3] {
        let [dx, dy, dz] = self.normal;
        let s = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        if s != 0.0 {
            let scale = u8::MAX as f32 / s;
            [
                (dx.abs() * scale) as u8,
                (dy.abs() * scale) as u8,
                (dz.abs() * scale) as u8,
            ]
        } else {
            [0; 3]
        }
    }
}

/// Image containing depth and normal at each pixel
pub type GeometryBuffer = Image<GeometryPixel, VoxelSize>;

impl<P: Default + Copy + Clone> Image<P, VoxelSize> {
    /// Returns the image depth in voxels
    pub fn depth(&self) -> usize {
        self.size.depth() as usize
    }
}

/// Three-channel color image
pub type ColorImage = Image<[u8; 3]>;
