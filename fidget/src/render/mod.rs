//! 2D and 3D rendering
//!
//! To render something, build a configuration object then call its `run`
//! function, e.g. [`ImageRenderConfig::run`] and [`VoxelRenderConfig::run`].
use crate::{
    eval::{BulkEvaluator, Function, Trace, TracingEvaluator},
    shape::{Shape, ShapeTape, ShapeVars},
    Error,
};
use nalgebra::Point2;
use rayon::prelude::*;

pub mod effects;

mod config;
mod region;
mod render2d;
mod render3d;
mod view;

use config::Tile;
pub use config::{
    CancelToken, ImageRenderConfig, ThreadPool, VoxelRenderConfig,
};
pub use region::{ImageSize, RegionSize, VoxelSize};
pub use view::{RotateHandle, TranslateHandle, View2, View3};

use render2d::render as render2d;
use render3d::render as render3d;

pub use render2d::{
    BitRenderMode, DebugRenderMode, RenderMode, SdfPixelRenderMode,
    SdfRenderMode,
};

/// A `RenderHandle` contains lazily-populated tapes for rendering
///
/// This can be cheaply cloned, although it is usually passed by mutable
/// reference to a recursive function.
///
/// The most recent simplification is cached for reuse (if the trace matches).
pub struct RenderHandle<F: Function> {
    shape: Shape<F>,

    i_tape: Option<ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape>>,
    f_tape: Option<ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape>>,
    g_tape: Option<ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape>>,

    next: Option<(F::Trace, Box<Self>)>,
}

impl<F: Function> RenderHandle<F> {
    /// Build a new [`RenderHandle`] for the given shape
    ///
    /// None of the tapes are populated here.
    pub fn new(shape: Shape<F>) -> Self {
        Self {
            shape,
            i_tape: None,
            f_tape: None,
            g_tape: None,
            next: None,
        }
    }

    /// Returns a tape for tracing interval evaluation
    pub fn i_tape(
        &mut self,
        storage: &mut Vec<F::TapeStorage>,
    ) -> &ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape> {
        self.i_tape.get_or_insert_with(|| {
            self.shape.interval_tape(storage.pop().unwrap_or_default())
        })
    }

    /// Returns a tape for bulk float evaluation
    pub fn f_tape(
        &mut self,
        storage: &mut Vec<F::TapeStorage>,
    ) -> &ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape> {
        self.f_tape.get_or_insert_with(|| {
            self.shape
                .float_slice_tape(storage.pop().unwrap_or_default())
        })
    }

    /// Returns a tape for bulk gradient evaluation
    pub fn g_tape(
        &mut self,
        storage: &mut Vec<F::TapeStorage>,
    ) -> &ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape> {
        self.g_tape.get_or_insert_with(|| {
            self.shape
                .grad_slice_tape(storage.pop().unwrap_or_default())
        })
    }

    /// Simplifies the shape with the given trace
    ///
    /// As an internal optimization, this may reuse a previous simplification if
    /// the trace matches.
    pub fn simplify(
        &mut self,
        trace: &F::Trace,
        workspace: &mut F::Workspace,
        shape_storage: &mut Vec<F::Storage>,
        tape_storage: &mut Vec<F::TapeStorage>,
    ) -> &mut Self {
        // Free self.next if it doesn't match our new set of choices
        let mut trace_storage = if let Some(neighbor) = &self.next {
            if &neighbor.0 != trace {
                let (trace, neighbor) = self.next.take().unwrap();
                neighbor.recycle(shape_storage, tape_storage);
                Some(trace)
                // continue with simplification
            } else {
                None
            }
        } else {
            None
        };

        // Ordering is a little weird here, to persuade the borrow checker to be
        // happy about things.  At this point, `next` is empty if we can't reuse
        // it, and `Some(..)` if we can.
        if self.next.is_none() {
            let s = shape_storage.pop().unwrap_or_default();
            let next = self.shape.simplify(trace, s, workspace).unwrap();
            if next.size() >= self.shape.size() {
                // Optimization: if the simplified shape isn't any shorter, then
                // don't use it (this saves time spent generating tapes)
                shape_storage.extend(next.recycle());
                self
            } else {
                assert!(self.next.is_none());
                if let Some(t) = trace_storage.as_mut() {
                    t.copy_from(trace);
                } else {
                    trace_storage = Some(trace.clone());
                }
                self.next = Some((
                    trace_storage.unwrap(),
                    Box::new(RenderHandle {
                        shape: next,
                        i_tape: None,
                        f_tape: None,
                        g_tape: None,
                        next: None,
                    }),
                ));
                &mut self.next.as_mut().unwrap().1
            }
        } else {
            &mut self.next.as_mut().unwrap().1
        }
    }

    /// Recycles the entire handle into the given storage vectors
    fn recycle(
        mut self,
        shape_storage: &mut Vec<F::Storage>,
        tape_storage: &mut Vec<F::TapeStorage>,
    ) {
        // Recycle the child first, in case it borrowed from us
        if let Some((_trace, shape)) = self.next.take() {
            shape.recycle(shape_storage, tape_storage);
        }

        if let Some(i_tape) = self.i_tape.take() {
            tape_storage.extend(i_tape.recycle());
        }
        if let Some(g_tape) = self.g_tape.take() {
            tape_storage.extend(g_tape.recycle());
        }
        if let Some(f_tape) = self.f_tape.take() {
            tape_storage.extend(f_tape.recycle());
        }

        // Do this step last because the evaluators may borrow the shape
        shape_storage.extend(self.shape.recycle());
    }
}

/// Container representing an ordered, checked list of tile sizes
///
/// This object wraps a `Vec<usize>`, guaranteeing three invariants:
///
/// - There must be at least one tile size
/// - Tiles must be ordered from largest to smallest
/// - Each tile size must be exactly divisible by subsequent tile sizes
#[derive(Debug, Eq, PartialEq)]
pub struct TileSizes(Vec<usize>);

impl std::ops::Index<usize> for TileSizes {
    type Output = usize;

    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl TileSizes {
    /// Builds a new tile size list, checking invariants
    pub fn new(sizes: &[usize]) -> Result<Self, Error> {
        if sizes.is_empty() {
            return Err(Error::EmptyTileSizes);
        }
        for i in 1..sizes.len() {
            if sizes[i - 1] <= sizes[i] {
                return Err(Error::BadTileOrder(sizes[i - 1], sizes[i]));
            } else if sizes[i - 1] % sizes[i] != 0 {
                return Err(Error::BadTileSize(sizes[i - 1], sizes[i]));
            }
        }
        Ok(Self(sizes.to_vec()))
    }
    /// Returns the length of the tile list
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over tile sizes (largest to smallest)
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.0.iter()
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
    pub(crate) fn pixel_offset(&self, pos: nalgebra::Point2<usize>) -> usize {
        // Find the relative position within the root tile
        let x = pos.x % self.0[0];
        let y = pos.y % self.0[0];

        // Apply the relative offset and find the data index
        x + y * self.0[0]
    }
}

/// Hints for how to render this particular type
pub trait RenderHints {
    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> TileSizes;

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> TileSizes;

    /// Indicates whether we run tape simplification at the given cell depth
    /// during meshing.
    ///
    /// By default, this is always true; for evaluators where simplification is
    /// more expensive than evaluation (i.e. the JIT), it may only be true at
    /// certain depths.
    fn simplify_tree_during_meshing(_d: usize) -> bool {
        true
    }
}

/// Grand unified render function
///
/// This handles tile generation and building + calling render workers in
/// parallel (using [`rayon`] for parallelism at the tile level).
///
/// It returns a set of output tiles, or `None` if rendering has been cancelled
pub(crate) fn render_tiles<'a, F: Function, W: RenderWorker<'a, F>>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &'a W::Config,
) -> Option<Vec<(Tile<2>, W::Output)>>
where
    W::Config: Send + Sync,
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

    // Precompute i_tape so that we only need to do it once
    let mut rh = RenderHandle::new(shape.clone());
    let i_tape = rh.i_tape(&mut vec![]).clone();
    let init = || {
        let mut rh = RenderHandle::new(shape.clone());
        rh.i_tape = Some(i_tape.clone());
        let worker = W::new(config);
        (worker, rh)
    };

    let out = match config.threads() {
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
    };

    out
}

/// Helper trait for tiled rendering configuration
pub(crate) trait RenderConfig {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn tile_sizes(&self) -> &TileSizes;
    fn threads(&self) -> Option<&ThreadPool>;
    fn is_cancelled(&self) -> bool;
}

/// Helper trait for a tiled renderer worker
pub(crate) trait RenderWorker<'a, F: Function> {
    type Config: RenderConfig;
    type Output: Send;

    /// Build a new worker
    ///
    /// Workers are typically built on a per-thread basis
    fn new(cfg: &'a Self::Config) -> Self;

    /// Render a single tile, returning a worker-dependent output
    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F>,
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
pub struct Image<P> {
    data: Vec<P>,
    width: usize, // XXX use ImageSize instead?
    height: usize,
}

impl<P: Default + Clone> Image<P> {
    /// Builds a new image filled with `P::default()`
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![P::default(); width * height],
            width,
            height,
        }
    }
}

impl<P: Send> Image<P> {
    /// Generates an image by computing a per-pixel function
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
                self.data.par_chunks_mut(self.width).enumerate().for_each(r)
            })
        } else {
            self.data.chunks_mut(self.width).enumerate().for_each(r)
        }
    }
}

impl<P> Default for Image<P> {
    fn default() -> Self {
        Image {
            data: vec![],
            width: 0,
            height: 0,
        }
    }
}

impl<P> Image<P> {
    /// Returns the image width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height
    pub fn height(&self) -> usize {
        self.height
    }

    /// Checks a `(row, column)` position
    ///
    /// Returns the input position in the 1D array if valid; panics otherwise
    fn decode_position(&self, pos: (usize, usize)) -> usize {
        let (row, col) = pos;
        assert!(
            row < self.height,
            "row ({row}) must be less than image height ({})",
            self.height
        );
        assert!(
            col < self.width,
            "column ({row}) must be less than image width ({})",
            self.width
        );
        row * self.width + col
    }

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

impl<'a, P: 'a> IntoIterator for &'a Image<P> {
    type Item = &'a P;
    type IntoIter = std::slice::Iter<'a, P>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<P> IntoIterator for Image<P> {
    type Item = P;
    type IntoIter = std::vec::IntoIter<P>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<P> std::ops::Index<usize> for Image<P> {
    type Output = P;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<P> std::ops::IndexMut<usize> for Image<P> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

macro_rules! define_image_index {
    ($ty:ty) => {
        impl<P> std::ops::Index<$ty> for Image<P> {
            type Output = [P];
            fn index(&self, index: $ty) -> &Self::Output {
                &self.data[index]
            }
        }

        impl<P> std::ops::IndexMut<$ty> for Image<P> {
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
impl<P> std::ops::Index<(usize, usize)> for Image<P> {
    type Output = P;
    fn index(&self, pos: (usize, usize)) -> &Self::Output {
        let index = self.decode_position(pos);
        &self.data[index]
    }
}

impl<P> std::ops::IndexMut<(usize, usize)> for Image<P> {
    fn index_mut(&mut self, pos: (usize, usize)) -> &mut Self::Output {
        let index = self.decode_position(pos);
        &mut self.data[index]
    }
}

/// Single-channel depth image
pub type DepthImage = Image<u32>;

/// Three-channel normal image
pub type NormalImage = Image<[f32; 3]>;

impl NormalImage {
    /// Converts from floating-point normals to RGB colors
    pub fn to_color(&self) -> ColorImage {
        let mut data = Vec::with_capacity(self.width * self.height);
        for [dx, dy, dz] in self.data.iter() {
            let s = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
            let rgb = if s != 0.0 {
                let scale = u8::MAX as f32 / s;
                [
                    (dx.abs() * scale) as u8,
                    (dy.abs() * scale) as u8,
                    (dz.abs() * scale) as u8,
                ]
            } else {
                [0; 3]
            };
            data.push(rgb);
        }
        ColorImage {
            data,
            width: self.width,
            height: self.height,
        }
    }
}

/// Three-channel color image
pub type ColorImage = Image<[u8; 3]>;
