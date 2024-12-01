use crate::{
    eval::Function,
    render::{ImageSize, RenderMode, TileSizes, View2, View3, VoxelSize},
    shape::{Shape, ShapeVars},
};
use nalgebra::{Const, Matrix3, Matrix4, OPoint, Point2, Vector2};

/// Number of threads to use during evaluation
///
/// In a WebAssembly build, only the [`ThreadCount::One`] variant is available.
#[derive(Copy, Clone, Debug)]
pub enum ThreadCount {
    /// Perform all evaluation in the main thread, not spawning any workers
    One,

    /// Spawn some number of worker threads for evaluation
    ///
    /// This can be set to `1`, in which case a single worker thread will be
    /// spawned; this is different from doing work in the main thread, but not
    /// particularly useful!
    #[cfg(not(target_arch = "wasm32"))]
    Many(std::num::NonZeroUsize),
}

#[cfg(not(target_arch = "wasm32"))]
impl From<std::num::NonZeroUsize> for ThreadCount {
    fn from(v: std::num::NonZeroUsize) -> Self {
        match v.get() {
            0 => unreachable!(),
            1 => ThreadCount::One,
            _ => ThreadCount::Many(v),
        }
    }
}

/// Single-threaded mode is shown as `-`; otherwise, an integer
impl std::fmt::Display for ThreadCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadCount::One => write!(f, "-"),
            #[cfg(not(target_arch = "wasm32"))]
            ThreadCount::Many(n) => write!(f, "{n}"),
        }
    }
}

impl ThreadCount {
    /// Gets the thread count
    ///
    /// Returns `None` if we are required to be single-threaded
    pub fn get(&self) -> Option<usize> {
        match self {
            ThreadCount::One => None,
            #[cfg(not(target_arch = "wasm32"))]
            ThreadCount::Many(v) => Some(v.get()),
        }
    }
}

impl Default for ThreadCount {
    #[cfg(target_arch = "wasm32")]
    fn default() -> Self {
        Self::One
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn default() -> Self {
        Self::Many(std::num::NonZeroUsize::new(8).unwrap())
    }
}

/// Thread pool to use for multithreaded rendering
///
/// Most users will use the global Rayon pool, but it's possible to provide your
/// own as well.
#[derive(Clone)]
pub enum ThreadPool<'a> {
    /// User-provided pool
    Custom(&'a rayon::ThreadPool),
    /// Global Rayon pool
    Global,
}

/// Settings for 2D rendering
pub struct ImageRenderConfig<'a> {
    /// Render size
    pub image_size: ImageSize,

    /// World-to-model transform
    pub view: View2,

    /// Tile sizes to use during evaluation.
    ///
    /// You'll likely want to use
    /// [`RenderHints::tile_sizes_2d`](crate::render::RenderHints::tile_sizes_2d)
    /// to select this based on evaluator type.
    pub tile_sizes: TileSizes,

    /// Number of worker threads
    pub threads: Option<ThreadPool<'a>>,
}

impl Default for ImageRenderConfig<'_> {
    fn default() -> Self {
        Self {
            image_size: ImageSize::from(512),
            tile_sizes: TileSizes::new(&[128, 32, 8]).unwrap(),
            view: View2::default(),
            threads: Some(ThreadPool::Global),
        }
    }
}

impl ImageRenderConfig<'_> {
    /// Render a shape in 2D using this configuration
    pub fn run<F: Function, M: RenderMode + Sync>(
        &self,
        shape: Shape<F>,
    ) -> Vec<<M as RenderMode>::Output> {
        self.run_with_vars::<F, M>(shape, &ShapeVars::new())
    }

    /// Render a shape in 2D using this configuration and variables
    pub fn run_with_vars<F: Function, M: RenderMode + Sync>(
        &self,
        shape: Shape<F>,
        vars: &ShapeVars<f32>,
    ) -> Vec<<M as RenderMode>::Output> {
        crate::render::render2d::<F, M>(shape, vars, self)
    }

    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> Matrix3<f32> {
        self.view.world_to_model() * self.image_size.screen_to_world()
    }
}

/// Settings for 3D rendering
pub struct VoxelRenderConfig<'a> {
    /// Render size
    ///
    /// The resulting image will have the given width and height; depth sets the
    /// number of voxels to evaluate within each pixel of the image (stacked
    /// into a column going into the screen).
    pub image_size: VoxelSize,

    /// World-to-model transform
    pub view: View3,

    /// Tile sizes to use during evaluation.
    ///
    /// You'll likely want to use
    /// [`RenderHints::tile_sizes_3d`](crate::render::RenderHints::tile_sizes_3d)
    /// to select this based on evaluator type.
    pub tile_sizes: TileSizes,

    /// Number of worker threads
    pub threads: Option<ThreadPool<'a>>,
}

impl Default for VoxelRenderConfig<'_> {
    fn default() -> Self {
        Self {
            image_size: VoxelSize::from(512),
            tile_sizes: TileSizes::new(&[128, 64, 32, 16, 8]).unwrap(),
            view: View3::default(),

            threads: Some(ThreadPool::Global),
        }
    }
}

impl VoxelRenderConfig<'_> {
    /// Render a shape in 3D using this configuration
    ///
    /// Returns a tuple of heightmap, RGB image.
    pub fn run<F: Function>(
        &self,
        shape: Shape<F>,
    ) -> (Vec<u32>, Vec<[u8; 3]>) {
        self.run_with_vars::<F>(shape, &ShapeVars::new())
    }

    /// Render a shape in 2D using this configuration and variables
    pub fn run_with_vars<F: Function>(
        &self,
        shape: Shape<F>,
        vars: &ShapeVars<f32>,
    ) -> (Vec<u32>, Vec<[u8; 3]>) {
        crate::render::render3d::<F>(shape, vars, self)
    }

    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> Matrix4<f32> {
        self.view.world_to_model() * self.image_size.screen_to_world()
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

    /// Converts a relative position within the tile into a global position
    ///
    /// This function operates in pixel space, using the `.xy` coordinates
    pub(crate) fn add(&self, pos: Vector2<usize>) -> Point2<usize> {
        let corner = Point2::new(self.corner[0], self.corner[1]);
        corner + pos
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
        let config = ImageRenderConfig {
            image_size: ImageSize::from(512),
            ..Default::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, -1.0)),
            Point2::new(-1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, -1.0)),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 511.0)),
            Point2::new(1.0, -1.0)
        );

        let config = ImageRenderConfig {
            image_size: ImageSize::from(575),
            ..Default::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, -1.0)),
            Point2::new(-1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(
                config.image_size.width() as f32,
                -1.0
            )),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(
                config.image_size.width() as f32,
                config.image_size.height() as f32 - 1.0,
            )),
            Point2::new(1.0, -1.0)
        );
    }

    #[test]
    fn test_camera_render_config() {
        let config = ImageRenderConfig {
            image_size: ImageSize::from(512),
            view: View2::from_center_and_scale(
                nalgebra::Vector2::new(0.5, 0.5),
                0.5,
            ),
            ..Default::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, -1.0)),
            Point2::new(0.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, -1.0)),
            Point2::new(1.0, 1.0)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 511.0)),
            Point2::new(1.0, 0.0)
        );

        let config = ImageRenderConfig {
            image_size: ImageSize::from(512),
            view: View2::from_center_and_scale(
                nalgebra::Vector2::new(0.5, 0.5),
                0.25,
            ),
            ..Default::default()
        };
        let mat = config.mat();
        assert_eq!(
            mat.transform_point(&Point2::new(0.0, -1.0)),
            Point2::new(0.25, 0.75)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, -1.0)),
            Point2::new(0.75, 0.75)
        );
        assert_eq!(
            mat.transform_point(&Point2::new(512.0, 511.0)),
            Point2::new(0.75, 0.25)
        );
    }
}
