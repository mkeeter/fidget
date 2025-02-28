use crate::{
    eval::Function,
    render::{
        DepthImage, Image, ImageSize, NormalImage, RenderConfig, RenderMode,
        TileSizes, View2, View3, VoxelSize,
    },
    shape::{Shape, ShapeVars},
};
use nalgebra::{Const, Matrix3, Matrix4, OPoint, Point2, Vector2};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

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

impl ThreadPool<'_> {
    /// Runs a function across the thread pool
    pub fn run<F: FnOnce() -> V + Send, V: Send>(&self, f: F) -> V {
        match self {
            ThreadPool::Custom(p) => p.install(f),
            ThreadPool::Global => f(),
        }
    }
}

/// Token to cancel an in-progress operation
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Build a new token, which is initialize as "not cancelled"
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark this token as cancelled
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Check if the token is cancelled
    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Returns a raw pointer to the inner flag
    ///
    /// This is used in shared memory environments where the `CancelToken`
    /// itself cannot be passed between threads, i.e. to send a cancel token to
    /// a web worker.
    ///
    /// To avoid a memory leak, the pointer must be converted back to a
    /// `CancelToken` using [`CancelToken::from_raw`].  In the meantime, users
    /// should refrain from writing to the raw pointer.
    #[doc(hidden)]
    pub fn into_raw(self) -> *const AtomicBool {
        Arc::into_raw(self.0)
    }

    /// Reclaims a released cancel token pointer
    ///
    /// # Safety
    /// The pointer must have been previously returned by a call to
    /// [`CancelToken::into_raw`].
    #[doc(hidden)]
    pub unsafe fn from_raw(ptr: *const AtomicBool) -> Self {
        Self(Arc::from_raw(ptr))
    }
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

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<ThreadPool<'a>>,

    /// Token to cancel rendering
    pub cancel: CancelToken,
}

impl Default for ImageRenderConfig<'_> {
    fn default() -> Self {
        Self {
            image_size: ImageSize::from(512),
            tile_sizes: TileSizes::new(&[128, 32, 8]).unwrap(),
            view: View2::default(),
            threads: Some(ThreadPool::Global),
            cancel: CancelToken::new(),
        }
    }
}

impl RenderConfig for ImageRenderConfig<'_> {
    fn width(&self) -> u32 {
        self.image_size.width()
    }
    fn height(&self) -> u32 {
        self.image_size.height()
    }
    fn threads(&self) -> Option<&ThreadPool> {
        self.threads.as_ref()
    }
    fn tile_sizes(&self) -> &TileSizes {
        &self.tile_sizes
    }
    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

impl ImageRenderConfig<'_> {
    /// Render a shape in 2D using this configuration
    pub fn run<F: Function, M: RenderMode + Sync>(
        &self,
        shape: Shape<F>,
    ) -> Option<Image<<M as RenderMode>::Output>> {
        self.run_with_vars::<F, M>(shape, &ShapeVars::new())
    }

    /// Render a shape in 2D using this configuration and variables
    pub fn run_with_vars<F: Function, M: RenderMode + Sync>(
        &self,
        shape: Shape<F>,
        vars: &ShapeVars<f32>,
    ) -> Option<Image<<M as RenderMode>::Output>> {
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

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<ThreadPool<'a>>,

    /// Token to cancel rendering
    pub cancel: CancelToken,
}

impl Default for VoxelRenderConfig<'_> {
    fn default() -> Self {
        Self {
            image_size: VoxelSize::from(512),
            tile_sizes: TileSizes::new(&[128, 64, 32, 16, 8]).unwrap(),
            view: View3::default(),

            threads: Some(ThreadPool::Global),
            cancel: CancelToken::new(),
        }
    }
}

impl RenderConfig for VoxelRenderConfig<'_> {
    fn width(&self) -> u32 {
        self.image_size.width()
    }
    fn height(&self) -> u32 {
        self.image_size.height()
    }
    fn threads(&self) -> Option<&ThreadPool> {
        self.threads.as_ref()
    }
    fn tile_sizes(&self) -> &TileSizes {
        &self.tile_sizes
    }
    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

impl VoxelRenderConfig<'_> {
    /// Render a shape in 3D using this configuration
    ///
    /// Returns a tuple of `(heightmap, RGB image)` or `None` if rendering was
    /// cancelled.
    pub fn run<F: Function>(
        &self,
        shape: Shape<F>,
    ) -> Option<(DepthImage, NormalImage)> {
        self.run_with_vars::<F>(shape, &ShapeVars::new())
    }

    /// Render a shape in 3D using this configuration and variables
    pub fn run_with_vars<F: Function>(
        &self,
        shape: Shape<F>,
        vars: &ShapeVars<f32>,
    ) -> Option<(DepthImage, NormalImage)> {
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
