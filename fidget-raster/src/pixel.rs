//! 2D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    Image as GenericImage, RenderConfig as RenderConfigLike, RenderError,
    RenderWorker, Tile, TileSizesRef,
};
use fidget_core::{
    eval::Function,
    render::{CancelToken, RenderHints, ThreadPool, TileSizes},
    shape::{
        Shape, ShapeBulkEval, ShapeBulkEvalError, ShapeTracingEval,
        ShapeTracingEvalError, ShapeVars,
    },
    types::Interval,
};
use nalgebra::{Matrix3, Point2, Vector2};

////////////////////////////////////////////////////////////////////////////////

/// Image type for 2D rendering
pub type Image = GenericImage<RawDistancePixel>;

/// Size for 2D rendering
pub type RenderSize = fidget_core::render::ImageSize;

/// Settings for 2D rendering
pub struct RenderConfig<'a> {
    /// Render size
    pub image_size: RenderSize,

    /// World-to-model transform
    pub world_to_model: Matrix3<f32>,

    /// Render the distance values of individual pixels
    pub pixel_perfect: bool,

    /// Tile sizes to use during evaluation.
    ///
    /// If this is `None`, then evaluation will use
    /// [`RenderHints::tile_sizes_2d`] to select based on evaluator type.
    pub tile_sizes: Option<TileSizes>,

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<&'a ThreadPool>,

    /// Token to cancel rendering
    pub cancel: CancelToken,
}

impl Default for RenderConfig<'_> {
    fn default() -> Self {
        Self {
            image_size: RenderSize::from(512),
            tile_sizes: None,
            world_to_model: Matrix3::identity(),
            pixel_perfect: false,
            threads: Some(&ThreadPool::Global),
            cancel: CancelToken::new(),
        }
    }
}

impl RenderConfigLike for RenderConfig<'_> {
    fn width(&self) -> u32 {
        self.image_size.width()
    }
    fn height(&self) -> u32 {
        self.image_size.height()
    }
    fn threads(&self) -> Option<&ThreadPool> {
        self.threads
    }
    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

impl RenderConfig<'_> {
    /// Render a shape in 2D using this configuration
    ///
    ///
    /// Returns [`Ok(Some(Image))`](Image) of pixel data on success, `Ok(None)`
    /// if the render was cancelled, or an error.
    pub fn run<F: Function + RenderHints>(
        &self,
        shape: Shape<F>,
    ) -> Result<Option<Image>, RenderError> {
        self.run_with_vars::<F>(shape, &ShapeVars::new())
    }

    /// Render a shape in 2D using this configuration and variables
    pub fn run_with_vars<F: Function + RenderHints>(
        &self,
        shape: Shape<F>,
        vars: &ShapeVars<f32>,
    ) -> Result<Option<Image>, RenderError> {
        render(shape, vars, self)
    }

    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> Matrix3<f32> {
        self.world_to_model * self.image_size.screen_to_world()
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}

impl Scratch {
    fn new(size: usize) -> Self {
        Self {
            x: vec![0.0; size],
            y: vec![0.0; size],
            z: vec![0.0; size],
        }
    }
}

/// A raw pixel in a 2D image
///
/// This is either a single distance value or a description of a fill; both
/// cases are packed into an `f32` (using `NaN`-boxing in the latter case).
///
/// The `NaN`-boxing is transparent to the user; for the curious, it is
/// implemented by splitting the mantissa into three components:
///
/// - Bit 0 indicates the sign of the fill (1 for inside, 0 for outside)
/// - Bits 1-9 indicate the depth at which evaluation terminated, which is
///   useful for certain debug visualizations
/// - Bits 10-17 are a fixed bit pattern, which both ensures that the value is
///   treated as `NaN` in the `[empty, depth 0]` case (rather than infinity) and
///   distinguishes from `NaN` values generated during normal evaluation
#[derive(Copy, Clone, Debug, Default)]
pub struct RawDistancePixel(f32);

/// Unpacked data from a [`RawDistancePixel`]
#[derive(Copy, Clone, Debug)]
pub enum DistancePixel {
    /// The pixel represents a pseudo-distance value
    Value(f32),
    /// The pixel is from a fill region and no distance is available
    Fill {
        /// Recursion depth which wrote the pixel
        depth: u8,
        /// Is the pixel inside or outside the image
        inside: bool,
    },
}

impl RawDistancePixel {
    const KEY: u32 = 0b1111_0110 << 9;
    const KEY_MASK: u32 = 0b1111_1111 << 9;

    /// Checks whether the pixel is inside or outside the model
    ///
    /// Distances of exactly `0.0` and non-boxed `NaN` values are treated as
    /// outside (i.e. returning `false`).
    #[inline]
    pub fn inside(self) -> bool {
        match self.unpack() {
            DistancePixel::Fill { inside, .. } => inside,
            DistancePixel::Value(v) => v < 0.0,
        }
    }

    /// Checks whether this is a distance point sample
    #[inline]
    pub fn is_distance(self) -> bool {
        if !self.0.is_nan() {
            return true;
        }
        let bits = self.0.to_bits();
        (bits & Self::KEY_MASK) != Self::KEY
    }

    /// Unpacks into a [`DistancePixel`]
    #[inline]
    pub fn unpack(self) -> DistancePixel {
        if self.is_distance() {
            DistancePixel::Value(self.0)
        } else {
            let bits = self.0.to_bits();
            let inside = (bits & 1) == 1;
            let depth = (bits >> 1) as u8;
            DistancePixel::Fill { inside, depth }
        }
    }
}

impl From<DistancePixel> for RawDistancePixel {
    #[inline]
    fn from(p: DistancePixel) -> Self {
        match p {
            DistancePixel::Value(f) => Self::from(f),
            DistancePixel::Fill { depth, inside } => {
                let bits = 0x7FC00000
                    | (u32::from(depth) << 1)
                    | u32::from(inside)
                    | Self::KEY;
                Self(f32::from_bits(bits))
            }
        }
    }
}

impl From<f32> for RawDistancePixel {
    #[inline]
    fn from(p: f32) -> Self {
        // Canonicalize the NAN value to avoid colliding with a fill
        Self(if p.is_nan() { f32::NAN } else { p })
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Per-thread worker
struct Worker<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,
    vars: &'a ShapeVars<f32>,

    pixel_perfect: bool,
    scratch: Scratch,
    transform: nalgebra::Matrix4<f32>,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    /// Spare tape storage for reuse
    tape_storage: Vec<F::TapeStorage>,

    /// Spare shape storage for reuse
    shape_storage: Vec<F::Storage>,

    /// Workspace for shape simplification
    workspace: F::Workspace,

    /// Tile being rendered
    ///
    /// This is a root tile, i.e. width and height of `config.tile_sizes[0]`
    image: Image,
}

impl<'a, F: Function> RenderWorker<'a, F> for Worker<'a, F> {
    type Config = RenderConfig<'a>;
    type Output = Image;
    fn new(
        cfg: &'a Self::Config,
        tile_sizes: TileSizesRef<'a>,
        vars: &'a ShapeVars<f32>,
    ) -> Self {
        // Convert to a 4x4 matrix and apply to the shape
        let transform = cfg.mat();
        let transform = transform.insert_row(2, 0.0);
        let transform = transform.insert_column(2, 0.0);

        Worker::<F> {
            tile_sizes,
            transform,
            vars,

            scratch: Scratch::new(tile_sizes.last().pow(2)),
            pixel_perfect: cfg.pixel_perfect,
            image: Default::default(),
            eval_float_slice: Default::default(),
            eval_interval: Default::default(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F>,
        tile: Tile<2>,
    ) -> Self::Output {
        self.image = Image::new((self.tile_sizes[0] as u32).into());
        self.render_tile_recurse(shape, 0, tile);
        std::mem::take(&mut self.image)
    }
}

impl<F: Function> Worker<'_, F> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
        depth: usize,
        tile: Tile<2>,
    ) {
        let tile_size = self.tile_sizes[depth];

        // Find the interval bounds of the region, in screen coordinates
        let base = Point2::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(0.0, 0.0);

        // Evaluation applies the world-to-model transform.  We know that vars
        // are valid because we check them at the top of `render`
        let (i, simplify) =
            match self.eval_interval.eval_with_transform_and_vars(
                shape.i_tape(&mut self.tape_storage),
                x,
                y,
                z,
                &self.transform,
                self.vars,
            ) {
                Ok(v) => v,
                Err(ShapeTracingEvalError::MissingVar(..)) => unreachable!(),
            };

        if !self.pixel_perfect {
            let pixel = if i.upper() < 0.0 {
                Some(DistancePixel::Fill {
                    inside: true,
                    depth: depth as u8,
                })
            } else if i.lower() > 0.0 {
                Some(DistancePixel::Fill {
                    inside: false,
                    depth: depth as u8,
                })
            } else {
                None
            };
            if let Some(pixel) = pixel {
                let fill = pixel.into();
                for y in 0..tile_size {
                    let start = self
                        .tile_sizes
                        .pixel_offset(tile.add(Vector2::new(0, y)));
                    self.image[start..][..tile_size].fill(fill);
                }
                return;
            }
        }

        let sub_tape = if let Some(trace) = simplify.as_ref() {
            shape.simplify(
                trace,
                &mut self.workspace,
                &mut self.shape_storage,
                &mut self.tape_storage,
            )
        } else {
            shape
        };

        if let Some(next_tile_size) = self.tile_sizes.get(depth + 1) {
            let n = tile_size / next_tile_size;
            for j in 0..n {
                for i in 0..n {
                    self.render_tile_recurse(
                        sub_tape,
                        depth + 1,
                        Tile::new(
                            tile.corner + Vector2::new(i, j) * next_tile_size,
                        ),
                    );
                }
            }
        } else {
            self.render_tile_pixels(sub_tape, tile_size, tile);
        }
    }

    fn render_tile_pixels(
        &mut self,
        shape: &mut RenderHandle<F>,
        tile_size: usize,
        tile: Tile<2>,
    ) {
        let mut index = 0;
        for j in 0..tile_size {
            for i in 0..tile_size {
                self.scratch.x[index] = (tile.corner[0] + i) as f32;
                self.scratch.y[index] = (tile.corner[1] + j) as f32;
                index += 1;
            }
        }

        let out = match self.eval_float_slice.eval_with_transform_and_vars(
            shape.f_tape(&mut self.tape_storage),
            &self.scratch.x,
            &self.scratch.y,
            &self.scratch.z,
            &self.transform,
            self.vars,
        ) {
            Ok(v) => v,
            // We checked the var map at the beginning of `render`
            Err(ShapeBulkEvalError::MissingVar(..))
            // We know that our X/Y/Z slices are all the same length
            | Err(ShapeBulkEvalError::MismatchedVarSlices { .. })
                => unreachable!(),
        };

        let mut index = 0;
        for j in 0..tile_size {
            let o = self.tile_sizes.pixel_offset(tile.add(Vector2::new(0, j)));
            for i in 0..tile_size {
                self.image[o + i] = out[index].into();
                index += 1;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Renders the given tape into a 2D image at Z = 0 according to the provided
/// configuration.
///
/// The tape provides the shape; the configuration supplies resolution,
/// transforms, etc.
///
/// This function is parameterized by shape type (which determines how we
/// perform evaluation).
///
/// Returns an [`Ok(Some(Image))`](Image) of pixel data if rendering succeeds,
/// `Ok(None)` if rendering was cancelled (using the [`RenderConfig::cancel`]
/// token), or an error.
pub fn render<F: Function + RenderHints>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &RenderConfig,
) -> Result<Option<Image>, RenderError> {
    vars.check(&shape)?;
    let max_size = config.width().max(config.height()) as usize;
    let default_tile_sizes;
    let tile_sizes = if let Some(ts) = &config.tile_sizes {
        TileSizesRef::new(ts, max_size)
    } else {
        default_tile_sizes = F::tile_sizes_2d();
        TileSizesRef::new(&default_tile_sizes, max_size)
    };
    let tiles = match super::render_tiles::<F, Worker<F>>(
        shape.clone(),
        vars,
        config,
        tile_sizes,
    ) {
        Some(t) => t,
        None => return Ok(None),
    };

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = Image::new(config.image_size);
    for (tile, data) in tiles.iter() {
        let mut index = 0;
        for j in 0..tile_sizes[0] {
            let y = j + tile.corner.y;
            for i in 0..tile_sizes[0] {
                let x = i + tile.corner.x;
                if y < height && x < width {
                    image[(y, x)] = data[index];
                }
                index += 1;
            }
        }
    }
    Ok(Some(image))
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{
        Context,
        shape::Shape,
        var::Var,
        vm::{VmFunction, VmShape},
    };

    const HI: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/hi.vm"));

    #[test]
    fn render2d_cancel() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = Shape::<VmFunction>::new(&ctx, root).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::new(64, 64),
            ..Default::default()
        };
        let cancel = cfg.cancel.clone();
        cancel.cancel();
        assert!(cfg.run(shape).unwrap().is_none());
    }

    #[test]
    fn missing_var() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let v = ctx.var(Var::new());
        let s = ctx.sub(x, v).unwrap();
        let shape = VmShape::new(&ctx, s).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::new(64, 64),
            ..Default::default()
        };
        let Err(out) = cfg.run::<_>(shape.clone()) else {
            panic!("expected error")
        };
        let var = ctx.get_var(v).unwrap();
        let i = var.index().expect("expected Var::V");
        assert_eq!(
            out,
            RenderError::MissingVar(fidget_core::shape::MissingVar { var: i })
        );

        let mut vars = ShapeVars::new();
        vars.insert(i, 1.0);
        cfg.run_with_vars::<_>(shape, &vars)
            .expect("rendering worked")
            .expect("not cancelled");
    }

    #[test]
    fn test_default_render_config() {
        let config = RenderConfig {
            image_size: RenderSize::from(512),
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

        let config = RenderConfig {
            image_size: RenderSize::from(575),
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
}
