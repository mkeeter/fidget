//! 3D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    Image as GenericImage, RenderConfig as RenderConfigLike, RenderError,
    RenderWorker, Tile, TileSizesRef,
};
use fidget_core::{
    eval::Function,
    render::{CancelToken, RenderHints, ThreadPool, TileSizes},
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    types::{Grad, Interval},
};

use nalgebra::{Matrix4, Point3, Vector2, Vector3};
use zerocopy::{FromBytes, Immutable, IntoBytes};

/// Image containing depth and normal at each pixel
pub type Image = GenericImage<GeometryPixel, RenderSize>;

/// Size type for 3D rendering
pub type RenderSize = fidget_core::render::VoxelSize;

////////////////////////////////////////////////////////////////////////////////

/// Settings for 3D rendering
pub struct RenderConfig<'a> {
    /// Render size
    ///
    /// The resulting image will have the given width and height; depth sets the
    /// number of voxels to evaluate within each pixel of the image (stacked
    /// into a column going into the screen).
    pub image_size: RenderSize,

    /// World-to-model transform
    pub world_to_model: Matrix4<f32>,

    /// Tile sizes to use during evaluation.
    ///
    /// If this is `None`, then evaluation will use
    /// [`RenderHints::tile_sizes_3d`] to select based on evaluator type.
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
            world_to_model: Matrix4::identity(),
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
    /// Render a shape in 3D using this configuration
    ///
    /// Returns [`Ok(Some(Image))`](Image) of pixel data on success, `Ok(None)`
    /// if the render was cancelled, or an error.
    ///
    /// In the resulting image, saturated pixels (i.e. pixels in the image which
    /// are fully occupied up to the camera) are represented with `depth =
    /// self.image_size.depth()` and a normal of `[0, 0, 1]`.
    pub fn run<F: Function + RenderHints>(
        &self,
        shape: Shape<F>,
    ) -> Result<Option<Image>, RenderError> {
        self.run_with_vars::<F>(shape, &ShapeVars::new())
    }

    /// Render a shape in 3D using this configuration and variables
    pub fn run_with_vars<F: Function + RenderHints>(
        &self,
        shape: Shape<F>,
        vars: &ShapeVars<f32>,
    ) -> Result<Option<Image>, RenderError> {
        render(shape, vars, self)
    }

    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> Matrix4<f32> {
        self.world_to_model * self.image_size.screen_to_world()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Pixel type for a [`voxel::Image`](Image)
///
/// This type can be passed directly in a buffer to the GPU.
#[repr(C)]
#[derive(
    Debug, Default, Copy, Clone, IntoBytes, FromBytes, Immutable, PartialEq,
)]
pub struct GeometryPixel {
    /// Z position of this pixel, in voxel units
    ///
    /// The fractional component is always zero. Empty pixels always have a
    /// depth of 0.
    pub depth: f32,
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

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,

    xg: Vec<Grad>,
    yg: Vec<Grad>,
    zg: Vec<Grad>,

    /// Depth of each column
    columns: Vec<usize>,
}

impl Scratch {
    fn new(tile_size: usize) -> Self {
        let size2 = tile_size.pow(2);
        let size3 = tile_size.pow(3);

        Self {
            x: vec![0.0; size3],
            y: vec![0.0; size3],
            z: vec![0.0; size3],

            xg: vec![Grad::from(0.0); size2],
            yg: vec![Grad::from(0.0); size2],
            zg: vec![Grad::from(0.0); size2],

            columns: vec![0; size2],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,
    vars: &'a ShapeVars<f32>,

    transform: nalgebra::Matrix4<f32>,
    image_size: RenderSize,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,

    /// Output images for this specific tile
    out: Image,
}

impl<'a, F: Function> RenderWorker<'a, F> for Worker<'a, F> {
    type Config = RenderConfig<'a>;
    type Output = Image;

    fn new(
        cfg: &'a Self::Config,
        tile_sizes: TileSizesRef<'a>,
        vars: &'a ShapeVars<f32>,
    ) -> Self {
        let transform = cfg.mat();
        let buf_size = tile_sizes.last();
        let scratch = Scratch::new(buf_size);
        Worker {
            tile_sizes,
            vars,

            scratch,
            out: Default::default(),

            transform,
            image_size: cfg.image_size,

            eval_float_slice: Default::default(),
            eval_interval: Default::default(),
            eval_grad_slice: Default::default(),

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
        // Prepare local tile data to fill out
        let root_tile_size = self.tile_sizes[0];
        self.out = Image::new(RenderSize::from(root_tile_size as u32));
        for k in (0..self.image_size[2].div_ceil(root_tile_size as u32)).rev() {
            let tile = Tile::new(Point3::new(
                tile.corner.x,
                tile.corner.y,
                k as usize * root_tile_size,
            ));
            if !self.render_tile_recurse(shape, 0, tile) {
                break;
            }
        }
        std::mem::take(&mut self.out)
    }
}

impl<F: Function> Worker<'_, F> {
    /// Returns the data offset of a row within a subtile
    pub(crate) fn tile_row_offset(&self, tile: Tile<3>, row: usize) -> usize {
        self.tile_sizes.pixel_offset(tile.add(Vector2::new(0, row)))
    }

    /// Render a single tile
    ///
    /// Returns `true` if we should keep rendering, `false` otherwise
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
        depth: usize,
        tile: Tile<3>,
    ) -> bool {
        // Early exit if every single pixel is filled
        let tile_size = self.tile_sizes[depth];
        let fill_z = (tile.corner[2] + tile_size + 1) as f32;
        if (0..tile_size).all(|y| {
            let i = self.tile_row_offset(tile, y);
            (0..tile_size).all(|x| self.out[i + x].depth >= fill_z)
        }) {
            return false;
        }

        let base = Point3::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(base.z, base.z + tile_size as f32);

        let (i, trace) = self
            .eval_interval
            .eval_with_transform_and_vars(
                shape.i_tape(&mut self.tape_storage),
                x,
                y,
                z,
                &self.transform,
                self.vars,
            )
            .unwrap();

        // Return early if this tile is completely empty or full, returning
        // `data_interval` to scratch memory for reuse.
        if i.upper() < 0.0 {
            for y in 0..tile_size {
                let i = self.tile_row_offset(tile, y);
                for x in 0..tile_size {
                    self.out[i + x].depth = self.out[i + x].depth.max(fill_z);
                }
            }
            return false; // completely full, stop rendering
        } else if i.lower() > 0.0 {
            return true; // complete empty, keep going
        }

        // Calculate a simplified tape based on the trace
        let sub_tape = if let Some(trace) = trace.as_ref() {
            shape.simplify(
                trace,
                &mut self.workspace,
                &mut self.shape_storage,
                &mut self.tape_storage,
            )
        } else {
            shape
        };

        // Recurse!
        if let Some(next_tile_size) = self.tile_sizes.get(depth + 1) {
            let n = tile_size / next_tile_size;

            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        self.render_tile_recurse(
                            sub_tape,
                            depth + 1,
                            Tile::new(
                                tile.corner
                                    + Vector3::new(i, j, k) * next_tile_size,
                            ),
                        );
                    }
                }
            }
        } else {
            self.render_tile_pixels(sub_tape, tile_size, tile);
        };
        // TODO recycle something here?
        true // keep going
    }

    fn render_tile_pixels(
        &mut self,
        shape: &mut RenderHandle<F>,
        tile_size: usize,
        tile: Tile<3>,
    ) {
        // Prepare for pixel-by-pixel evaluation
        let mut index = 0;
        assert!(self.scratch.x.len() >= tile_size.pow(3));
        assert!(self.scratch.y.len() >= tile_size.pow(3));
        assert!(self.scratch.z.len() >= tile_size.pow(3));
        self.scratch.columns.clear();
        for xy in 0..tile_size.pow(2) {
            let i = xy % tile_size;
            let j = xy / tile_size;

            let o = self.tile_sizes.pixel_offset(tile.add(Vector2::new(i, j)));

            // Skip pixels which are behind the image
            let zmax = (tile.corner[2] + tile_size) as f32;
            if self.out[o].depth >= zmax {
                continue;
            }

            for k in (0..tile_size).rev() {
                // SAFETY:
                // Index cannot exceed tile_size**3, which is (a) the size
                // that we allocated in `Scratch::new` and (b) checked by
                // assertions above.
                //
                // Using unsafe indexing here is a roughly 2.5% speedup,
                // since this is the hottest loop.
                unsafe {
                    *self.scratch.x.get_unchecked_mut(index) =
                        (tile.corner[0] + i) as f32;
                    *self.scratch.y.get_unchecked_mut(index) =
                        (tile.corner[1] + j) as f32;
                    *self.scratch.z.get_unchecked_mut(index) =
                        (tile.corner[2] + k) as f32;
                }
                index += 1;
            }
            self.scratch.columns.push(xy);
        }
        let size = index;
        assert!(size > 0);

        let out = self
            .eval_float_slice
            .eval_with_transform_and_vars(
                shape.f_tape(&mut self.tape_storage),
                &self.scratch.x[..index],
                &self.scratch.y[..index],
                &self.scratch.z[..index],
                &self.transform,
                self.vars,
            )
            .unwrap();

        // We're iterating over a few things simultaneously
        // - col refers to the xy position in the tile
        // - grad refers to points that we must do gradient evaluation on
        let mut grad = 0;
        let mut depth = out.chunks(tile_size);
        for col in 0..self.scratch.columns.len() {
            // Find the first set pixel in the column
            let depth = depth.next().unwrap();
            let k = match depth.iter().enumerate().find(|(_, d)| **d < 0.0) {
                Some((i, _)) => i,
                None => continue,
            };

            // Get X and Y values from the `columns` array.  Note that we can't
            // iterate over the array directly because we're also modifying it
            // (below)
            let xy = self.scratch.columns[col];
            let i = xy % tile_size;
            let j = xy / tile_size;

            // Flip Z value, since voxels are packed front-to-back
            let k = tile_size - 1 - k;

            // Set the depth of the pixel
            let o = self.tile_sizes.pixel_offset(tile.add(Vector2::new(i, j)));
            let z = (tile.corner[2] + k + 1) as f32;
            assert!(self.out[o].depth < z);
            self.out[o].depth = z;

            // Prepare to do gradient rendering of this point.
            // We step one voxel above the surface to reduce
            // glitchiness on edges and corners, where rendering
            // inside the surface could pick the wrong normal.
            self.scratch.xg[grad] =
                Grad::new((tile.corner[0] + i) as f32, 1.0, 0.0, 0.0);
            self.scratch.yg[grad] =
                Grad::new((tile.corner[1] + j) as f32, 0.0, 1.0, 0.0);
            self.scratch.zg[grad] =
                Grad::new((tile.corner[2] + k) as f32, 0.0, 0.0, 1.0);

            // This can only be called once per iteration, so we'll
            // never overwrite parts of columns that are still used
            // by the outer loop
            self.scratch.columns[grad] = o;
            grad += 1;
        }

        if grad > 0 {
            let out = self
                .eval_grad_slice
                .eval_with_transform_and_vars(
                    shape.g_tape(&mut self.tape_storage),
                    &self.scratch.xg[..grad],
                    &self.scratch.yg[..grad],
                    &self.scratch.zg[..grad],
                    &self.transform,
                    self.vars,
                )
                .unwrap();

            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
                let g = out[index];
                self.out[*o].normal = [g.dx, g.dy, g.dz];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Renders the given tape into a 3D image according to the provided
/// configuration.
///
/// The tape provides the shape; the configuration supplies resolution,
/// transforms, etc.
///
/// This function is parameterized by shape type, which determines how we
/// perform evaluation.
///
/// Returns [`Ok(Some(Image))`](Image) of pixel data on success, `Ok(None)` if
/// the render was cancelled, or an error.
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
        default_tile_sizes = F::tile_sizes_3d();
        TileSizesRef::new(&default_tile_sizes, max_size)
    };
    let tiles = match super::render_tiles::<F, Worker<F>>(
        shape, vars, config, tile_sizes,
    ) {
        Some(t) => t,
        None => return Ok(None),
    };

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image = Image::new(config.image_size);
    for (tile, out) in tiles {
        let mut index = 0;
        for j in 0..tile_sizes[0] {
            let y = j + tile.corner.y;
            for i in 0..tile_sizes[0] {
                let x = i + tile.corner.x;
                if x < width && y < height {
                    let o = y * width + x;
                    if out[index].depth >= image[o].depth {
                        // Clamp voxels to the image depth
                        let d = (config.image_size.depth() - 1) as f32;
                        if out[index].depth >= d {
                            image[o] = GeometryPixel {
                                depth: d + 1.0,
                                normal: [0.0, 0.0, 1.0],
                            };
                        } else {
                            image[o] = out[index];
                        }
                    }
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
    use fidget_core::{Context, var::Var, vm::VmShape};

    /// Make sure we don't crash if there's only a single tile
    #[test]
    fn test_tile_queues() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::from(128), // very small!
            ..Default::default()
        };
        let image = cfg
            .run(shape)
            .expect("rendering should not fail")
            .expect("rendering should not be cancelled");
        assert_eq!(image.len(), 128 * 128);
    }

    #[test]
    fn cancel_render() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::new(64, 64, 64),
            ..Default::default()
        };
        let cancel = cfg.cancel.clone();
        cancel.cancel();
        assert!(cfg.run::<_>(shape).unwrap().is_none());
    }

    #[test]
    fn missing_var() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let v = ctx.var(Var::new());
        let s = ctx.sub(x, v).unwrap();
        let shape = VmShape::new(&ctx, s).unwrap();

        let cfg = RenderConfig {
            image_size: RenderSize::new(64, 64, 64),
            ..Default::default()
        };
        let Err(out) = cfg.run::<_>(shape.clone()) else {
            panic!("expected error")
        };
        let var = ctx.get_var(v).unwrap();
        let Var::V(i) = var else {
            panic!("expected Var::V")
        };
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
}
