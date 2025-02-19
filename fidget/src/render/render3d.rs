//! 3D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    eval::Function,
    render::{
        config::{Tile, VoxelRenderConfig},
        DepthImage, NormalImage, RenderWorker, TileSizes, VoxelSize,
    },
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    types::{Grad, Interval},
};

use nalgebra::{Point3, Vector2, Vector3};

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
    tile_sizes: &'a TileSizes,
    image_size: VoxelSize,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,

    /// Output images for this specific tile
    depth: DepthImage,
    color: NormalImage,
}

impl<'a, F: Function> RenderWorker<'a, F> for Worker<'a, F> {
    type Config = VoxelRenderConfig<'a>;
    type Output = (DepthImage, NormalImage);

    fn new(cfg: &'a Self::Config) -> Self {
        let buf_size = cfg.tile_sizes.last();
        let scratch = Scratch::new(buf_size);
        Worker {
            scratch,
            depth: Default::default(),
            color: Default::default(),
            tile_sizes: &cfg.tile_sizes,
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
        vars: &ShapeVars<f32>,
        tile: super::config::Tile<2>,
    ) -> Self::Output {
        // Prepare local tile data to fill out
        let root_tile_size = self.tile_sizes[0];
        self.depth = DepthImage::new(root_tile_size, root_tile_size);
        self.color = NormalImage::new(root_tile_size, root_tile_size);
        for k in (0..self.image_size[2].div_ceil(root_tile_size as u32)).rev() {
            let tile = Tile::new(Point3::new(
                tile.corner.x,
                tile.corner.y,
                k as usize * root_tile_size,
            ));
            if !self.render_tile_recurse(shape, vars, 0, tile) {
                break;
            }
        }
        let depth = std::mem::take(&mut self.depth);
        let color = std::mem::take(&mut self.color);
        (depth, color)
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
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: Tile<3>,
    ) -> bool {
        // Early exit if every single pixel is filled
        let tile_size = self.tile_sizes[depth];
        let fill_z = (tile.corner[2] + tile_size + 1).try_into().unwrap();
        if (0..tile_size).all(|y| {
            let i = self.tile_row_offset(tile, y);
            (0..tile_size).all(|x| self.depth[i + x] >= fill_z)
        }) {
            return false;
        }

        let base = Point3::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(base.z, base.z + tile_size as f32);
        println!("int: {x} {y} => {}", shape.shape.size());

        let (i, trace) = self
            .eval_interval
            .eval_v(shape.i_tape(&mut self.tape_storage), x, y, z, vars)
            .unwrap();

        // Return early if this tile is completely empty or full, returning
        // `data_interval` to scratch memory for reuse.
        if i.upper() < 0.0 {
            for y in 0..tile_size {
                let i = self.tile_row_offset(tile, y);
                for x in 0..tile_size {
                    self.depth[i + x] = self.depth[i + x].max(fill_z);
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
                            vars,
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
            self.render_tile_pixels(sub_tape, vars, tile_size, tile);
        };
        // TODO recycle something here?
        true // keep going
    }

    fn render_tile_pixels(
        &mut self,
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        tile_size: usize,
        tile: Tile<3>,
    ) {
        println!(
            "pix: [{}, {}] [{}, {}] => {}",
            tile.corner[0],
            tile.corner[0] + tile_size,
            tile.corner[1],
            tile.corner[1] + tile_size,
            shape.shape.size()
        );

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
            let zmax = (tile.corner[2] + tile_size).try_into().unwrap();
            if self.depth[o] >= zmax {
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
            .eval_v(
                shape.f_tape(&mut self.tape_storage),
                &self.scratch.x[..index],
                &self.scratch.y[..index],
                &self.scratch.z[..index],
                vars,
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
            let z = (tile.corner[2] + k + 1).try_into().unwrap();
            assert!(self.depth[o] < z);
            self.depth[o] = z;

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
                .eval_v(
                    shape.g_tape(&mut self.tape_storage),
                    &self.scratch.xg[..grad],
                    &self.scratch.yg[..grad],
                    &self.scratch.zg[..grad],
                    vars,
                )
                .unwrap();

            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
                let g = out[index];
                self.color[*o] = [g.dx, g.dy, g.dz];
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
/// Returns two `Vec` of pixel data (color, normals) if rendering succeeds, or
/// `None` if rendering was cancelled (using the [`VoxelRenderConfig::cancel`]
/// token)
pub fn render<F: Function>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &VoxelRenderConfig,
) -> Option<(DepthImage, NormalImage)> {
    let shape = shape.apply_transform(config.mat());

    let tiles = super::render_tiles::<F, Worker<F>>(shape, vars, config)?;

    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    let mut image_depth = DepthImage::new(width, height);
    let mut image_color = NormalImage::new(width, height);
    for (tile, (depth, color)) in tiles {
        let mut index = 0;
        for j in 0..config.tile_sizes[0] {
            let y = j + tile.corner.y;
            for i in 0..config.tile_sizes[0] {
                let x = i + tile.corner.x;
                if x < width && y < height {
                    let o = y * width + x;
                    if depth[index] >= image_depth[o] {
                        image_color[o] = color[index];
                        image_depth[o] = depth[index];
                    }
                }
                index += 1;
            }
        }
    }
    Some((image_depth, image_color))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        context::Tree,
        eval::MathFunction,
        render::{View3, VoxelSize},
        var::Var,
        vm::VmShape,
        Context,
    };

    /// Make sure we don't crash if there's only a single tile
    #[test]
    fn test_tile_queues() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::from(128), // very small!
            ..Default::default()
        };
        let (depth, rgb) = cfg.run(shape).unwrap();
        assert_eq!(depth.len(), 128 * 128);
        assert_eq!(rgb.len(), 128 * 128);
    }

    fn sphere_var<F: Function + MathFunction>() {
        let (x, y, z) = Tree::axes();
        let v = Var::new();
        let c = Tree::from(v);
        let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
        let shape = Shape::<F>::from(sphere);

        let size = 32;
        for scale in [1.0, 0.5] {
            let cfg = VoxelRenderConfig {
                image_size: VoxelSize::from(size),
                view: View3::from_center_and_scale(Vector3::zeros(), scale),
                ..Default::default()
            };

            for r in [0.5, 0.75] {
                let mut vars = ShapeVars::new();
                vars.insert(v.index().unwrap(), r);
                let (depth, _normal) =
                    cfg.run_with_vars::<_>(shape.clone(), &vars).unwrap();

                let epsilon = 0.08;
                for (i, p) in depth.iter().enumerate() {
                    let size = size as i32;
                    let i = i as i32;
                    let x = (((i % size) - size / 2) as f32 / size as f32)
                        * 2.0
                        * scale;
                    let y = (((i / size) - size / 2) as f32 / size as f32)
                        * 2.0
                        * scale;
                    let z = (*p as i32 - size / 2) as f32 / size as f32
                        * 2.0
                        * scale;
                    if *p == 0 {
                        let v = (x.powi(2) + y.powi(2)).sqrt();
                        assert!(
                            v + epsilon > r,
                            "got z = 0 inside the sphere ({x}, {y}, {z}); \
                         radius is {v}"
                        );
                    } else {
                        let v = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
                        let err = (r - v).abs();
                        assert!(
                            err < epsilon,
                            "too much error {err} at ({x}, {y}, {z}); \
                         radius is {v}, expected {r}"
                        );
                    }
                }
            }
        }
    }

    macro_rules! render_tests {
        ($i:ident) => {
            mod $i {
                use super::*;
                #[test]
                fn vm() {
                    $i::<$crate::vm::VmFunction>();
                }
                #[test]
                fn vm3() {
                    $i::<$crate::vm::GenericVmFunction<3>>();
                }
                #[cfg(feature = "jit")]
                #[test]
                fn jit() {
                    $i::<$crate::jit::JitFunction>();
                }
            }
        };
    }

    render_tests!(sphere_var);

    #[test]
    fn cancel_render() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = VmShape::new(&ctx, x).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::new(64, 64, 64),
            ..Default::default()
        };
        let cancel = cfg.cancel.clone();
        cancel.cancel();
        let out = cfg.run::<_>(shape);
        assert!(out.is_none());
    }
}
