//! 2D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    Image, RenderConfig, RenderWorker, TileSizesRef,
    config::{ImageRenderConfig, Tile},
};
use fidget_core::{
    eval::Function,
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    types::Interval,
};
use nalgebra::{Point2, Vector2};

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

/// A pixel in a 2D image
///
/// This is either a single distance value or a description of a fill; both
/// cases are packed into an `f32` (using `NaN`-boxing in the latter case).
#[derive(Copy, Clone, Debug, Default)]
pub struct DistancePixel(f32);

#[derive(Copy, Clone, Debug)]
pub struct PixelFill {
    pub depth: u8,
    pub inside: bool,
}

impl DistancePixel {
    const KEY: u32 = 0b1111_0110 << 9;
    const KEY_MASK: u32 = 0b1111_1111 << 9;

    /// Checks whether the pixel is inside or outside the model
    ///
    /// This will return `false` if the distance value is `NAN`
    #[inline]
    pub fn inside(self) -> bool {
        match self.fill() {
            Ok(f) => f.inside,
            Err(v) => v < 0.0,
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

    /// Returns the distance value (if present)
    #[inline]
    pub fn distance(self) -> Result<f32, PixelFill> {
        match self.fill() {
            Ok(v) => Err(v),
            Err(v) => Ok(v),
        }
    }

    /// Returns the fill details (if present)
    #[inline]
    pub fn fill(self) -> Result<PixelFill, f32> {
        if self.is_distance() {
            Err(self.0)
        } else {
            let bits = self.0.to_bits();
            let inside = (bits & 1) == 1;
            let depth = (bits >> 1) as u8;
            Ok(PixelFill { inside, depth })
        }
    }
}

impl From<PixelFill> for DistancePixel {
    #[inline]
    fn from(p: PixelFill) -> Self {
        let bits = 0x7FC00000
            | (u32::from(p.depth) << 1)
            | u32::from(p.inside)
            | Self::KEY;
        DistancePixel(f32::from_bits(bits))
    }
}

impl From<f32> for DistancePixel {
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
    pixel_perfect: bool,
    scratch: Scratch,

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
    image: Image<DistancePixel>,
}

impl<'a, F: Function, T> RenderWorker<'a, F, T> for Worker<'a, F> {
    type Config = ImageRenderConfig<'a>;
    type Output = Image<DistancePixel>;
    fn new(cfg: &'a Self::Config) -> Self {
        let tile_sizes = cfg.tile_sizes();
        Worker::<F> {
            scratch: Scratch::new(tile_sizes.last().pow(2)),
            pixel_perfect: cfg.pixel_perfect,
            image: Default::default(),
            tile_sizes,
            eval_float_slice: Default::default(),
            eval_interval: Default::default(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        tile: super::config::Tile<2>,
    ) -> Self::Output {
        self.image = Image::new((self.tile_sizes[0] as u32).into());
        self.render_tile_recurse(shape, vars, 0, tile);
        std::mem::take(&mut self.image)
    }
}

impl<F: Function> Worker<'_, F> {
    fn render_tile_recurse<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: Tile<2>,
    ) {
        let tile_size = self.tile_sizes[depth];

        // Find the interval bounds of the region, in screen coordinates
        let base = Point2::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(0.0, 0.0);

        // The shape applies the screen-to-model transform
        let (i, simplify) = self
            .eval_interval
            .eval_v(shape.i_tape(&mut self.tape_storage), x, y, z, vars)
            .unwrap();

        if !self.pixel_perfect {
            let pixel = if i.upper() < 0.0 {
                Some(PixelFill {
                    inside: true,
                    depth: depth as u8,
                })
            } else if i.lower() > 0.0 {
                Some(PixelFill {
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
                        vars,
                        depth + 1,
                        Tile::new(
                            tile.corner + Vector2::new(i, j) * next_tile_size,
                        ),
                    );
                }
            }
        } else {
            self.render_tile_pixels(sub_tape, vars, tile_size, tile);
        }
    }

    fn render_tile_pixels<T>(
        &mut self,
        shape: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
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

        let out = self
            .eval_float_slice
            .eval_v(
                shape.f_tape(&mut self.tape_storage),
                &self.scratch.x,
                &self.scratch.y,
                &self.scratch.z,
                vars,
            )
            .unwrap();

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
/// Returns an `Image<DistancePixel>` of pixel data if rendering succeeds, or
/// `None` if rendering was cancelled (using the [`ImageRenderConfig::cancel`]
/// token)
pub fn render<F: Function>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &ImageRenderConfig,
) -> Option<Image<DistancePixel>> {
    // Convert to a 4x4 matrix and apply to the shape
    let mat = config.mat();
    let mat = mat.insert_row(2, 0.0);
    let mat = mat.insert_column(2, 0.0);
    let shape = shape.with_transform(mat);

    let tiles =
        super::render_tiles::<F, Worker<F>, _>(shape.clone(), vars, config)?;
    let tile_sizes = config.tile_sizes();

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
    Some(image)
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{
        Context, render::ImageSize, shape::Shape, vm::VmFunction,
    };

    const HI: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/hi.vm"));

    #[test]
    fn render2d_cancel() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = Shape::<VmFunction>::new(&ctx, root).unwrap();

        let cfg = ImageRenderConfig {
            image_size: ImageSize::new(64, 64),
            ..Default::default()
        };
        let cancel = cfg.cancel.clone();
        cancel.cancel();
        let out = cfg.run(shape);
        assert!(out.is_none());
    }
}
