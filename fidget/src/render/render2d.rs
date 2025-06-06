//! 2D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    eval::Function,
    render::{
        Image, RenderConfig, RenderWorker, TileSizesRef,
        config::{ImageRenderConfig, Tile},
    },
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
    pub fn inside(self) -> bool {
        match self.fill() {
            Ok(f) => f.inside,
            Err(v) => v < 0.0,
        }
    }

    /// Checks whether this is a distance point sample
    pub fn is_distance(self) -> bool {
        if !self.0.is_nan() {
            return true;
        }
        let bits = self.0.to_bits();
        (bits & Self::KEY_MASK) != Self::KEY
    }

    /// Returns the distance value (if present)
    pub fn distance(self) -> Result<f32, PixelFill> {
        match self.fill() {
            Ok(v) => Err(v),
            Err(v) => Ok(v),
        }
    }

    /// Returns the fill details (if present)
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
    fn from(p: PixelFill) -> Self {
        let bits = 0x7FC00000
            | (u32::from(p.depth) << 1)
            | u32::from(p.inside)
            | Self::KEY;
        DistancePixel(f32::from_bits(bits))
    }
}

impl From<f32> for DistancePixel {
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

impl<'a, F: Function> RenderWorker<'a, F> for Worker<'a, F> {
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
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        tile: super::config::Tile<2>,
    ) -> Self::Output {
        self.image = Image::new((self.tile_sizes[0] as u32).into());
        self.render_tile_recurse(shape, vars, 0, tile);
        std::mem::take(&mut self.image)
    }
}

impl<F: Function> Worker<'_, F> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
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

    fn render_tile_pixels(
        &mut self,
        shape: &mut RenderHandle<F>,
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
/// Returns an `Image<f32>` of pixel data if rendering succeeds, or `None` if
/// rendering was cancelled (using the [`ImageRenderConfig::cancel`] token)
pub fn render<F: Function>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &ImageRenderConfig,
) -> Option<Image<DistancePixel>> {
    // Convert to a 4x4 matrix and apply to the shape
    let mat = config.mat();
    let mat = mat.insert_row(2, 0.0);
    let mat = mat.insert_column(2, 0.0);
    let shape = shape.apply_transform(mat);

    let tiles =
        super::render_tiles::<F, Worker<F>>(shape.clone(), vars, config)?;
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
    use crate::{
        Context,
        eval::{Function, MathFunction},
        render::{ImageSize, View2},
        shape::Shape,
        var::Var,
        vm::{GenericVmFunction, VmFunction},
    };

    const HI: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/hi.vm"));
    const QUARTER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/quarter.vm"
    ));

    #[derive(Default)]
    struct Cfg {
        vars: ShapeVars<f32>,
        view: View2,
        wide: bool,
    }

    impl Cfg {
        fn test<F: Function>(&self, shape: Shape<F>, expected: &'static str) {
            let width = if self.wide { 64 } else { 32 };
            let cfg = ImageRenderConfig {
                image_size: ImageSize::new(width, 32),
                view: self.view,
                ..Default::default()
            };
            let out = cfg
                .run_with_vars(shape, &self.vars)
                .expect("rendering should not be cancelled");
            let mut img_str = String::new();
            for (i, b) in out.iter().enumerate() {
                if i % width as usize == 0 {
                    img_str += "\n            ";
                }
                img_str.push(if b.inside() { 'X' } else { '.' });
            }
            if img_str != expected {
                println!("image mismatch detected!");
                println!("Expected:\n{expected}\nGot:\n{img_str}");
                println!("Diff:");
                for (a, b) in img_str.chars().zip(expected.chars()) {
                    print!("{}", if a != b { '!' } else { a });
                }
                panic!("image mismatch");
            }
        }
    }

    fn check_hi<F: Function + MathFunction>() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = Shape::<F>::new(&ctx, root).unwrap();
        const EXPECTED: &str = "
            .................X..............
            .................X..............
            .................X..............
            .................X..........XX..
            .................X..........XX..
            .................X..............
            .................X..............
            .................XXXXXX.....XX..
            .................XXX..XX....XX..
            .................XX....XX...XX..
            .................X......X...XX..
            .................X......X...XX..
            .................X......X...XX..
            .................X......X...XX..
            .................X......X...XX..
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................";
        Cfg::default().test(shape, EXPECTED);
    }

    fn check_hi_wide<F: Function + MathFunction>() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = Shape::<F>::new(&ctx, root).unwrap();
        const EXPECTED: &str = "
            .................................X..............................
            .................................X..............................
            .................................X..............................
            .................................X..........XX..................
            .................................X..........XX..................
            .................................X..............................
            .................................X..............................
            .................................XXXXXX.....XX..................
            .................................XXX..XX....XX..................
            .................................XX....XX...XX..................
            .................................X......X...XX..................
            .................................X......X...XX..................
            .................................X......X...XX..................
            .................................X......X...XX..................
            .................................X......X...XX..................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................
            ................................................................";
        Cfg {
            wide: true,
            ..Default::default()
        }
        .test(shape, EXPECTED);
    }

    fn check_hi_transformed<F: Function + MathFunction>() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = Shape::<F>::new(&ctx, root).unwrap();
        let mut mat = nalgebra::Matrix4::<f32>::identity();
        mat.prepend_translation_mut(&nalgebra::Vector3::new(0.5, 0.5, 0.0));
        mat.prepend_scaling_mut(0.5);
        let shape = shape.apply_transform(mat);
        const EXPECTED: &str = "
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX....................XXX.....
            .XXX...................XXXXX....
            .XXX...................XXXXX....
            .XXX...................XXXX.....
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX..XXXXXX............XXX.....
            .XXXXXXXXXXXXX..........XXX.....
            .XXXXXXXXXXXXXXX........XXX.....
            .XXXXXX....XXXXX........XXX.....
            .XXXXX.......XXXX.......XXX.....
            .XXXX.........XXX.......XXX.....
            .XXX..........XXXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            ................................";
        Cfg::default().test(shape, EXPECTED);
    }

    fn check_hi_bounded<F: Function + MathFunction>() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = Shape::<F>::new(&ctx, root).unwrap();
        const EXPECTED: &str = "
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX....................XXX.....
            .XXX...................XXXXX....
            .XXX...................XXXXX....
            .XXX...................XXXX.....
            .XXX............................
            .XXX............................
            .XXX............................
            .XXX..XXXXXX............XXX.....
            .XXXXXXXXXXXXX..........XXX.....
            .XXXXXXXXXXXXXXX........XXX.....
            .XXXXXX....XXXXX........XXX.....
            .XXXXX.......XXXX.......XXX.....
            .XXXX.........XXX.......XXX.....
            .XXX..........XXXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            .XXX...........XXX......XXX.....
            ................................";
        let view =
            View2::from_center_and_scale(nalgebra::Vector2::new(0.5, 0.5), 0.5);
        Cfg {
            view,
            ..Default::default()
        }
        .test(shape, EXPECTED);
    }

    fn check_quarter<F: Function + MathFunction>() {
        let (ctx, root) = Context::from_text(QUARTER.as_bytes()).unwrap();
        let shape = Shape::<F>::new(&ctx, root).unwrap();
        const EXPECTED: &str = "
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            .....XXXXXXXXXXX................
            .....XXXXXXXXXXX................
            ......XXXXXXXXXX................
            ......XXXXXXXXXX................
            ......XXXXXXXXXX................
            .......XXXXXXXXX................
            ........XXXXXXXX................
            .........XXXXXXX................
            ..........XXXXXX................
            ...........XXXXX................
            ..............XX................
            ................................
            ................................
            ................................
            ................................
            ................................";
        Cfg::default().test(shape, EXPECTED);
    }

    fn check_circle_var<F: Function + MathFunction>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let r2 = ctx.add(x2, y2).unwrap();
        let r = ctx.sqrt(r2).unwrap();
        let v = Var::new();
        let c = ctx.var(v);
        let root = ctx.sub(r, c).unwrap();
        let shape = Shape::<F>::new(&ctx, root).unwrap();
        const EXPECTED_075: &str = "
            ................................
            ................................
            ................................
            ................................
            ............XXXXXXXXX...........
            ..........XXXXXXXXXXXXX.........
            .........XXXXXXXXXXXXXXX........
            ........XXXXXXXXXXXXXXXXX.......
            .......XXXXXXXXXXXXXXXXXXX......
            ......XXXXXXXXXXXXXXXXXXXXX.....
            ......XXXXXXXXXXXXXXXXXXXXX.....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            .....XXXXXXXXXXXXXXXXXXXXXXX....
            ......XXXXXXXXXXXXXXXXXXXXX.....
            ......XXXXXXXXXXXXXXXXXXXXX.....
            .......XXXXXXXXXXXXXXXXXXX......
            ........XXXXXXXXXXXXXXXXX.......
            .........XXXXXXXXXXXXXXX........
            ..........XXXXXXXXXXXXX.........
            ............XXXXXXXXX...........
            ................................
            ................................
            ................................
            ................................
            ................................";
        let mut vars = ShapeVars::new();
        vars.insert(v.index().unwrap(), 0.75);
        Cfg {
            vars,
            ..Default::default()
        }
        .test(shape.clone(), EXPECTED_075);

        const EXPECTED_05: &str = "
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            .............XXXXXXX............
            ...........XXXXXXXXXXX..........
            ..........XXXXXXXXXXXXX.........
            ..........XXXXXXXXXXXXX.........
            .........XXXXXXXXXXXXXXX........
            .........XXXXXXXXXXXXXXX........
            .........XXXXXXXXXXXXXXX........
            .........XXXXXXXXXXXXXXX........
            .........XXXXXXXXXXXXXXX........
            .........XXXXXXXXXXXXXXX........
            .........XXXXXXXXXXXXXXX........
            ..........XXXXXXXXXXXXX.........
            ..........XXXXXXXXXXXXX.........
            ...........XXXXXXXXXXX..........
            .............XXXXXXX............
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................";
        let mut vars = ShapeVars::new();
        vars.insert(v.index().unwrap(), 0.5);
        Cfg {
            vars,
            ..Default::default()
        }
        .test(shape, EXPECTED_05);
    }

    macro_rules! render_tests {
        ($i:ident) => {
            mod $i {
                use super::*;
                #[test]
                fn vm() {
                    $i::<VmFunction>();
                }
                #[test]
                fn vm3() {
                    $i::<GenericVmFunction<3>>();
                }
                #[cfg(feature = "jit")]
                #[test]
                fn jit() {
                    $i::<$crate::jit::JitFunction>();
                }
            }
        };
    }

    render_tests!(check_hi);
    render_tests!(check_hi_wide);
    render_tests!(check_hi_transformed);
    render_tests!(check_hi_bounded);
    render_tests!(check_quarter);
    render_tests!(check_circle_var);

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
