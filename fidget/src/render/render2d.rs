//! 2D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    eval::Function,
    render::config::{ImageRenderConfig, Tile},
    render::ThreadCount,
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    types::Interval,
};
use nalgebra::{Point2, Vector2};

////////////////////////////////////////////////////////////////////////////////

/// Response type for [`RenderMode::interval`]
pub enum IntervalAction<T> {
    Fill(T),
    Interpolate,
    Recurse,
}

/// Configuration trait for rendering
pub trait RenderMode {
    /// Type of output pixel
    type Output: Default + Copy + Clone + Send;

    /// Decide whether to subdivide or fill an interval
    fn interval(i: Interval, depth: usize) -> IntervalAction<Self::Output>;

    /// Per-pixel drawing
    fn pixel(f: f32) -> Self::Output;
}

////////////////////////////////////////////////////////////////////////////////

/// Renderer that emits `DebugPixel`
pub struct DebugRenderMode;

impl RenderMode for DebugRenderMode {
    type Output = DebugPixel;
    fn interval(i: Interval, depth: usize) -> IntervalAction<DebugPixel> {
        if i.upper() < 0.0 {
            if depth > 1 {
                IntervalAction::Fill(DebugPixel::FilledSubtile)
            } else {
                IntervalAction::Fill(DebugPixel::FilledTile)
            }
        } else if i.lower() > 0.0 {
            if depth > 1 {
                IntervalAction::Fill(DebugPixel::EmptySubtile)
            } else {
                IntervalAction::Fill(DebugPixel::EmptyTile)
            }
        } else {
            IntervalAction::Recurse
        }
    }
    fn pixel(f: f32) -> DebugPixel {
        if f < 0.0 {
            DebugPixel::Filled
        } else {
            DebugPixel::Empty
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub enum DebugPixel {
    EmptyTile,
    FilledTile,
    EmptySubtile,
    FilledSubtile,
    Empty,
    Filled,
    #[default]
    Invalid,
}

impl DebugPixel {
    #[inline]
    pub fn as_debug_color(&self) -> [u8; 4] {
        match self {
            DebugPixel::EmptyTile => [50, 0, 0, 255],
            DebugPixel::FilledTile => [255, 0, 0, 255],
            DebugPixel::EmptySubtile => [0, 50, 0, 255],
            DebugPixel::FilledSubtile => [0, 255, 0, 255],
            DebugPixel::Empty => [0, 0, 0, 255],
            DebugPixel::Filled => [255, 255, 255, 255],
            DebugPixel::Invalid => panic!(),
        }
    }

    #[inline]
    pub fn is_filled(&self) -> bool {
        match self {
            DebugPixel::EmptyTile
            | DebugPixel::EmptySubtile
            | DebugPixel::Empty => false,
            DebugPixel::FilledTile
            | DebugPixel::FilledSubtile
            | DebugPixel::Filled => true,
            DebugPixel::Invalid => panic!(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Renderer that emits `bool`
pub struct BitRenderMode;

impl RenderMode for BitRenderMode {
    type Output = bool;
    fn interval(i: Interval, _depth: usize) -> IntervalAction<bool> {
        if i.upper() < 0.0 {
            IntervalAction::Fill(true)
        } else if i.lower() > 0.0 {
            IntervalAction::Fill(false)
        } else {
            IntervalAction::Recurse
        }
    }
    fn pixel(f: f32) -> bool {
        f < 0.0
    }
}

/// Pixel-perfect render mode which mimics many SDF demos on ShaderToy
///
/// This mode recurses down to individual pixels, so it doesn't take advantage
/// of skipping empty / full regions; use [`SdfRenderMode`] for a
/// faster-but-approximate visualization.
pub struct SdfPixelRenderMode;

impl RenderMode for SdfPixelRenderMode {
    type Output = [u8; 3];
    fn interval(_i: Interval, _depth: usize) -> IntervalAction<[u8; 3]> {
        IntervalAction::Recurse
    }
    fn pixel(f: f32) -> [u8; 3] {
        let r = 1.0 - 0.1f32.copysign(f);
        let g = 1.0 - 0.4f32.copysign(f);
        let b = 1.0 - 0.7f32.copysign(f);

        let dim = 1.0 - (-4.0 * f.abs()).exp(); // dimming near 0
        let bands = 0.8 + 0.2 * (140.0 * f).cos(); // banding

        let smoothstep = |edge0: f32, edge1: f32, x: f32| {
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let mix = |x: f32, y: f32, a: f32| x * (1.0 - a) + y * a;

        let run = |v: f32| {
            let mut v = v * dim * bands;
            v = mix(v, 1.0, 1.0 - smoothstep(0.0, 0.015, f.abs()));
            v = mix(v, 1.0, 1.0 - smoothstep(0.0, 0.005, f.abs()));
            (v.clamp(0.0, 1.0) * 255.0) as u8
        };

        [run(r), run(g), run(b)]
    }
}

/// Fast rendering mode which mimics many SDF demos on ShaderToy
///
/// Unlike [`SdfPixelRenderMode`], this mode uses linear interpolation when
/// evaluating empty or full regions, which is significantly faster.
pub struct SdfRenderMode;

impl RenderMode for SdfRenderMode {
    type Output = [u8; 3];
    fn interval(i: Interval, _depth: usize) -> IntervalAction<[u8; 3]> {
        if i.upper() < 0.0 || i.lower() > 0.0 {
            IntervalAction::Interpolate
        } else {
            IntervalAction::Recurse
        }
    }
    fn pixel(f: f32) -> [u8; 3] {
        SdfPixelRenderMode::pixel(f)
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

////////////////////////////////////////////////////////////////////////////////

/// Per-thread worker
struct Worker<'a, F: Function, M: RenderMode> {
    config: &'a ImageRenderConfig,
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
    image: Vec<M::Output>,
}

impl<F: Function, M: RenderMode> Worker<'_, F, M> {
    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        tile: Tile<2>,
    ) -> Vec<M::Output> {
        self.image =
            vec![M::Output::default(); self.config.tile_sizes[0].pow(2)];
        self.render_tile_recurse(shape, vars, 0, tile);
        std::mem::take(&mut self.image)
    }

    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: Tile<2>,
    ) {
        let tile_size = self.config.tile_sizes[depth];

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

        match M::interval(i, depth) {
            IntervalAction::Fill(fill) => {
                for y in 0..tile_size {
                    let start = self
                        .config
                        .tile_sizes
                        .pixel_offset(tile.add(Vector2::new(0, y)));
                    self.image[start..][..tile_size].fill(fill);
                }
                return;
            }
            IntervalAction::Interpolate => {
                let xs = [x.lower(), x.lower(), x.upper(), x.upper()];
                let ys = [y.lower(), y.upper(), y.lower(), y.upper()];
                let zs = [0.0; 4];
                let vs = self
                    .eval_float_slice
                    .eval(shape.f_tape(&mut self.tape_storage), &xs, &ys, &zs)
                    .unwrap();

                // Bilinear interpolation on a per-pixel basis
                for y in 0..tile_size {
                    // Y interpolation
                    let y_frac = (y as f32 - 1.0) / (tile_size as f32);
                    let v0 = vs[0] * (1.0 - y_frac) + vs[1] * y_frac;
                    let v1 = vs[2] * (1.0 - y_frac) + vs[3] * y_frac;

                    let mut i = self
                        .config
                        .tile_sizes
                        .pixel_offset(tile.add(Vector2::new(0, y)));
                    for x in 0..tile_size {
                        // X interpolation
                        let x_frac = (x as f32 - 1.0) / (tile_size as f32);
                        let v = v0 * (1.0 - x_frac) + v1 * x_frac;

                        // Write out the pixel
                        self.image[i] = M::pixel(v);
                        i += 1;
                    }
                }
                return;
            }
            IntervalAction::Recurse => (), // keep going
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

        if let Some(next_tile_size) = self.config.tile_sizes.get(depth + 1) {
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
            let o = self
                .config
                .tile_sizes
                .pixel_offset(tile.add(Vector2::new(0, j)));
            for i in 0..tile_size {
                self.image[o + i] = M::pixel(out[index]);
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
/// This function is parameterized by both shape type (which determines how we
/// perform evaluation) and render mode (which tells us how to color in the
/// resulting pixels).
pub fn render<F: Function, M: RenderMode + Sync>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &ImageRenderConfig,
) -> Vec<M::Output> {
    // Convert to a 4x4 matrix and apply to the shape
    let mat = config.mat();
    let mat = mat.insert_row(2, 0.0);
    let mat = mat.insert_column(2, 0.0);
    let shape = shape.apply_transform(mat);

    render_inner::<_, M>(shape, vars, config)
}

fn render_inner<F: Function, M: RenderMode + Sync>(
    shape: Shape<F>,
    vars: &ShapeVars<f32>,
    config: &ImageRenderConfig,
) -> Vec<M::Output> {
    let mut tiles = vec![];
    let t = config.tile_sizes[0];
    let width = config.image_size.width() as usize;
    let height = config.image_size.height() as usize;
    for i in 0..width.div_ceil(t) {
        for j in 0..height.div_ceil(t) {
            tiles.push(Tile::new(Point2::new(
                i * config.tile_sizes[0],
                j * config.tile_sizes[0],
            )));
        }
    }

    let mut rh = RenderHandle::new(shape);
    let _ = rh.i_tape(&mut vec![]); // populate i_tape before cloning
    let init = || {
        let scratch = Scratch::new(config.tile_sizes.last().pow(2));
        let rh = rh.clone();

        let worker = Worker::<F, M> {
            scratch,
            image: vec![],
            config,
            eval_float_slice: Default::default(),
            eval_interval: Default::default(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        };
        (worker, rh)
    };

    let out: Vec<_> = match config.threads {
        ThreadCount::One => {
            let (mut worker, mut rh) = init();
            tiles
                .into_iter()
                .map(|tile| {
                    let pixels = worker.render_tile(&mut rh, vars, tile);
                    (tile, pixels)
                })
                .collect()
        }

        #[cfg(not(target_arch = "wasm32"))]
        ThreadCount::Many(v) => {
            use rayon::prelude::*;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(v.get())
                .build()
                .expect("could not build thread pool");
            pool.install(|| {
                tiles
                    .into_par_iter()
                    .map_init(init, |(w, rh), tile| {
                        let pixels = w.render_tile(rh, vars, tile);
                        (tile, pixels)
                    })
                    .collect()
            })
        }
    };

    let mut image = vec![M::Output::default(); width * height];
    for (tile, data) in out.iter() {
        let mut index = 0;
        for j in 0..config.tile_sizes[0] {
            let y = j + tile.corner.y;
            for i in 0..config.tile_sizes[0] {
                let x = i + tile.corner.x;
                if y < height && x < width {
                    image[y * width + x] = data[index];
                }
                index += 1;
            }
        }
    }
    image
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        eval::{Function, MathFunction},
        render::{ImageSize, View2},
        shape::Shape,
        var::Var,
        vm::{GenericVmFunction, VmFunction},
        Context,
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
            let out = cfg.run_with_vars::<_, BitRenderMode>(shape, &self.vars);
            let mut img_str = String::new();
            for (i, b) in out.iter().enumerate() {
                if i % width as usize == 0 {
                    img_str += "\n            ";
                }
                img_str.push(if *b { 'X' } else { '.' });
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
}
