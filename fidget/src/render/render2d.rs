//! 2D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    eval::Function,
    render::config::{AlignedRenderConfig, Queue, RenderConfig, Tile},
    shape::{Shape, ShapeBulkEval, ShapeTracingEval},
    types::Interval,
};
use nalgebra::Point2;

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
    config: &'a AlignedRenderConfig<2>,
    scratch: Scratch,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,

    /// Spare tape storage for reuse
    tape_storage: Vec<F::TapeStorage>,

    /// Spare shape storage for reuse
    shape_storage: Vec<F::Storage>,

    /// Workspace for shape simplification
    workspace: F::Workspace,

    image: Vec<M::Output>,
}

impl<F: Function, M: RenderMode> Worker<'_, F, M> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
        depth: usize,
        tile: Tile<2>,
    ) {
        let tile_size = self.config.tile_sizes[depth];

        // Brute-force way to find the (interval) bounding box of the region
        let base = Point2::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(0.0, 0.0);

        let (i, simplify) = self
            .eval_interval
            .eval(shape.i_tape(&mut self.tape_storage), x, y, z)
            .unwrap();

        match M::interval(i, depth) {
            IntervalAction::Fill(fill) => {
                for y in 0..tile_size {
                    let start = self.config.tile_to_offset(tile, 0, y);
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

                    let mut i = self.config.tile_to_offset(tile, 0, y);
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
                        depth + 1,
                        self.config.new_tile([
                            tile.corner[0] + i * next_tile_size,
                            tile.corner[1] + j * next_tile_size,
                        ]),
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

        let out = self
            .eval_float_slice
            .eval(
                shape.f_tape(&mut self.tape_storage),
                &self.scratch.x,
                &self.scratch.y,
                &self.scratch.z,
            )
            .unwrap();

        let mut index = 0;
        for j in 0..tile_size {
            let o = self.config.tile_to_offset(tile, 0, j);
            for i in 0..tile_size {
                self.image[o + i] = M::pixel(out[index]);
                index += 1;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

fn worker<F: Function, M: RenderMode>(
    mut shape: RenderHandle<F>,
    queue: &Queue<2>,
    config: &AlignedRenderConfig<2>,
) -> Vec<(Tile<2>, Vec<M::Output>)> {
    let mut out = vec![];
    let scratch = Scratch::new(config.tile_sizes.last().unwrap_or(&0).pow(2));

    let mut w: Worker<F, M> = Worker {
        scratch,
        image: vec![],
        config,
        eval_float_slice: Default::default(),
        eval_interval: Default::default(),
        tape_storage: vec![],
        shape_storage: vec![],
        workspace: Default::default(),
    };
    while let Some(tile) = queue.next() {
        w.image = vec![M::Output::default(); config.tile_sizes[0].pow(2)];
        w.render_tile_recurse(&mut shape, 0, tile);
        let pixels = std::mem::take(&mut w.image);
        out.push((tile, pixels))
    }
    out
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
    config: &RenderConfig<2>,
) -> Vec<M::Output> {
    let (config, mat) = config.align();
    assert!(config.image_size % config.tile_sizes[0] == 0);
    for i in 0..config.tile_sizes.len() - 1 {
        assert!(config.tile_sizes[i] % config.tile_sizes[i + 1] == 0);
    }

    // Convert to a 4x4 matrix and apply to the shape
    let mat = mat.insert_row(2, 0.0);
    let mat = mat.insert_column(2, 0.0);
    let shape = shape.apply_transform(mat);

    render_inner::<_, M>(shape, config)
}

fn render_inner<F: Function, M: RenderMode + Sync>(
    shape: Shape<F>,
    config: AlignedRenderConfig<2>,
) -> Vec<M::Output> {
    let mut tiles = vec![];
    for i in 0..config.image_size / config.tile_sizes[0] {
        for j in 0..config.image_size / config.tile_sizes[0] {
            tiles.push(config.new_tile([
                i * config.tile_sizes[0],
                j * config.tile_sizes[0],
            ]));
        }
    }

    let queue = Queue::new(tiles);
    let threads = config.threads();

    let mut rh = RenderHandle::new(shape);
    let _ = rh.i_tape(&mut vec![]); // populate i_tape before cloning

    let out: Vec<_> = if threads == 1 {
        worker::<F, M>(rh, &queue, &config).into_iter().collect()
    } else {
        #[cfg(target_arch = "wasm32")]
        unreachable!("multithreaded rendering is not supported on wasm32");

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::scope(|s| {
            let mut handles = vec![];
            for _ in 0..threads {
                let rh = rh.clone();
                handles.push(s.spawn(|| worker::<F, M>(rh, &queue, &config)));
            }
            let mut out = vec![];
            for h in handles {
                out.extend(h.join().unwrap().into_iter());
            }
            out
        })
    };

    let mut image = vec![M::Output::default(); config.orig_image_size.pow(2)];
    for (tile, data) in out.iter() {
        let mut index = 0;
        for j in 0..config.tile_sizes[0] {
            let y = j + tile.corner[1];
            for i in 0..config.tile_sizes[0] {
                let x = i + tile.corner[0];
                if y < config.orig_image_size && x < config.orig_image_size {
                    let o = (config.orig_image_size - y - 1)
                        * config.orig_image_size
                        + x;
                    image[o] = data[index];
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
        shape::{Bounds, Shape},
        vm::{GenericVmFunction, VmFunction},
        Context,
    };

    const HI: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/hi.vm"));
    const QUARTER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/quarter.vm"
    ));

    fn render_and_compare_with_bounds<F: Function>(
        shape: Shape<F>,
        expected: &'static str,
        bounds: Bounds<2>,
    ) {
        let cfg = RenderConfig::<2> {
            image_size: 32,
            bounds,
            ..RenderConfig::default()
        };
        let out = cfg.run::<_, BitRenderMode>(shape).unwrap();
        let mut img_str = String::new();
        for (i, b) in out.iter().enumerate() {
            if i % 32 == 0 {
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

    fn render_and_compare<F: Function>(
        shape: Shape<F>,
        expected: &'static str,
    ) {
        render_and_compare_with_bounds(shape, expected, Bounds::default())
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
        render_and_compare(shape, EXPECTED);
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
        render_and_compare(shape, EXPECTED);
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
        render_and_compare_with_bounds(
            shape,
            EXPECTED,
            Bounds {
                center: nalgebra::Vector2::new(0.5, 0.5),
                size: 0.5,
            },
        );
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
        render_and_compare(shape, EXPECTED);
    }

    #[test]
    fn render_hi_vm() {
        check_hi::<VmFunction>();
    }

    #[test]
    fn render_hi_vm3() {
        check_hi::<GenericVmFunction<3>>();
    }

    #[cfg(feature = "jit")]
    #[test]
    fn render_hi_jit() {
        check_hi::<crate::jit::JitFunction>();
    }

    #[test]
    fn render_hi_transformed_vm() {
        check_hi_transformed::<VmFunction>();
    }

    #[test]
    fn render_hi_transformed_vm3() {
        check_hi_transformed::<GenericVmFunction<3>>();
    }

    #[cfg(feature = "jit")]
    #[test]
    fn render_hi_transformed_jit() {
        check_hi_transformed::<crate::jit::JitFunction>();
    }

    #[test]
    fn render_hi_bounded_vm() {
        check_hi_bounded::<VmFunction>();
    }

    #[test]
    fn render_hi_bounded_vm3() {
        check_hi_bounded::<GenericVmFunction<3>>();
    }

    #[cfg(feature = "jit")]
    #[test]
    fn render_hi_bounded_jit() {
        check_hi_bounded::<crate::jit::JitFunction>();
    }

    #[test]
    fn render_quarter_vm() {
        check_quarter::<VmFunction>();
    }

    #[test]
    fn render_quarter_vm3() {
        check_quarter::<GenericVmFunction<3>>();
    }

    #[cfg(feature = "jit")]
    #[test]
    fn render_quarter_jit() {
        check_quarter::<crate::jit::JitFunction>();
    }
}
