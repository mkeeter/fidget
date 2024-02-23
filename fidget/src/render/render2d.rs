//! 2D bitmap rendering / rasterization
use super::RenderHandle;
use crate::{
    eval::{types::Interval, BulkEvaluator, Shape, TracingEvaluator},
    render::config::{AlignedRenderConfig, Queue, RenderConfig, Tile},
};
use nalgebra::{Point2, Vector2};
use std::sync::Arc;

////////////////////////////////////////////////////////////////////////////////

/// Configuration trait for rendering
pub trait RenderMode {
    /// Type of output pixel
    type Output: Default + Copy + Clone + Send;

    /// Decide whether to subdivide or fill an interval
    fn interval(&self, i: Interval, depth: usize) -> Option<Self::Output>;

    /// Per-pixel drawing
    fn pixel(&self, f: f32) -> Self::Output;
}

////////////////////////////////////////////////////////////////////////////////

/// Renderer that emits `DebugPixel`
pub struct DebugRenderMode;

impl RenderMode for DebugRenderMode {
    type Output = DebugPixel;
    fn interval(&self, i: Interval, depth: usize) -> Option<DebugPixel> {
        if i.upper() < 0.0 {
            if depth > 1 {
                Some(DebugPixel::FilledSubtile)
            } else {
                Some(DebugPixel::FilledTile)
            }
        } else if i.lower() > 0.0 {
            if depth > 1 {
                Some(DebugPixel::EmptySubtile)
            } else {
                Some(DebugPixel::EmptyTile)
            }
        } else {
            None
        }
    }
    fn pixel(&self, f: f32) -> DebugPixel {
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
    fn interval(&self, i: Interval, _depth: usize) -> Option<bool> {
        if i.upper() < 0.0 {
            Some(true)
        } else if i.lower() > 0.0 {
            Some(false)
        } else {
            None
        }
    }
    fn pixel(&self, f: f32) -> bool {
        f < 0.0
    }
}

/// Rendering mode which mimicks many SDF demos on ShaderToy
pub struct SdfRenderMode;

impl RenderMode for SdfRenderMode {
    type Output = [u8; 3];
    fn interval(&self, _i: Interval, _depth: usize) -> Option<[u8; 3]> {
        None // always recurse
    }
    fn pixel(&self, f: f32) -> [u8; 3] {
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
struct Worker<'a, S: Shape, M: RenderMode> {
    config: &'a AlignedRenderConfig<2>,
    scratch: Scratch,

    eval_float_slice: S::FloatSliceEval,
    eval_interval: S::IntervalEval,

    /// Spare tape storage for reuse
    tape_storage: Vec<S::TapeStorage>,

    /// Spare shape storage for reuse
    shape_storage: Vec<S::Storage>,

    /// Workspace for shape simplification
    workspace: S::Workspace,

    image: Vec<M::Output>,
}

impl<S: Shape, M: RenderMode> Worker<'_, S, M> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<S>,
        depth: usize,
        tile: Tile<2>,
        mode: &M,
    ) {
        let tile_size = self.config.tile_sizes[depth];

        // Brute-force way to find the (interval) bounding box of the region
        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        let base = Point2::from(tile.corner);
        for i in 0..4 {
            let offset = Vector2::new(
                if (i & 1) == 0 { 0 } else { tile_size },
                if (i & 2) == 0 { 0 } else { tile_size },
            );
            let p = (base + offset).cast::<f32>();
            let p = self.config.mat.transform_point(&p);
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
        let x = Interval::new(x_min, x_max);
        let y = Interval::new(y_min, y_max);
        let z = Interval::new(0.0, 0.0);

        let (i, simplify) = self
            .eval_interval
            .eval(shape.i_tape(&mut self.tape_storage), x, y, z, &[])
            .unwrap();

        let fill = mode.interval(i, depth);

        if let Some(fill) = fill {
            for y in 0..tile_size {
                let start = self.config.tile_to_offset(tile, 0, y);
                self.image[start..][..tile_size].fill(fill);
            }
            return;
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
                        mode,
                    );
                }
            }
        } else {
            self.render_tile_pixels(sub_tape, tile_size, tile, mode);
        }
    }

    fn render_tile_pixels(
        &mut self,
        shape: &mut RenderHandle<S>,
        tile_size: usize,
        tile: Tile<2>,
        mode: &M,
    ) {
        let mut index = 0;
        for j in 0..tile_size {
            for i in 0..tile_size {
                let p = self.config.mat.transform_point(&Point2::new(
                    (tile.corner[0] + i) as f32,
                    (tile.corner[1] + j) as f32,
                ));
                self.scratch.x[index] = p.x;
                self.scratch.y[index] = p.y;
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
                &[],
            )
            .unwrap();

        let mut index = 0;
        for j in 0..tile_size {
            let o = self.config.tile_to_offset(tile, 0, j);
            for i in 0..tile_size {
                self.image[o + i] = mode.pixel(out[index]);
                index += 1;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

fn worker<S: Shape, M: RenderMode>(
    mut shape: RenderHandle<S>,
    queue: &Queue<2>,
    config: &AlignedRenderConfig<2>,
    mode: &M,
) -> Vec<(Tile<2>, Vec<M::Output>)> {
    let mut out = vec![];
    let scratch = Scratch::new(config.tile_sizes.last().unwrap_or(&0).pow(2));

    let mut w: Worker<S, M> = Worker {
        scratch,
        image: vec![],
        config,
        eval_float_slice: S::FloatSliceEval::new(),
        eval_interval: S::IntervalEval::new(),
        tape_storage: vec![],
        shape_storage: vec![],
        workspace: Default::default(),
    };
    while let Some(tile) = queue.next() {
        w.image = vec![M::Output::default(); config.tile_sizes[0].pow(2)];
        w.render_tile_recurse(&mut shape, 0, tile, mode);
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
pub fn render<S: Shape, M: RenderMode + Sync>(
    shape: S,
    config: &RenderConfig<2>,
    mode: &M,
) -> Vec<M::Output> {
    let config = config.align();
    assert!(config.image_size % config.tile_sizes[0] == 0);
    for i in 0..config.tile_sizes.len() - 1 {
        assert!(config.tile_sizes[i] % config.tile_sizes[i + 1] == 0);
    }

    let mut tiles = vec![];
    for i in 0..config.image_size / config.tile_sizes[0] {
        for j in 0..config.image_size / config.tile_sizes[0] {
            tiles.push(config.new_tile([
                i * config.tile_sizes[0],
                j * config.tile_sizes[0],
            ]));
        }
    }

    let i_tape = Arc::new(shape.interval_tape(Default::default()));
    let queue = Queue::new(tiles);
    let out = std::thread::scope(|s| {
        let mut handles = vec![];
        for _ in 0..config.threads {
            let shape = RenderHandle::new(shape.clone(), i_tape.clone());
            handles
                .push(s.spawn(|| worker::<S, M>(shape, &queue, &config, mode)));
        }
        let mut out = vec![];
        for h in handles {
            out.extend(h.join().unwrap().into_iter());
        }
        out
    });

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
        eval::{MathShape, Shape},
        vm::{GenericVmShape, VmShape},
        Context,
    };

    const HI: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/hi.vm"));
    const QUARTER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/quarter.vm"
    ));

    fn render_and_compare<S: Shape>(shape: S, expected: &'static str) {
        let cfg = RenderConfig::<2> {
            image_size: 32,
            ..RenderConfig::default()
        };
        let out = cfg.run(shape, &BitRenderMode).unwrap();
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

    fn check_hi<S: Shape + MathShape>() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = S::new(&ctx, root).unwrap();
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

    fn check_quarter<S: Shape + MathShape>() {
        let (ctx, root) = Context::from_text(QUARTER.as_bytes()).unwrap();
        let shape = S::new(&ctx, root).unwrap();
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
        check_hi::<VmShape>();
    }

    #[test]
    fn render_hi_vm3() {
        check_hi::<GenericVmShape<3>>();
    }

    #[cfg(feature = "jit")]
    #[test]
    fn render_hi_jit() {
        check_hi::<crate::jit::JitShape>();
    }

    #[test]
    fn render_quarter_vm() {
        check_quarter::<VmShape>();
    }

    #[test]
    fn render_quarter_vm3() {
        check_quarter::<GenericVmShape<3>>();
    }

    #[cfg(feature = "jit")]
    #[test]
    fn render_quarter_jit() {
        check_quarter::<crate::jit::JitShape>();
    }
}
