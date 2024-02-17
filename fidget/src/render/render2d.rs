//! 2D bitmap rendering / rasterization
use crate::{
    eval::{types::Interval, BulkEvaluator, Shape, Tape, TracingEvaluator},
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

/// A specific shape, and its associated tapes
struct ShapeAndTape<S: Shape> {
    shape: S,

    /// Interval tape to evaluate the given shape
    ///
    /// This is shareable because we use the same tape for all threads when they
    /// start evaluating the root shape (which is the most expensive to build a
    /// tape for, since it's the largest).
    i_tape: Option<Arc<<S::IntervalEval as TracingEvaluator>::Tape>>,
    f_tape: Option<<S::FloatSliceEval as BulkEvaluator>::Tape>,
}

impl<S: Shape> ShapeAndTape<S> {
    fn i_tape(
        &mut self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::IntervalEval as TracingEvaluator>::Tape {
        self.i_tape.get_or_insert_with(|| {
            Arc::new(
                self.shape.interval_tape(storage.pop().unwrap_or_default()),
            )
        })
    }
    fn f_tape(
        &mut self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::FloatSliceEval as BulkEvaluator>::Tape {
        self.f_tape.get_or_insert_with(|| {
            self.shape
                .float_slice_tape(storage.pop().unwrap_or_default())
        })
    }
}

impl<S: Shape, M: RenderMode> Worker<'_, S, M> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut ShapeAndTape<S>,
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

        let mut sub_tape = if let Some(data) = simplify.as_ref() {
            let s = self.shape_storage.pop().unwrap_or_default();
            Some(ShapeAndTape {
                shape: shape
                    .shape
                    .simplify(data, s, &mut self.workspace)
                    .unwrap(),
                i_tape: None,
                f_tape: None,
            })
        } else {
            None
        };
        let sub_tape_ref = sub_tape
            .as_mut()
            .filter(|t| t.shape.size() < shape.shape.size())
            .unwrap_or(shape);

        if let Some(next_tile_size) = self.config.tile_sizes.get(depth + 1) {
            let n = tile_size / next_tile_size;
            for j in 0..n {
                for i in 0..n {
                    self.render_tile_recurse(
                        sub_tape_ref,
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
            self.render_tile_pixels(sub_tape_ref, tile_size, tile, mode);
        }
        if let Some(sub_tape) = sub_tape {
            if let Some(i_tape) = sub_tape.i_tape {
                if let Ok(v) = Arc::try_unwrap(i_tape) {
                    self.tape_storage.push(v.recycle());
                }
            }
            if let Some(f_tape) = sub_tape.f_tape {
                self.tape_storage.push(f_tape.recycle());
            }
            if let Some(s) = sub_tape.shape.recycle() {
                self.shape_storage.push(s);
            }
        }
    }

    fn render_tile_pixels(
        &mut self,
        shape: &mut ShapeAndTape<S>,
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
    mut shape: ShapeAndTape<S>,
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
/// This function is parameterized by both evaluator family (which determines
/// how we perform evaluation) and render mode (which tells us how to color in
/// the resulting pixels).
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

    let i_tape = Some(Arc::new(shape.interval_tape(Default::default())));
    let queue = Queue::new(tiles);
    let out = std::thread::scope(|s| {
        let mut handles = vec![];
        for _ in 0..config.threads {
            let shape = ShapeAndTape {
                shape: shape.clone(),
                i_tape: i_tape.clone(),
                f_tape: None,
            };
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
