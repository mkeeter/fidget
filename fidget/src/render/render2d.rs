//! Bitmap rendering
use crate::{
    eval::{
        float_slice::FloatSliceEval,
        interval::{Interval, IntervalEval},
        EvalFamily,
    },
    render::config::{RenderConfig, Tile},
    tape::Tape,
};
use nalgebra::{Matrix3, Point2, Vector2};
use std::sync::atomic::{AtomicUsize, Ordering};

////////////////////////////////////////////////////////////////////////////////

/// Configuration trait for rendering
pub trait RenderMode {
    /// Type of output pixel
    type Output: Default + Copy + Clone + Send;

    /// Decide whether to subdivide or fill an interval
    fn interval(i: Interval, depth: usize) -> Option<Self::Output>;

    /// Per-pixel drawing
    fn pixel(f: f32) -> Self::Output;
}

////////////////////////////////////////////////////////////////////////////////

/// Renderer that emits `DebugPixel`
pub struct DebugRenderMode;

impl RenderMode for DebugRenderMode {
    type Output = DebugPixel;
    fn interval(i: Interval, depth: usize) -> Option<DebugPixel> {
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
    fn pixel(f: f32) -> DebugPixel {
        if f < 0.0 {
            DebugPixel::Filled
        } else {
            DebugPixel::Empty
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum DebugPixel {
    EmptyTile,
    FilledTile,
    EmptySubtile,
    FilledSubtile,
    Empty,
    Filled,
    Invalid,
}

impl Default for DebugPixel {
    fn default() -> Self {
        DebugPixel::Invalid
    }
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
    fn interval(i: Interval, _depth: usize) -> Option<bool> {
        if i.upper() < 0.0 {
            Some(true)
        } else if i.lower() > 0.0 {
            Some(false)
        } else {
            None
        }
    }
    fn pixel(f: f32) -> bool {
        f < 0.0
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    out: Vec<f32>,
}

impl Scratch {
    fn new(size: usize) -> Self {
        Self {
            x: vec![0.0; size],
            y: vec![0.0; size],
            z: vec![0.0; size],
            out: vec![0.0; size],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

fn worker<I: EvalFamily, M: RenderMode>(
    mut i_handle: IntervalEval<I::IntervalEval>,
    tiles: &[Tile<2>],
    i: &AtomicUsize,
    config: &RenderConfig<2>,
) -> Vec<(Tile<2>, Vec<M::Output>)> {
    let mat = config.mat.matrix()
        * nalgebra::Matrix3::identity()
            .append_scaling(2.0 / config.image_size as f32)
            .append_translation(&Vector2::new(-1.0, -1.0));

    let mut out = vec![];
    let mut scratch =
        Scratch::new(config.tile_sizes.last().unwrap_or(&0).pow(2));
    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        let mut pixels =
            vec![M::Output::default(); config.tile_sizes[0].pow(2)];
        render_tile_recurse::<I, M>(
            &mut i_handle,
            &mut pixels,
            mat,
            config,
            0,
            tile,
            &mut scratch,
        );
        out.push((tile, pixels))
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

#[inline(never)]
fn render_tile_recurse<I: EvalFamily, M: RenderMode>(
    i_handle: &mut IntervalEval<I::IntervalEval>,
    out: &mut [M::Output],
    mat: Matrix3<f32>,
    config: &RenderConfig<2>,
    depth: usize,
    tile: Tile<2>,
    scratch: &mut Scratch,
) {
    let tile_size = config.tile_sizes[depth];

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
        let p = mat.transform_point(&p);
        x_min = x_min.min(p.x);
        x_max = x_max.max(p.x);
        y_min = y_min.min(p.y);
        y_max = y_max.max(p.y);
    }
    let x = Interval::new(x_min, x_max);
    let y = Interval::new(y_min, y_max);
    let z = Interval::new(0.0, 0.0);
    let i = i_handle.eval_i(x, y, z);

    let fill = M::interval(i, depth);

    if let Some(fill) = fill {
        for y in 0..tile_size {
            let start = config.tile_to_offset(tile, 0, y);
            out[start..][..tile_size].fill(fill);
        }
    } else if let Some(next_tile_size) = config.tile_sizes.get(depth + 1) {
        let sub_tape = i_handle.simplify(I::REG_LIMIT);
        let mut sub_jit = IntervalEval::from(sub_tape);
        let n = tile_size / next_tile_size;
        for j in 0..n {
            for i in 0..n {
                render_tile_recurse::<I, M>(
                    &mut sub_jit,
                    out,
                    mat,
                    config,
                    depth + 1,
                    config.new_tile([
                        tile.corner[0] + i * next_tile_size,
                        tile.corner[1] + j * next_tile_size,
                    ]),
                    scratch,
                );
            }
        }
    } else {
        let sub_tape = i_handle.simplify(I::REG_LIMIT);
        let mut sub_jit = FloatSliceEval::<I::FloatSliceEval>::from(sub_tape);

        let mut index = 0;
        for j in 0..tile_size {
            for i in 0..tile_size {
                let p = mat.transform_point(&Point2::new(
                    (tile.corner[0] + i) as f32,
                    (tile.corner[1] + j) as f32,
                ));
                scratch.x[index] = p.x;
                scratch.y[index] = p.y;
                index += 1;
            }
        }

        sub_jit.eval_s(&scratch.x, &scratch.y, &scratch.z, &mut scratch.out);

        let mut index = 0;
        for j in 0..tile_size {
            let o = config.tile_to_offset(tile, 0, j);
            for i in 0..tile_size {
                out[o + i] = M::pixel(scratch.out[index]);
                index += 1;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub fn render<I: EvalFamily, M: RenderMode>(
    tape: Tape,
    config: &RenderConfig<2>,
) -> Vec<M::Output> {
    assert!(config.image_size % config.tile_sizes[0] == 0);
    for i in 0..config.tile_sizes.len() - 1 {
        assert!(config.tile_sizes[i] % config.tile_sizes[i + 1] == 0);
    }

    let i_handle = IntervalEval::from(tape);
    let mut tiles = vec![];
    for i in 0..config.image_size / config.tile_sizes[0] {
        for j in 0..config.image_size / config.tile_sizes[0] {
            tiles.push(config.new_tile([
                i * config.tile_sizes[0],
                j * config.tile_sizes[0],
            ]));
        }
    }

    let index = AtomicUsize::new(0);
    let out = std::thread::scope(|s| {
        let mut handles = vec![];
        for _ in 0..config.threads {
            let i = i_handle.clone();
            handles.push(s.spawn(|| worker::<I, M>(i, &tiles, &index, config)));
        }
        let mut out = vec![];
        for h in handles {
            out.extend(h.join().unwrap().into_iter());
        }
        out
    });

    let mut image =
        vec![M::Output::default(); config.image_size * config.image_size];
    for (tile, data) in out.iter() {
        for j in 0..config.tile_sizes[0] {
            let y = j + tile.corner[1];
            let offset = (config.image_size - y - 1) * config.image_size
                + tile.corner[0];
            image[offset..][..config.tile_sizes[0]].copy_from_slice(
                &data[(j * config.tile_sizes[0])..][..config.tile_sizes[0]],
            );
        }
    }
    image
}
