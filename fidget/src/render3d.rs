//! Bitmap rendering
use crate::{
    eval::{
        interval::{Interval, IntervalFunc},
        EvalFamily,
    },
    tape::Tape,
};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct RenderConfig<const N: usize> {
    pub image_size: usize,
    pub tile_sizes: [usize; N],
    pub interval_subdiv: usize,
    pub threads: usize,

    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub scale: f32,
}

impl<const N: usize> RenderConfig<N> {
    fn pixel_to_pos(&self, p: usize) -> f32 {
        (2.0 * (p as f32) / (self.image_size as f32) - 1.0) * self.scale
    }
    fn tile_to_offset(&self, tile: Tile, x: usize, y: usize) -> usize {
        let x = tile.corner[0] % self.tile_sizes[0] + x;
        let y = tile.corner[1] % self.tile_sizes[0] + y;
        x + self.tile_sizes[0] * y
    }
}

#[derive(Copy, Clone, Debug)]
struct Tile {
    corner: [usize; 3],
}

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

fn worker<'a, I, const N: usize>(
    i_handle: &IntervalFunc<'a, <I as EvalFamily<'a>>::IntervalFunc>,
    tiles: &[Tile],
    i: &AtomicUsize,
    config: &RenderConfig<N>,
) -> Vec<(Tile, Vec<f32>)>
where
    for<'s> I: EvalFamily<'s>,
{
    let mut out = vec![];
    let mut pixels_rendered = 0;

    // Calculate maximum evaluation buffer size
    let mt = config.tile_sizes.last().cloned().unwrap_or(0);
    let mut scratch = Scratch::new(mt * mt * mt);

    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        let mut pixels = vec![
            f32::NEG_INFINITY;
            config.tile_sizes[0] * config.tile_sizes[0]
        ];
        render_tile_recurse::<I, N>(
            i_handle,
            &mut pixels,
            config,
            &config.tile_sizes,
            tile,
            &mut scratch,
            &mut pixels_rendered,
        );
        out.push((tile, pixels));
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

fn render_tile_recurse<'a, I, const N: usize>(
    handle: &IntervalFunc<'a, <I as EvalFamily<'a>>::IntervalFunc>,
    out: &mut [f32],
    config: &RenderConfig<N>,
    tile_sizes: &[usize],
    tile: Tile,
    scratch: &mut Scratch,
    pixels_rendered: &mut usize,
) where
    for<'s> I: EvalFamily<'s>,
{
    let mut eval = handle.get_evaluator();

    let x_min = config.pixel_to_pos(tile.corner[0]) + config.dx;
    let x_max = config.pixel_to_pos(tile.corner[0] + tile_sizes[0]) + config.dx;
    let y_min = config.pixel_to_pos(tile.corner[1]) + config.dy;
    let y_max = config.pixel_to_pos(tile.corner[1] + tile_sizes[0]) + config.dy;
    let z_min = config.pixel_to_pos(tile.corner[2]) + config.dz;
    let z_max = config.pixel_to_pos(tile.corner[2] + tile_sizes[0]) + config.dz;

    let x = Interval::new(x_min, x_max);
    let y = Interval::new(y_min, y_max);
    let z = Interval::new(z_min, z_max);
    let i = eval.eval_i_subdiv(x, y, z, config.interval_subdiv);

    let fill = if i.upper() < 0.0 {
        Some(z_max)
    } else if i.lower() > 0.0 {
        // Return early if this tile is completely empty
        return;
    } else {
        None
    };

    if let Some(fill) = fill {
        for x in 0..tile_sizes[0] {
            for y in 0..tile_sizes[0] {
                let i = config.tile_to_offset(tile, x, y);
                out[i] = out[i].max(fill);
            }
        }
    } else if let Some(next_tile_size) = tile_sizes.get(1) {
        let sub_tape = eval.simplify(I::REG_LIMIT);
        let sub_jit = I::from_tape_i(&sub_tape);
        let n = tile_sizes[0] / next_tile_size;
        for j in 0..n {
            for i in 0..n {
                for k in 0..n {
                    render_tile_recurse::<I, N>(
                        &sub_jit,
                        out,
                        config,
                        &tile_sizes[1..],
                        Tile {
                            corner: [
                                tile.corner[0] + i * next_tile_size,
                                tile.corner[1] + j * next_tile_size,
                                tile.corner[2] + k * next_tile_size,
                            ],
                        },
                        scratch,
                        pixels_rendered,
                    );
                }
            }
        }
    } else {
        let sub_tape = eval.simplify(I::REG_LIMIT);
        let sub_jit = I::from_tape_s(&sub_tape);

        let mut index = 0;
        for j in 0..tile_sizes[0] {
            let y = config.pixel_to_pos(tile.corner[1] + j) + config.dy;
            for i in 0..tile_sizes[0] {
                let x = config.pixel_to_pos(tile.corner[0] + i) + config.dx;
                for k in 0..tile_sizes[0] {
                    let z = config.pixel_to_pos(tile.corner[2] + k) + config.dz;
                    scratch.x[index] = x;
                    scratch.y[index] = y;
                    scratch.z[index] = z;
                    index += 1;
                }
            }
        }
        *pixels_rendered += tile_sizes[0] * tile_sizes[0] * tile_sizes[0];
        assert_eq!(index, scratch.x.len());

        let mut eval = sub_jit.get_evaluator();
        eval.eval_s(&scratch.x, &scratch.y, &scratch.z, &mut scratch.out);

        let mut index = 0;
        for j in 0..tile_sizes[0] {
            for i in 0..tile_sizes[0] {
                for k in 0..tile_sizes[0] {
                    let z = config.pixel_to_pos(tile.corner[2] + k) + config.dz;
                    let o = config.tile_to_offset(tile, i, j);
                    if scratch.out[index] < 0.0 {
                        out[o] = out[o].max(z);
                    }
                    index += 1;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub fn render<I, const N: usize>(
    tape: Tape,
    config: &RenderConfig<N>,
) -> Vec<f32>
where
    for<'s> I: EvalFamily<'s>,
{
    assert!(config.image_size % config.tile_sizes[0] == 0);
    for i in 0..config.tile_sizes.len() - 1 {
        assert!(config.tile_sizes[i] % config.tile_sizes[i + 1] == 0);
    }

    let i_handle = I::from_tape_i(&tape);
    let mut tiles = vec![];
    for i in 0..config.image_size / config.tile_sizes[0] {
        for j in 0..config.image_size / config.tile_sizes[0] {
            for k in 0..config.image_size / config.tile_sizes[0] {
                tiles.push(Tile {
                    corner: [
                        i * config.tile_sizes[0],
                        j * config.tile_sizes[0],
                        k * config.tile_sizes[0],
                    ],
                });
            }
        }
    }

    let index = AtomicUsize::new(0);
    let out = std::thread::scope(|s| {
        let mut handles = vec![];
        for _ in 0..config.threads {
            handles.push(
                s.spawn(|| worker::<I, N>(&i_handle, &tiles, &index, config)),
            );
        }
        let mut out = vec![];
        for h in handles {
            out.extend(h.join().unwrap().into_iter());
        }
        out
    });

    let mut image =
        vec![f32::NEG_INFINITY; config.image_size * config.image_size];
    for (tile, data) in out.iter() {
        let mut index = 0;
        for j in 0..config.tile_sizes[0] {
            let y = j + tile.corner[1];
            for i in 0..config.tile_sizes[0] {
                let x = i + tile.corner[0];
                let o = (config.image_size - y - 1) * config.image_size + x;
                image[o] = image[o].max(data[index]);
                index += 1;
            }
        }
    }
    image
}
