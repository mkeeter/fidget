//! Bitmap rendering
use crate::{
    eval::{Interval, IntervalEval, IntervalFunc, VecEval, VecFunc},
    tape::Tape,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Copy, Clone, Debug)]
pub enum Pixel {
    EmptyTile,
    FilledTile,
    EmptySubtile,
    FilledSubtile,
    Empty,
    Filled,
}

impl Pixel {
    pub fn as_color(&self) -> [u8; 4] {
        match self {
            Pixel::EmptyTile => [50, 0, 0, 255],
            Pixel::FilledTile => [255, 0, 0, 255],
            Pixel::EmptySubtile => [0, 50, 0, 255],
            Pixel::FilledSubtile => [0, 255, 0, 255],
            Pixel::Empty => [0, 0, 0, 255],
            Pixel::Filled => [255, 255, 255, 255],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct RenderConfig {
    pub image_size: usize,
    pub tile_size: usize,
    pub subtile_size: usize,
    pub interval_subdiv: usize,
    pub threads: usize,
}

impl RenderConfig {
    fn pixel_to_pos(&self, p: usize) -> f32 {
        2.0 * (p as f32) / (self.image_size as f32) - 1.0
    }
}

#[derive(Copy, Clone, Debug)]
struct Tile {
    corner: [usize; 2],
}

////////////////////////////////////////////////////////////////////////////////

fn worker<'a, 'b, I: IntervalFunc<'a>, V: VecFunc<'b>>(
    i_handle: &I,
    tiles: &[Tile],
    i: &AtomicUsize,
    config: &RenderConfig,
) -> Vec<(Tile, Vec<Pixel>)> {
    let mut out = vec![];
    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        let mut pixels = vec![None; config.tile_size * config.tile_size];
        render_tile_recurse::<I, V>(
            i_handle,
            &mut pixels,
            config,
            &[config.tile_size, config.subtile_size],
            tile,
        );
        let pixels = pixels.into_iter().map(Option::unwrap).collect();
        out.push((tile, pixels))
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

fn render_tile_recurse<'a, 'b, I: IntervalFunc<'a>, V: VecFunc<'b>>(
    handle: &I,
    out: &mut [Option<Pixel>],
    config: &RenderConfig,
    tile_sizes: &[usize],
    tile: Tile,
) {
    let mut eval = handle.get_evaluator();

    let x_min = config.pixel_to_pos(tile.corner[0]);
    let x_max = config.pixel_to_pos(tile.corner[0] + tile_sizes[0]);
    let y_min = config.pixel_to_pos(tile.corner[1]);
    let y_max = config.pixel_to_pos(tile.corner[1] + tile_sizes[0]);

    let (i, token) = eval.eval_i(
        Interval {
            lower: x_min,
            upper: x_max,
        },
        Interval {
            lower: y_min,
            upper: y_max,
        },
        Interval {
            lower: 0.0,
            upper: 0.0,
        },
        // TODO: config.interval_subdiv,
    );

    let fill = if i.upper < 0.0 {
        if tile_sizes.len() > 1 {
            Some(Pixel::FilledTile)
        } else {
            Some(Pixel::FilledSubtile)
        }
    } else if i.lower > 0.0 {
        if tile_sizes.len() > 1 {
            Some(Pixel::EmptyTile)
        } else {
            Some(Pixel::EmptySubtile)
        }
    } else {
        None
    };

    if let Some(fill) = fill {
        for y in 0..tile_sizes[0] {
            for x in 0..tile_sizes[0] {
                out[x
                    + (tile.corner[0] % config.tile_size)
                    + (y + (tile.corner[1] % config.tile_size))
                        * config.tile_size] = Some(fill)
            }
        }
    } else if let Some(next_tile_size) = tile_sizes.get(1) {
        let sub_tape = token.simplify();
        let sub_jit = I::from_tape(&sub_tape);
        let n = tile_sizes[0] / next_tile_size;
        for j in 0..n {
            for i in 0..n {
                render_tile_recurse::<I::Recurse<'_>, V>(
                    &sub_jit,
                    out,
                    config,
                    &tile_sizes[1..],
                    Tile {
                        corner: [
                            tile.corner[0] + i * next_tile_size,
                            tile.corner[1] + j * next_tile_size,
                        ],
                    },
                );
            }
        }
    } else {
        let sub_tape = token.simplify();
        let sub_jit = V::from_tape(&sub_tape);
        for j in 0..tile_sizes[0] {
            for i in 0..(tile_sizes[0] / 4) {
                render_pixels(
                    &sub_jit,
                    out,
                    config,
                    Tile {
                        corner: [tile.corner[0] + i * 4, tile.corner[1] + j],
                    },
                );
            }
        }
    }
}

fn render_pixels<'a, V: VecFunc<'a>>(
    handle: &V,
    out: &mut [Option<Pixel>],
    config: &RenderConfig,
    tile: Tile,
) {
    let mut eval = handle.get_evaluator();

    let mut x_vec = [0.0; 4];
    for (i, x) in x_vec.iter_mut().enumerate() {
        *x = config.pixel_to_pos(tile.corner[0] + i);
    }
    let y_vec = [config.pixel_to_pos(tile.corner[1]); 4];
    let v = eval.eval_v(x_vec, y_vec, [0.0; 4]);

    for (i, v) in v.iter().enumerate() {
        out[tile.corner[0] % config.tile_size
            + i
            + (tile.corner[1] % config.tile_size) * config.tile_size] =
            Some(if *v < 0.0 {
                Pixel::Filled
            } else {
                Pixel::Empty
            });
    }
}

////////////////////////////////////////////////////////////////////////////////

pub fn render<'a, I: IntervalFunc<'a>, V: VecFunc<'a>>(
    tape: Tape,
    config: &RenderConfig,
) -> Vec<Pixel> {
    assert!(config.image_size % config.tile_size == 0);
    assert!(config.tile_size % config.subtile_size == 0);
    assert!(config.subtile_size % 4 == 0);

    let i_handle = I::from_tape(&tape);
    let mut tiles = vec![];
    for i in 0..config.image_size / config.tile_size {
        for j in 0..config.image_size / config.tile_size {
            tiles.push(Tile {
                corner: [i * config.tile_size, j * config.tile_size],
            });
        }
    }

    let index = AtomicUsize::new(0);
    let out = std::thread::scope(|s| {
        let mut handles = vec![];
        for _ in 0..config.threads {
            handles.push(
                s.spawn(|| worker::<_, V>(&i_handle, &tiles, &index, config)),
            );
        }
        let mut out = vec![];
        for h in handles {
            out.extend(h.join().unwrap().into_iter());
        }
        out
    });

    let mut image = vec![None; config.image_size * config.image_size];
    for (tile, data) in out.iter() {
        for j in 0..config.tile_size {
            for i in 0..config.tile_size {
                let x = i + tile.corner[0];
                let y = j + tile.corner[1];
                image[x + (config.image_size - y - 1) * config.image_size] =
                    Some(data[i + j * config.tile_size]);
            }
        }
    }
    image.into_iter().map(Option::unwrap).collect()
}
