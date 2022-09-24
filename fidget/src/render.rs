//! Bitmap rendering
use crate::{
    eval::{
        EvalFamily, FloatSliceEval, FloatSliceFunc, Interval, IntervalEval,
        IntervalFunc,
    },
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
    pub fn as_debug_color(&self) -> [u8; 4] {
        match self {
            Pixel::EmptyTile => [50, 0, 0, 255],
            Pixel::FilledTile => [255, 0, 0, 255],
            Pixel::EmptySubtile => [0, 50, 0, 255],
            Pixel::FilledSubtile => [0, 255, 0, 255],
            Pixel::Empty => [0, 0, 0, 255],
            Pixel::Filled => [255, 255, 255, 255],
        }
    }

    pub fn as_alpha(&self) -> u8 {
        match self {
            Pixel::EmptyTile | Pixel::EmptySubtile | Pixel::Empty => 0,
            Pixel::FilledTile | Pixel::FilledSubtile | Pixel::Filled => u8::MAX,
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

fn worker<'a, I>(
    i_handle: &<I as EvalFamily<'a>>::IntervalFunc,
    tiles: &[Tile],
    i: &AtomicUsize,
    config: &RenderConfig,
) -> Vec<(Tile, Vec<Pixel>)>
where
    for<'s> I: EvalFamily<'s>,
{
    let mut out = vec![];
    let mut scratch = Scratch::new(config.subtile_size * config.subtile_size);
    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        let mut pixels = vec![None; config.tile_size * config.tile_size];
        render_tile_recurse::<I>(
            i_handle,
            &mut pixels,
            config,
            &[config.tile_size, config.subtile_size],
            tile,
            &mut scratch,
        );
        let pixels = pixels.into_iter().map(Option::unwrap).collect();
        out.push((tile, pixels))
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

fn render_tile_recurse<'a, I>(
    handle: &<I as EvalFamily<'a>>::IntervalFunc,
    out: &mut [Option<Pixel>],
    config: &RenderConfig,
    tile_sizes: &[usize],
    tile: Tile,
    scratch: &mut Scratch,
) where
    for<'s> I: EvalFamily<'s>,
{
    let mut eval = handle.get_evaluator();

    let x_min = config.pixel_to_pos(tile.corner[0]);
    let x_max = config.pixel_to_pos(tile.corner[0] + tile_sizes[0]);
    let y_min = config.pixel_to_pos(tile.corner[1]);
    let y_max = config.pixel_to_pos(tile.corner[1] + tile_sizes[0]);

    let x = Interval::new(x_min, x_max);
    let y = Interval::new(y_min, y_max);
    let z = Interval::new(0.0, 0.0);
    let i = eval.eval_i_subdiv(x, y, z, config.interval_subdiv);

    let fill = if i.upper() < 0.0 {
        if tile_sizes.len() > 1 {
            Some(Pixel::FilledTile)
        } else {
            Some(Pixel::FilledSubtile)
        }
    } else if i.lower() > 0.0 {
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
        let sub_tape = eval.simplify(I::REG_LIMIT);
        let sub_jit = I::from_tape_i(&sub_tape);
        let n = tile_sizes[0] / next_tile_size;
        for j in 0..n {
            for i in 0..n {
                render_tile_recurse::<I>(
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
                    scratch,
                );
            }
        }
    } else {
        let sub_tape = eval.simplify(I::REG_LIMIT);
        let sub_jit = I::from_tape_s(&sub_tape);

        let mut index = 0;
        for j in 0..tile_sizes[0] {
            let y = config.pixel_to_pos(tile.corner[1] + j);
            for i in 0..tile_sizes[0] {
                scratch.x[index] = config.pixel_to_pos(tile.corner[0] + i);
                scratch.y[index] = y;
                index += 1;
            }
        }

        let mut eval = sub_jit.get_evaluator();
        eval.eval_s(&scratch.x, &scratch.y, &scratch.z, &mut scratch.out);

        let mut index = 0;
        for j in 0..tile_sizes[0] {
            for i in 0..tile_sizes[0] {
                out[tile.corner[0] % config.tile_size
                    + i
                    + ((tile.corner[1] % config.tile_size) + j)
                        * config.tile_size] =
                    Some(if scratch.out[index] < 0.0 {
                        Pixel::Filled
                    } else {
                        Pixel::Empty
                    });
                index += 1;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub fn render<I>(tape: Tape, config: &RenderConfig) -> Vec<Pixel>
where
    for<'s> I: EvalFamily<'s>,
{
    assert!(config.image_size % config.tile_size == 0);
    assert!(config.tile_size % config.subtile_size == 0);
    assert!(config.subtile_size % 4 == 0);

    let i_handle = I::from_tape_i(&tape);
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
                s.spawn(|| worker::<I>(&i_handle, &tiles, &index, config)),
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
