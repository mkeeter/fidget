//! Bitmap rendering
use crate::{
    eval::{
        float_slice::{FloatSliceEval, FloatSliceEvalT, FloatSliceFunc},
        interval::{Interval, IntervalFunc},
        EvalFamily,
    },
    tape::Tape,
};
use std::sync::atomic::{AtomicUsize, Ordering};

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug)]
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

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug)]
struct Tile {
    corner: [usize; 3],
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
    fn eval_s<'a, E>(&mut self, f: &mut FloatSliceEval<'a, E>)
    where
        E: FloatSliceEvalT<'a>,
    {
        f.eval_s(&self.x, &self.y, &self.z, &mut self.out);
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<const N: usize> {
    config: RenderConfig<N>,
    scratch: Scratch,
    out: Vec<f32>,
}

impl<const N: usize> Worker<N> {
    fn render_tile_recurse<'a, 'b, I>(
        &mut self,
        handle: &'b IntervalFunc<'a, <I as EvalFamily<'a>>::IntervalFunc>,
        depth: usize,
        tile: Tile,
        float_handle: Option<
            &FloatSliceFunc<'b, <I as EvalFamily<'b>>::FloatSliceFunc>,
        >,
    ) -> Option<FloatSliceFunc<'b, <I as EvalFamily<'b>>::FloatSliceFunc>>
    where
        for<'s> I: EvalFamily<'s>,
    {
        let mut eval = handle.get_evaluator();
        let tile_size = self.config.tile_sizes[depth];

        let x_min = self.config.pixel_to_pos(tile.corner[0]) + self.config.dx;
        let x_max = self.config.pixel_to_pos(tile.corner[0] + tile_size)
            + self.config.dx;
        let y_min = self.config.pixel_to_pos(tile.corner[1]) + self.config.dy;
        let y_max = self.config.pixel_to_pos(tile.corner[1] + tile_size)
            + self.config.dy;
        let z_min = self.config.pixel_to_pos(tile.corner[2]) + self.config.dz;
        let z_max = self.config.pixel_to_pos(tile.corner[2] + tile_size)
            + self.config.dz;

        let x = Interval::new(x_min, x_max);
        let y = Interval::new(y_min, y_max);
        let z = Interval::new(z_min, z_max);
        let i = eval.eval_i_subdiv(x, y, z, self.config.interval_subdiv);

        let fill = if i.upper() < 0.0 {
            Some(z_max)
        } else if i.lower() > 0.0 {
            // Return early if this tile is completely empty
            return None;
        } else {
            None
        };

        if let Some(fill) = fill {
            for x in 0..tile_size {
                for y in 0..tile_size {
                    let i = self.config.tile_to_offset(tile, x, y);
                    self.out[i] = self.out[i].max(fill);
                }
            }
            None
        } else if let Some(next_tile_size) =
            self.config.tile_sizes.get(depth + 1).cloned()
        {
            let sub_tape = eval.simplify(I::REG_LIMIT);
            let sub_jit = I::from_tape_i(&sub_tape);
            let n = tile_size / next_tile_size;
            let mut float_handle = None;
            for j in 0..n {
                for i in 0..n {
                    for k in 0..n {
                        let r = self.render_tile_recurse::<I>(
                            &sub_jit,
                            depth + 1,
                            Tile {
                                corner: [
                                    tile.corner[0] + i * next_tile_size,
                                    tile.corner[1] + j * next_tile_size,
                                    tile.corner[2] + k * next_tile_size,
                                ],
                            },
                            float_handle.as_ref(),
                        );
                        if r.is_some() {
                            float_handle = r;
                        }
                    }
                }
            }
            None
        } else {
            // Prepare for pixel-by-pixel evaluation
            let mut index = 0;
            for j in 0..tile_size {
                let y = self.config.pixel_to_pos(tile.corner[1] + j)
                    + self.config.dy;
                for i in 0..tile_size {
                    let x = self.config.pixel_to_pos(tile.corner[0] + i)
                        + self.config.dx;
                    for k in 0..tile_size {
                        let z = self.config.pixel_to_pos(tile.corner[2] + k)
                            + self.config.dz;
                        self.scratch.x[index] = x;
                        self.scratch.y[index] = y;
                        self.scratch.z[index] = z;
                        index += 1;
                    }
                }
            }
            assert_eq!(index, self.scratch.x.len());

            // This gets a little messy in terms of lifetimes.
            //
            // In some cases, the shortened tape isn't actually any shorter, so
            // it's a waste of time to rebuild it.  Instead, we we want to use a
            // float-slice evaluator that's bound to the *parent* tape.
            // Luckily, such a thing _may_ be passed into this function.  If
            // not, we build it here and then pass it out, so future calls can
            // use it.
            //
            // (this matters most for the JIT compiler, which is _expensive_)
            let sub_tape = eval.simplify(I::REG_LIMIT);
            let ret = if sub_tape.len() < handle.tape().len() {
                let sub_jit = I::from_tape_s(&sub_tape);

                let mut eval = sub_jit.get_evaluator();
                self.scratch.eval_s(&mut eval);

                None
            } else if let Some(r) = float_handle {
                // Reuse the FloatSliceFunc handle passed in
                let mut eval = r.get_evaluator();
                self.scratch.eval_s(&mut eval);
                None
            } else {
                // Build our own FloatSliceFunc handle, then return it
                let func = I::from_tape_s(handle.tape());
                let mut eval = func.get_evaluator();
                self.scratch.eval_s(&mut eval);
                Some(func)
            };

            // Copy from the scratch buffer to the output tile
            let mut index = 0;
            for j in 0..tile_size {
                for i in 0..tile_size {
                    for k in 0..tile_size {
                        let z = self.config.pixel_to_pos(tile.corner[2] + k)
                            + self.config.dz;
                        let o = self.config.tile_to_offset(tile, i, j);
                        if self.scratch.out[index] < 0.0 {
                            self.out[o] = self.out[o].max(z);
                        }
                        index += 1;
                    }
                }
            }
            ret
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

    // Calculate maximum evaluation buffer size
    let buf_size = config.tile_sizes.last().cloned().unwrap_or(0);
    let scratch = Scratch::new(buf_size * buf_size * buf_size);
    let mut w = Worker {
        scratch,
        out: vec![],
        config: *config,
    };
    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        w.out = vec![
            f32::NEG_INFINITY;
            config.tile_sizes[0] * config.tile_sizes[0]
        ];
        w.render_tile_recurse::<I>(i_handle, 0, tile, None);

        let mut pixels = vec![];
        std::mem::swap(&mut pixels, &mut w.out);
        out.push((tile, pixels));
    }
    out
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
