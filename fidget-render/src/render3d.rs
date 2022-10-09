//! Bitmap rendering
use fidget_core::{
    eval::{
        float_slice::{FloatSliceEval, FloatSliceEvalT, FloatSliceFunc},
        interval::{Interval, IntervalFunc},
        EvalFamily,
    },
    tape::Tape,
};

use nalgebra::{Transform3, Vector3};
use std::sync::atomic::{AtomicUsize, Ordering};

////////////////////////////////////////////////////////////////////////////////

pub struct RenderConfig<const N: usize> {
    pub image_size: usize,
    pub tile_sizes: [usize; N],
    pub interval_subdiv: usize,
    pub threads: usize,

    pub mat: Transform3<f32>,
}

impl<const N: usize> RenderConfig<N> {
    fn vec3_to_pos(&self, p: Vector3<usize>) -> Vector3<f32> {
        let p = 2.0 * (p.cast::<f32>() / self.image_size as f32)
            - Vector3::new(1.0, 1.0, 1.0);
        self.mat.transform_vector(&p)
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

struct Worker<'a, const N: usize> {
    config: &'a RenderConfig<N>,
    scratch: Scratch,
    out: Vec<usize>,
}

impl<const N: usize> Worker<'_, N> {
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
        let tile_size = self.config.tile_sizes[depth];

        // Brute-force way to find the (interval) bounding box of the region
        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        let mut z_min = f32::INFINITY;
        let mut z_max = f32::NEG_INFINITY;
        let base = Vector3::from(tile.corner);
        for i in 0..8 {
            let offset = Vector3::new(
                if (i & 1) == 0 { 0 } else { tile_size },
                if (i & 2) == 0 { 0 } else { tile_size },
                if (i & 4) == 0 { 0 } else { tile_size },
            );
            let p = self.config.vec3_to_pos(base + offset);
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
            z_min = z_min.min(p.z);
            z_max = z_max.max(p.z);
        }

        let x = Interval::new(x_min, x_max);
        let y = Interval::new(y_min, y_max);
        let z = Interval::new(z_min, z_max);

        let mut eval = handle.get_evaluator();
        let i = eval.eval_i_subdiv(x, y, z, self.config.interval_subdiv);

        let fill = if i.upper() < 0.0 {
            Some(tile.corner[2] + tile_size + 1)
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
                for i in 0..tile_size {
                    for k in (0..tile_size).rev() {
                        let v = self.config.vec3_to_pos(Vector3::new(
                            tile.corner[0] + i,
                            tile.corner[1] + j,
                            tile.corner[2] + k,
                        ));
                        self.scratch.x[index] = v.x;
                        self.scratch.y[index] = v.y;
                        self.scratch.z[index] = v.z;
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
                    for k in (0..tile_size).rev() {
                        let z = tile.corner[2] + k + 1;
                        let o = self.config.tile_to_offset(tile, i, j);
                        if self.scratch.out[index] < 0.0 && self.out[o] < z {
                            self.out[o] = z;
                            //index += k + 1;
                            //break;
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
) -> Vec<(Tile, Vec<usize>)>
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
        config,
    };
    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        // Prepare to render, allocating space for a tile
        w.out = vec![0; config.tile_sizes[0] * config.tile_sizes[0]];
        w.render_tile_recurse::<I>(i_handle, 0, tile, None);

        // Steal the tile, replacing it with an empty vec
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
) -> Vec<usize>
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

    let mut image = vec![0; config.image_size * config.image_size];
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
