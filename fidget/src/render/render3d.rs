//! Bitmap rendering
use crate::{
    eval::{
        float_slice::{FloatSliceEval, FloatSliceEvalT},
        grad::{Grad, GradEval, GradEvalT},
        interval::{Interval, IntervalEval},
        EvalFamily,
    },
    tape::Tape,
};

use nalgebra::{Matrix4, Point3, Transform3, Vector3};
use std::sync::atomic::{AtomicUsize, Ordering};

////////////////////////////////////////////////////////////////////////////////

pub struct RenderConfig {
    pub image_size: usize,
    pub tile_sizes: Vec<usize>,
    pub interval_subdiv: usize,
    pub threads: usize,

    pub mat: Transform3<f32>,
}

impl RenderConfig {
    #[inline]
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
    out_grad: Vec<Grad>,

    /// Depth of each column
    columns: Vec<usize>,
}

impl Scratch {
    fn new(tile_size: usize) -> Self {
        let size2 = tile_size.pow(2);
        let size3 = tile_size.pow(3);
        Self {
            x: vec![0.0; size3],
            y: vec![0.0; size3],
            z: vec![0.0; size3],
            out: vec![0.0; size3],
            out_grad: vec![0.0.into(); size3],
            columns: vec![0; size2],
        }
    }
    fn eval_s<E: FloatSliceEvalT>(
        &mut self,
        f: &mut FloatSliceEval<E>,
        size: usize,
    ) {
        f.eval_s(
            &self.x[0..size],
            &self.y[0..size],
            &self.z[0..size],
            &mut self.out[0..size],
        );
    }
    fn eval_g<E: GradEvalT>(&mut self, f: &mut GradEval<E>, size: usize) {
        f.eval_g(
            &self.x[0..size],
            &self.y[0..size],
            &self.z[0..size],
            &mut self.out_grad[0..size],
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<'a, I: EvalFamily> {
    config: &'a RenderConfig,
    mat: Matrix4<f32>,
    scratch: Scratch,
    depth: Vec<usize>,
    color: Vec<[u8; 3]>,
    buffers:
        Vec<<<I as EvalFamily>::FloatSliceEval as FloatSliceEvalT>::Storage>,
}

impl<I: EvalFamily> Worker<'_, I> {
    fn render_tile_recurse(
        &mut self,
        handle: &mut IntervalEval<I::IntervalEval>,
        depth: usize,
        tile: Tile,
        float_handle: &mut Option<FloatSliceEval<I::FloatSliceEval>>,
    ) {
        let tile_size = self.config.tile_sizes[depth];

        // Early exit if every single pixel is filled
        let fill_z = tile.corner[2] + tile_size + 1;
        let mut all_blocked = true;
        for y in 0..tile_size {
            for x in 0..tile_size {
                let i = self.config.tile_to_offset(tile, x, y);
                all_blocked &= self.depth[i] >= fill_z;
            }
        }
        if all_blocked {
            return;
        }

        // Brute-force way to find the (interval) bounding box of the region
        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        let mut z_min = f32::INFINITY;
        let mut z_max = f32::NEG_INFINITY;
        let base = Point3::from(tile.corner);
        for i in 0..8 {
            let offset = Vector3::new(
                if (i & 1) == 0 { 0 } else { tile_size },
                if (i & 2) == 0 { 0 } else { tile_size },
                if (i & 4) == 0 { 0 } else { tile_size },
            );
            let p = (base + offset).cast::<f32>();
            let p = self.mat.transform_point(&p);
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

        let i = handle.eval_i_subdiv(x, y, z, self.config.interval_subdiv);

        if i.upper() < 0.0 {
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let i = self.config.tile_to_offset(tile, x, y);
                    self.depth[i] = self.depth[i].max(fill_z);
                }
            }
            // TODO: handle gradients here as well?
            return;
        } else if i.lower() > 0.0 {
            // Return early if this tile is completely empty
            return;
        };

        if let Some(next_tile_size) =
            self.config.tile_sizes.get(depth + 1).cloned()
        {
            let sub_tape = handle.simplify(I::REG_LIMIT);
            let mut sub_jit = IntervalEval::from(sub_tape);
            let n = tile_size / next_tile_size;
            let mut float_handle = None;
            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        self.render_tile_recurse(
                            &mut sub_jit,
                            depth + 1,
                            Tile {
                                corner: [
                                    tile.corner[0] + i * next_tile_size,
                                    tile.corner[1] + j * next_tile_size,
                                    tile.corner[2] + k * next_tile_size,
                                ],
                            },
                            &mut float_handle,
                        );
                    }
                }
            }
            if let Some(f) = float_handle {
                self.buffers.push(f.take().unwrap());
            }
        } else {
            self.render_tile_pixels(handle, tile_size, tile, float_handle)
        }
    }

    fn render_tile_pixels(
        &mut self,
        handle: &mut IntervalEval<I::IntervalEval>,
        tile_size: usize,
        tile: Tile,
        float_handle: &mut Option<FloatSliceEval<I::FloatSliceEval>>,
    ) {
        // Prepare for pixel-by-pixel evaluation
        let mut index = 0;
        assert!(self.scratch.x.len() >= tile_size.pow(3));
        assert!(self.scratch.y.len() >= tile_size.pow(3));
        assert!(self.scratch.z.len() >= tile_size.pow(3));
        self.scratch.columns.clear();
        for xy in 0..tile_size.pow(2) {
            let i = xy % tile_size;
            let j = xy / tile_size;
            let o = self.config.tile_to_offset(tile, i, j);

            // The matrix transformation is separable until the final
            // division by w.  We can precompute the XY-1 portion of the
            // multiplication here, since it's shared by every voxel in this
            // column of the image.
            let v = ((tile.corner[0] + i) as f32) * self.mat.column(0)
                + ((tile.corner[1] + j) as f32) * self.mat.column(1)
                + self.mat.column(3);

            let zmax = tile.corner[2] + tile_size;
            if self.depth[o] >= zmax {
                continue;
            }

            for k in (0..tile_size).rev() {
                let v = v + ((tile.corner[2] + k) as f32) * self.mat.column(2);

                // SAFETY:
                // Index cannot exceed tile_size**3, which is (a) the size
                // that we allocated in `Scratch::new` and (b) checked by
                // assertions above.
                //
                // Using unsafe indexing here is a roughly 2.5% speedup,
                // since this is the hottest loop.
                unsafe {
                    *self.scratch.x.get_unchecked_mut(index) = v.x / v.w;
                    *self.scratch.y.get_unchecked_mut(index) = v.y / v.w;
                    *self.scratch.z.get_unchecked_mut(index) = v.z / v.w;
                }
                index += 1;
            }
            self.scratch.columns.push(xy);
        }
        let size = index;
        assert!(size > 0);

        // This gets a little messy in terms of lifetimes.
        //
        // In some cases, the shortened tape isn't actually any shorter, so
        // it's a waste of time to rebuild it.  Instead, we want to use a
        // float-slice evaluator that's bound to the *parent* tape.
        // Luckily, such a thing _may_ be passed into this function.  If
        // not, we build it here and then pass it out, so future calls can
        // use it.
        //
        // (this matters most for the JIT compiler, which is _expensive_)
        let sub_tape = handle.simplify(I::REG_LIMIT);
        if sub_tape.len() < handle.tape().len() {
            let mut func = self.get_float_slice_eval(sub_tape.clone());

            self.scratch.eval_s(&mut func, size);

            // We consume the evaluator, so any reuse of memory between the
            // FloatSliceFunc and FloatSliceEval should be cleared up and we
            // should be able to reuse the working memory.
            self.buffers.push(func.take().unwrap());
        } else {
            // Reuse the FloatSliceFunc handle passed in, or build one if it
            // wasn't already available (which makes it available to siblings)
            let mut func = float_handle.get_or_insert_with(|| {
                self.get_float_slice_eval(handle.tape())
            });
            self.scratch.eval_s(func, size);
        }

        // We're iterating over three different things simultaneously:
        // - col refers to the xy position in the tile
        // - index refers to the evaluation array, self.scratch.out
        // - grad refers to points that we must do gradient evaluation on
        let mut col = 0;
        let mut index = 0;
        let mut grad = 0;
        while col < self.scratch.columns.len() {
            let xy = self.scratch.columns[col];
            let i = xy % tile_size;
            let j = xy / tile_size;
            let o = self.config.tile_to_offset(tile, i, j);
            for k in (0..tile_size).rev() {
                let z = tile.corner[2] + k + 1;
                // Early exit for the first pixel in the column
                if self.scratch.out[index] < 0.0 && self.depth[o] <= z {
                    self.depth[o] = z;
                    index += k + 1;

                    // Prepare to do gradient rendering of this point.
                    // We step one voxel above the surface to reduce
                    // glitchiness on edges and corners, where rendering
                    // inside the surface could pick the wrong normal.
                    let p = self.mat.transform_point(&Point3::new(
                        (tile.corner[0] + i) as f32,
                        (tile.corner[1] + j) as f32,
                        (tile.corner[2] + k + 1) as f32,
                    ));
                    self.scratch.x[grad] = p.x;
                    self.scratch.y[grad] = p.y;
                    self.scratch.z[grad] = p.z;

                    // This can only be called once per k loop, so we'll
                    // never overwrite parts of columns that are still used
                    // by the outer loop
                    self.scratch.columns[grad] = o;
                    grad += 1;

                    break;
                }
                index += 1;
            }
            col += 1;
        }

        if grad > 0 {
            let mut func = GradEval::<I::GradEval>::from(sub_tape);
            self.scratch.eval_g(&mut func, grad);
            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
                self.color[*o] = self.scratch.out_grad[index]
                    .to_rgb()
                    .unwrap_or([255, 0, 0]);
            }
        }
    }

    fn get_float_slice_eval(
        &mut self,
        sub_tape: Tape,
    ) -> FloatSliceEval<I::FloatSliceEval> {
        match self.buffers.pop() {
            Some(s) => {
                let (sub_jit, s) =
                    FloatSliceEval::<I::FloatSliceEval>::new_give(sub_tape, s);
                self.buffers.extend(s);
                sub_jit
            }
            None => FloatSliceEval::<I::FloatSliceEval>::from(sub_tape),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

fn worker<I: EvalFamily>(
    mut i_handle: IntervalEval<<I as EvalFamily>::IntervalEval>,
    tiles: &[Tile],
    i: &AtomicUsize,
    config: &RenderConfig,
) -> Vec<(Tile, Vec<usize>, Vec<[u8; 3]>)> {
    let mut out = vec![];

    let mat = config.mat.matrix()
        * nalgebra::Matrix4::identity()
            .append_scaling(2.0 / config.image_size as f32)
            .append_translation(&Vector3::new(-1.0, -1.0, -1.0));

    // Calculate maximum evaluation buffer size
    let buf_size = *config.tile_sizes.last().unwrap();
    let scratch = Scratch::new(buf_size);
    let mut w: Worker<I> = Worker {
        scratch,
        depth: vec![],
        color: vec![],
        config,
        mat,
        buffers: vec![],
    };
    loop {
        let index = i.fetch_add(1, Ordering::Relaxed);
        if index >= tiles.len() {
            break;
        }
        let tile = tiles[index];

        // Prepare to render, allocating space for a tile
        w.depth = vec![0; config.tile_sizes[0].pow(2)];
        w.color = vec![[0; 3]; config.tile_sizes[0].pow(2)];
        w.render_tile_recurse(&mut i_handle, 0, tile, &mut None);

        // Steal the tile, replacing it with an empty vec
        let mut depth = vec![];
        let mut color = vec![];
        std::mem::swap(&mut depth, &mut w.depth);
        std::mem::swap(&mut color, &mut w.color);
        out.push((tile, depth, color));
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

pub fn render<I: EvalFamily>(
    tape: Tape,
    config: &RenderConfig,
) -> (Vec<usize>, Vec<[u8; 3]>) {
    assert!(config.image_size % config.tile_sizes[0] == 0);
    for i in 0..config.tile_sizes.len() - 1 {
        assert!(config.tile_sizes[i] % config.tile_sizes[i + 1] == 0);
    }

    let i_handle = IntervalEval::from(tape);
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
            let i = i_handle.clone();
            handles.push(s.spawn(|| worker::<I>(i, &tiles, &index, config)));
        }
        let mut out = vec![];
        for h in handles {
            out.extend(h.join().unwrap().into_iter());
        }
        out
    });

    let mut image_depth = vec![0; config.image_size.pow(2)];
    let mut image_color = vec![[0; 3]; config.image_size.pow(2)];
    for (tile, depth, color) in out.iter() {
        let mut index = 0;
        for j in 0..config.tile_sizes[0] {
            let y = j + tile.corner[1];
            for i in 0..config.tile_sizes[0] {
                let x = i + tile.corner[0];
                let o = (config.image_size - y - 1) * config.image_size + x;
                if depth[index] >= image_depth[o] {
                    image_color[o] = color[index];
                    image_depth[o] = depth[index];
                }
                index += 1;
            }
        }
    }
    (image_depth, image_color)
}
