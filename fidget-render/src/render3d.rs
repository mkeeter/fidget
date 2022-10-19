//! Bitmap rendering
use fidget_core::{
    eval::{
        float_slice::{
            FloatSliceEval, FloatSliceEvalT, FloatSliceFunc, FloatSliceFuncT,
        },
        grad_slice::{Grad, GradSliceEval, GradSliceEvalT, GradSliceFunc},
        interval::{Interval, IntervalFunc},
        EvalFamily,
    },
    tape::Tape,
};

use nalgebra::{Matrix4, Point3, Transform3, Vector3, Vector4};
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
        let size2 = tile_size * tile_size * tile_size;
        let size3 = size2 * tile_size;
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
    fn eval_g<E: GradSliceEvalT>(
        &mut self,
        f: &mut GradSliceEval<E>,
        size: usize,
    ) {
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
        Vec<<<I as EvalFamily>::FloatSliceFunc as FloatSliceFuncT>::Storage>,
}

impl<I: EvalFamily> Worker<'_, I> {
    fn render_tile_recurse(
        &mut self,
        handle: &IntervalFunc<I::IntervalFunc>,
        depth: usize,
        tile: Tile,
        float_handle: Option<&FloatSliceFunc<I::FloatSliceFunc>>,
    ) -> Option<FloatSliceFunc<I::FloatSliceFunc>> {
        let tile_size = self.config.tile_sizes[depth];

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

        let mut eval = handle.get_evaluator();
        let i = eval.eval_i_subdiv(x, y, z, self.config.interval_subdiv);

        let fill = if i.upper() < 0.0 {
            Some(tile.corner[2] + tile_size + 1)
        } else if i.lower() > 0.0 {
            // Return early if this tile is completely empty
            return None;
        } else {
            // Continue evaluating (this also includes the NaN case)
            None
        };

        if let Some(fill) = fill {
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let i = self.config.tile_to_offset(tile, x, y);
                    self.depth[i] = self.depth[i].max(fill);
                }
            }
            // TODO: handle gradients here as well?
            None
        } else if let Some(next_tile_size) =
            self.config.tile_sizes.get(depth + 1).cloned()
        {
            let sub_tape = eval.simplify(I::REG_LIMIT);
            let sub_jit = IntervalFunc::from_tape(sub_tape);
            let n = tile_size / next_tile_size;
            let mut float_handle = None;
            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        let r = self.render_tile_recurse(
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
            if let Some(f) = float_handle {
                self.buffers.push(f.take().unwrap());
            }
            None
        } else {
            // Prepare for pixel-by-pixel evaluation
            let mut index = 0;
            self.scratch.columns.clear();
            for xy in 0..(tile_size * tile_size) {
                let i = xy % tile_size;
                let j = xy / tile_size;
                let o = self.config.tile_to_offset(tile, i, j);

                // Precompute the XY part of the transform
                let v = self.mat
                    * Vector4::new(
                        (tile.corner[0] + i) as f32,
                        (tile.corner[1] + j) as f32,
                        0.0,
                        1.0,
                    );

                let zmax = tile.corner[2] + tile_size;
                if self.depth[o] >= zmax {
                    continue;
                }

                for k in (0..tile_size).rev() {
                    let v =
                        v + ((tile.corner[2] + k) as f32) * self.mat.column(2);
                    self.scratch.x[index] = v.x / v.w;
                    self.scratch.y[index] = v.y / v.w;
                    self.scratch.z[index] = v.z / v.w;
                    index += 1;
                }
                self.scratch.columns.push(xy);
            }
            let size = index;

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
            let sub_tape = eval.simplify(I::REG_LIMIT);
            let ret = if sub_tape.len() < handle.tape().len() {
                let func = self.get_float_slice_eval(sub_tape.clone());

                let mut eval = func.get_evaluator();
                self.scratch.eval_s(&mut eval, size);
                drop(eval);

                // We just dropped the evaluator, so any reuse of memory between
                // the FloatSliceFunc and FloatSliceEval should be cleared up
                // and we should be able to reuse the working memory.
                self.buffers.push(func.take().unwrap());

                None
            } else if let Some(r) = float_handle {
                // Reuse the FloatSliceFunc handle passed in
                let mut eval = r.get_evaluator();
                self.scratch.eval_s(&mut eval, size);
                None
            } else {
                let func = self.get_float_slice_eval(handle.tape());

                let mut eval = func.get_evaluator();
                self.scratch.eval_s(&mut eval, size);
                Some(func)
            };

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
                let func =
                    GradSliceFunc::<I::GradSliceFunc>::from_tape(sub_tape);

                let mut eval = func.get_evaluator();
                self.scratch.eval_g(&mut eval, grad);
                for (index, o) in
                    self.scratch.columns[0..grad].iter().enumerate()
                {
                    self.color[*o] = self.scratch.out_grad[index]
                        .to_rgb()
                        .unwrap_or([255, 0, 0]);
                }
            }

            ret
        }
    }

    fn get_float_slice_eval(
        &mut self,
        sub_tape: Tape,
    ) -> FloatSliceFunc<I::FloatSliceFunc> {
        match self.buffers.pop() {
            Some(s) => {
                let (sub_jit, s) =
                    FloatSliceFunc::<I::FloatSliceFunc>::new_give(sub_tape, s);
                self.buffers.extend(s);
                sub_jit
            }
            None => FloatSliceFunc::<I::FloatSliceFunc>::from_tape(sub_tape),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

fn worker<I: EvalFamily>(
    i_handle: &IntervalFunc<<I as EvalFamily>::IntervalFunc>,
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
        w.depth = vec![0; config.tile_sizes[0] * config.tile_sizes[0]];
        w.color = vec![[0; 3]; config.tile_sizes[0] * config.tile_sizes[0]];
        w.render_tile_recurse(i_handle, 0, tile, None);

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

    let i_handle = IntervalFunc::from_tape(tape);
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
                s.spawn(|| worker::<I>(&i_handle, &tiles, &index, config)),
            );
        }
        let mut out = vec![];
        for h in handles {
            out.extend(h.join().unwrap().into_iter());
        }
        out
    });

    let mut image_depth = vec![0; config.image_size * config.image_size];
    let mut image_color = vec![[0; 3]; config.image_size * config.image_size];
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
