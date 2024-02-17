//! 3D bitmap rendering / rasterization
use crate::{
    eval::{
        types::{Grad, Interval},
        BulkEvaluator, Shape, Tape, TracingEvaluator,
    },
    render::config::{AlignedRenderConfig, Queue, RenderConfig, Tile},
};

use nalgebra::{Point3, Vector3};
use std::{collections::HashMap, sync::Arc};

////////////////////////////////////////////////////////////////////////////////

struct Scratch {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,

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

            columns: vec![0; size2],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct ShapeAndTape<S: Shape> {
    shape: S,

    i_tape: Option<Arc<<S::IntervalEval as TracingEvaluator<Interval>>::Tape>>,
    f_tape: Option<<S::FloatSliceEval as BulkEvaluator<f32>>::Tape>,
    g_tape: Option<<S::GradSliceEval as BulkEvaluator<Grad>>::Tape>,
}

impl<S: Shape> ShapeAndTape<S> {
    fn i_tape(
        &mut self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::IntervalEval as TracingEvaluator<Interval>>::Tape {
        self.i_tape.get_or_insert_with(|| {
            Arc::new(
                self.shape.interval_tape(storage.pop().unwrap_or_default()),
            )
        })
    }
    fn f_tape(
        &mut self,

        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::FloatSliceEval as BulkEvaluator<f32>>::Tape {
        self.f_tape.get_or_insert_with(|| {
            self.shape
                .float_slice_tape(storage.pop().unwrap_or_default())
        })
    }
    fn g_tape(
        &mut self,

        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::GradSliceEval as BulkEvaluator<Grad>>::Tape {
        self.g_tape.get_or_insert_with(|| {
            self.shape
                .grad_slice_tape(storage.pop().unwrap_or_default())
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<'a, S: Shape> {
    config: &'a AlignedRenderConfig<3>,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch,

    eval_float_slice: S::FloatSliceEval,
    eval_grad_slice: S::GradSliceEval,
    eval_interval: S::IntervalEval,

    tape_storage: Vec<S::TapeStorage>,
    shape_storage: Vec<S::Storage>,
    workspace: S::Workspace,

    /// Output images for this specific tile
    depth: Vec<u32>,
    color: Vec<[u8; 3]>,
}

impl<S: Shape> Worker<'_, S> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut ShapeAndTape<S>,
        depth: usize,
        tile: Tile<3>,
    ) {
        // Early exit if every single pixel is filled
        let tile_size = self.config.tile_sizes[depth];
        let fill_z = (tile.corner[2] + tile_size + 1).try_into().unwrap();
        if (0..tile_size).all(|y| {
            let i = self.config.tile_to_offset(tile, 0, y);
            (0..tile_size).all(|x| self.depth[i + x] >= fill_z)
        }) {
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
            let p = self.config.mat.transform_point(&p);
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

        let (i, trace) = self
            .eval_interval
            .eval(shape.i_tape(&mut self.tape_storage), x, y, z, &[])
            .unwrap();

        // Return early if this tile is completely empty or full, returning
        // `data_interval` to scratch memory for reuse.
        if i.upper() < 0.0 {
            for y in 0..tile_size {
                let i = self.config.tile_to_offset(tile, 0, y);
                for x in 0..tile_size {
                    self.depth[i + x] = self.depth[i + x].max(fill_z);
                }
            }
            return;
        } else if i.lower() > 0.0 {
            return;
        }

        // Calculate a simplified tape, reverting to the parent tape if the
        // simplified tape isn't any shorter.
        let mut sub_tape = if let Some(trace) = trace.as_ref() {
            let s = self.shape_storage.pop().unwrap_or_default();
            let next =
                shape.shape.simplify(trace, s, &mut self.workspace).unwrap();
            Some(ShapeAndTape {
                shape: next,
                i_tape: None,
                f_tape: None,
                g_tape: None,
            })
        } else {
            None
        };
        let sub_tape_ref = sub_tape.as_mut().unwrap_or(shape);

        // Recurse!
        if let Some(next_tile_size) = self.config.tile_sizes.get(depth + 1) {
            let n = tile_size / next_tile_size;

            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        self.render_tile_recurse(
                            sub_tape_ref,
                            depth + 1,
                            self.config.new_tile([
                                tile.corner[0] + i * next_tile_size,
                                tile.corner[1] + j * next_tile_size,
                                tile.corner[2] + k * next_tile_size,
                            ]),
                        );
                    }
                }
            }
        } else {
            self.render_tile_pixels(sub_tape_ref, tile_size, tile);
        };

        if let Some(mut sub_tape) = sub_tape {
            if let Some(i_tape) = sub_tape.i_tape.take() {
                if let Ok(t) = Arc::try_unwrap(i_tape) {
                    self.tape_storage.push(t.recycle());
                }
            }
            if let Some(f_tape) = sub_tape.f_tape.take() {
                self.tape_storage.push(f_tape.recycle());
            }
            if let Some(g_tape) = sub_tape.g_tape.take() {
                self.tape_storage.push(g_tape.recycle());
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
        tile: Tile<3>,
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

            // Skip pixels which are behind the image
            let zmax = (tile.corner[2] + tile_size).try_into().unwrap();
            if self.depth[o] >= zmax {
                continue;
            }

            // The matrix transformation is separable until the final
            // division by w.  We can precompute the XY-1 portion of the
            // multiplication here, since it's shared by every voxel in this
            // column of the image.
            let v = ((tile.corner[0] + i) as f32) * self.config.mat.column(0)
                + ((tile.corner[1] + j) as f32) * self.config.mat.column(1)
                + self.config.mat.column(3);

            for k in (0..tile_size).rev() {
                let v = v
                    + ((tile.corner[2] + k) as f32) * self.config.mat.column(2);

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

        // We're iterating over a few things simultaneously
        // - col refers to the xy position in the tile
        // - grad refers to points that we must do gradient evaluation on
        let mut grad = 0;
        let mut depth = out.chunks(tile_size);
        for col in 0..self.scratch.columns.len() {
            // Find the first set pixel in the column
            let depth = depth.next().unwrap();
            let k = match depth.iter().enumerate().find(|(_, d)| **d < 0.0) {
                Some((i, _)) => i,
                None => continue,
            };

            // Get X and Y values from the `columns` array.  Note that we can't
            // iterate over the array directly because we're also modifying it
            // (below)
            let xy = self.scratch.columns[col];
            let i = xy % tile_size;
            let j = xy / tile_size;

            // Flip Z value, since voxels are packed front-to-back
            let k = tile_size - 1 - k;

            // Set the depth of the pixel
            let o = self.config.tile_to_offset(tile, i, j);
            let z = (tile.corner[2] + k + 1).try_into().unwrap();
            assert!(self.depth[o] < z);
            self.depth[o] = z;

            // Prepare to do gradient rendering of this point.
            // We step one voxel above the surface to reduce
            // glitchiness on edges and corners, where rendering
            // inside the surface could pick the wrong normal.
            let p = self.config.mat.transform_point(&Point3::new(
                (tile.corner[0] + i) as f32,
                (tile.corner[1] + j) as f32,
                (tile.corner[2] + k + 1) as f32,
            ));
            self.scratch.x[grad] = p.x;
            self.scratch.y[grad] = p.y;
            self.scratch.z[grad] = p.z;

            // This can only be called once per iteration, so we'll
            // never overwrite parts of columns that are still used
            // by the outer loop
            self.scratch.columns[grad] = o;
            grad += 1;
        }

        if grad > 0 {
            let out = self
                .eval_grad_slice
                .eval(
                    shape.g_tape(&mut self.tape_storage),
                    &self.scratch.x,
                    &self.scratch.y,
                    &self.scratch.z,
                    &[],
                )
                .unwrap();

            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
                self.color[*o] = out[index].to_rgb().unwrap_or([255, 0, 0]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
struct Image {
    depth: Vec<u32>,
    color: Vec<[u8; 3]>,
}

impl Image {
    fn new(size: usize) -> Self {
        Self {
            depth: vec![0; size.pow(2)],
            color: vec![[0; 3]; size.pow(2)],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

fn worker<S: Shape>(
    mut shape: ShapeAndTape<S>,
    queues: &[Queue<3>],
    mut index: usize,
    config: &AlignedRenderConfig<3>,
) -> HashMap<[usize; 2], Image> {
    let mut out = HashMap::new();

    // Calculate maximum evaluation buffer size
    let buf_size = *config.tile_sizes.last().unwrap();
    let scratch = Scratch::new(buf_size);
    let mut w: Worker<S> = Worker {
        scratch,
        depth: vec![],
        color: vec![],
        config,

        eval_float_slice: S::FloatSliceEval::new(),
        eval_interval: S::IntervalEval::new(),
        eval_grad_slice: S::GradSliceEval::new(),

        tape_storage: vec![],
        shape_storage: vec![],
        workspace: Default::default(),
    };

    // Every thread has a set of tiles assigned to it, which are in Z-sorted
    // order (to encourage culling).  Once the thread finishes its tiles, it
    // begins stealing from other thread queues; if every single thread queue is
    // empty, then we return.
    let start = index;
    loop {
        while let Some(tile) = queues[index].next() {
            let image = out
                .remove(&[tile.corner[0], tile.corner[1]])
                .unwrap_or_else(|| Image::new(config.tile_sizes[0]));

            // Prepare to render, allocating space for a tile
            w.depth = image.depth;
            w.color = image.color;
            w.render_tile_recurse(&mut shape, 0, tile);

            // Steal the tile, replacing it with an empty vec
            let depth = std::mem::take(&mut w.depth);
            let color = std::mem::take(&mut w.color);
            out.insert(
                [tile.corner[0], tile.corner[1]],
                Image { depth, color },
            );
        }
        // Move on to the next thread's queue
        index = (index + 1) % queues.len();
        if index == start {
            break;
        }
    }

    out
}

////////////////////////////////////////////////////////////////////////////////

/// Renders the given tape into a 3D image according to the provided
/// configuration.
///
/// The tape provides the shape; the configuration supplies resolution,
/// transforms, etc.
///
/// This function is parameterized by shape type, which determines how we
/// perform evaluation.
pub fn render<S: Shape>(
    shape: S,
    config: &RenderConfig<3>,
) -> (Vec<u32>, Vec<[u8; 3]>) {
    let config = config.align();
    assert!(config.image_size % config.tile_sizes[0] == 0);
    for i in 0..config.tile_sizes.len() - 1 {
        assert!(config.tile_sizes[i] % config.tile_sizes[i + 1] == 0);
    }

    let mut tiles = vec![];
    for i in 0..config.image_size / config.tile_sizes[0] {
        for j in 0..config.image_size / config.tile_sizes[0] {
            for k in (0..config.image_size / config.tile_sizes[0]).rev() {
                tiles.push(config.new_tile([
                    i * config.tile_sizes[0],
                    j * config.tile_sizes[0],
                    k * config.tile_sizes[0],
                ]));
            }
        }
    }
    let tiles_per_thread = (tiles.len() / config.threads).max(1);
    let mut tile_queues = vec![];
    for ts in tiles.chunks(tiles_per_thread) {
        tile_queues.push(Queue::new(ts.to_vec()));
    }

    let i_tape = Arc::new(shape.interval_tape(Default::default()));

    // Special-case for single-threaded operation, to give simpler backtraces
    let out = if config.threads == 1 {
        let shape = ShapeAndTape {
            shape,
            i_tape: Some(i_tape),
            f_tape: None,
            g_tape: None,
        };
        worker::<S>(shape, tile_queues.as_slice(), 0, &config)
            .into_iter()
            .collect()
    } else {
        let config_ref = &config;
        std::thread::scope(|s| {
            let mut handles = vec![];
            let queues = tile_queues.as_slice();
            for i in 0..config.threads {
                let shape = ShapeAndTape {
                    shape: shape.clone(),
                    i_tape: Some(i_tape.clone()),
                    f_tape: None,
                    g_tape: None,
                };
                handles.push(
                    s.spawn(move || worker::<S>(shape, queues, i, config_ref)),
                );
            }
            let mut out = vec![];
            for h in handles {
                out.extend(h.join().unwrap().into_iter());
            }
            out
        })
    };

    let mut image_depth = vec![0; config.orig_image_size.pow(2)];
    let mut image_color = vec![[0; 3]; config.orig_image_size.pow(2)];
    for (tile, patch) in out.iter() {
        let mut index = 0;
        for j in 0..config.tile_sizes[0] {
            let y = j + tile[1];
            for i in 0..config.tile_sizes[0] {
                let x = i + tile[0];
                if x < config.orig_image_size && y < config.orig_image_size {
                    let o = (config.orig_image_size - y - 1)
                        * config.orig_image_size
                        + x;
                    if patch.depth[index] >= image_depth[o] {
                        image_color[o] = patch.color[index];
                        image_depth[o] = patch.depth[index];
                    }
                }
                index += 1;
            }
        }
    }
    (image_depth, image_color)
}
