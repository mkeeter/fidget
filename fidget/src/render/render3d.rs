//! 3D bitmap rendering / rasterization
use crate::{
    eval::{
        float_slice::{
            FloatSliceEval, FloatSliceEvalData, FloatSliceEvalStorage,
        },
        grad_slice::{GradSliceEval, GradSliceEvalData, GradSliceEvalStorage},
        interval::{IntervalEval, IntervalEvalData},
        types::{Grad, Interval},
        Choice, EvaluatorStorage, Family,
    },
    render::config::{AlignedRenderConfig, Queue, RenderConfig, Tile},
    vm::{InnerTape, Tape},
};

use nalgebra::{Point3, Vector3};
use std::collections::BTreeMap;

////////////////////////////////////////////////////////////////////////////////
/// Tiny extension trait to add a checked counterpart to `.take().unwrap()`
trait Give<V> {
    /// Gives the value `v` to us.
    ///
    /// If we already contain a value, panics.
    fn give(&mut self, v: V);
}

impl<V> Give<V> for Option<V> {
    #[inline]
    fn give(&mut self, v: V) {
        assert!(self.is_none());
        *self = Some(v)
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Scratch<F: Family> {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,

    data_float: FloatSliceEvalData<F>,
    data_interval: IntervalEvalData<F>,
    data_grad: GradSliceEvalData<F>,

    /// Depth of each column
    columns: Vec<usize>,
}

impl<F: Family> Scratch<F> {
    fn new(tile_size: usize) -> Self {
        let size2 = tile_size.pow(2);
        let size3 = tile_size.pow(3);
        Self {
            x: vec![0.0; size3],
            y: vec![0.0; size3],
            z: vec![0.0; size3],

            data_float: Default::default(),
            data_interval: Default::default(),
            data_grad: Default::default(),

            columns: vec![0; size2],
        }
    }
    fn eval_s<'a>(
        &mut self,
        f: &mut FloatSliceEval<F>,
        size: usize,
        data: &'a mut FloatSliceEvalData<F>,
    ) -> &'a [f32] {
        f.eval_with(
            &self.x[0..size],
            &self.y[0..size],
            &self.z[0..size],
            &[],
            data,
        )
        .unwrap()
    }
    fn eval_g<'a>(
        &mut self,
        f: &mut GradSliceEval<F>,
        size: usize,
        data: &'a mut GradSliceEvalData<F>,
    ) -> &'a [Grad] {
        f.eval_with(
            &self.x[0..size],
            &self.y[0..size],
            &self.z[0..size],
            &[],
            data,
        )
        .unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Evaluators<I: Family> {
    level: usize,
    tape: Tape<I>,
    interval: Option<IntervalEval<I>>,
    float_slice: Option<FloatSliceEval<I>>,
    grad: Option<GradSliceEval<I>>,
}

////////////////////////////////////////////////////////////////////////////////

struct Worker<'a, I: Family> {
    config: &'a AlignedRenderConfig<3>,

    /// Reusable workspace for evaluation, to minimize allocation
    scratch: Scratch<I>,

    /// Output images for this specific tile
    depth: Vec<u32>,
    color: Vec<[u8; 3]>,

    /// Storage for float slice evaluators
    ///
    /// This has `n + 1` slots, where `n` is the number of render levels; one
    /// level for each round of interval evaluation, and one for the voxel
    /// stage.  Don't be mislead by the use of `Option` here; slots should
    /// always be populated except when someone is borrowing them, and borrowers
    /// must give them back (using the `.give()` extension method)
    ///
    /// This means that we'll often use `.take().unwrap()`, to ensure that we
    /// didn't mess up our invariants.
    float_storage: Vec<Option<FloatSliceEvalStorage<I>>>,

    /// Equivalent to `float_storage`, for gradient evaluators
    grad_storage: Vec<Option<GradSliceEvalStorage<I>>>,

    /// Storage for interval evaluators.
    ///
    /// This has `n` slots, where `n` is the number of render levels.  The first
    /// slot is always `None`, because philosophically, it represents the
    /// storage of `i_handle` (but `i_handle` is shared between multiple
    /// threads, so we never actually reclaim its storage).
    interval_storage: Vec<
        Option<<<I as Family>::IntervalEval as EvaluatorStorage<I>>::Storage>,
    >,

    /// Spare tapes to avoid allocation churn
    ///
    /// This has `n + 1` slots: one for each render level, and one for per-voxel
    /// evaluation (after the final tape simplification).
    spare_tapes: Vec<Option<InnerTape<I>>>,
}

impl<I: Family> Worker<'_, I> {
    fn reclaim_storage(&mut self, eval: Evaluators<I>) -> InnerTape<I> {
        if let Some(float) = eval.float_slice {
            self.float_storage[eval.level].give(float.take().unwrap());
        }
        if let Some(interval) = eval.interval {
            self.interval_storage[eval.level].give(interval.take().unwrap());
        }
        if let Some(grad) = eval.grad {
            self.grad_storage[eval.level].give(grad.take().unwrap());
        }

        // Use Err to indicate that we have to shorten the tape
        // (basically a bootleg Either)
        eval.tape.take().unwrap()
    }

    fn render_tile_recurse(
        &mut self,
        eval: &mut Evaluators<I>,
        sibling: Option<(Vec<Choice>, Evaluators<I>)>,
        level: usize,
        tile: Tile<3>,
    ) -> Option<(Vec<Choice>, Evaluators<I>)> {
        // Early exit if every single pixel is filled
        let tile_size = self.config.tile_sizes[level];
        let fill_z = (tile.corner[2] + tile_size + 1).try_into().unwrap();
        if (0..tile_size).all(|y| {
            let i = self.config.tile_to_offset(tile, 0, y);
            (0..tile_size).all(|x| self.depth[i + x] >= fill_z)
        }) {
            return sibling;
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

        let mut data_interval = std::mem::take(&mut self.scratch.data_interval);
        let (i, simplify) = eval
            .interval
            .get_or_insert_with(|| {
                let storage = self.interval_storage[eval.level].take().unwrap();
                eval.tape.new_interval_evaluator_with_storage(storage)
            })
            .eval_with(x, y, z, &[], &mut data_interval)
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
            // TODO: handle gradients here as well?
            self.scratch.data_interval = data_interval;
            return sibling;
        } else if i.lower() > 0.0 {
            self.scratch.data_interval = data_interval;
            return sibling;
        }

        // Calculate a simplified tape, reverting to the parent tape if the
        // simplified tape isn't any shorter.
        let (mut sub_eval, mut prev_sibling) = if let Some(simplify) =
            simplify.as_ref()
        {
            // See if our previous tape shortening used the exact same set of
            // choices; in that case, then we can reuse it.
            //
            // This is likely because of spatial locality!
            let res = if let Some((choices, sibling_eval)) = sibling {
                if choices == simplify.choices() {
                    Ok((choices, sibling_eval))
                } else {
                    // The sibling didn't make the same choices, so we'll tear
                    // it down for parts and build our own tape here.
                    //
                    // Release all of the sibling evaluators, which should free
                    // up the tape for reuse.
                    let tape = self.reclaim_storage(sibling_eval);

                    // Use Err to indicate that we have to shorten the tape
                    // (basically a bootleg Either)
                    Err((tape, choices))
                }
            } else {
                Err((self.spare_tapes[eval.level].take().unwrap(), vec![]))
            };

            let out = match res {
                Ok(out) => Some(out),
                Err((tape, mut choices)) => {
                    let sub_tape = simplify.simplify_with(tape);

                    if sub_tape.len() < eval.tape.len() {
                        choices
                            .resize(simplify.choices().len(), Choice::Unknown);
                        choices.copy_from_slice(simplify.choices());
                        Some((
                            choices,
                            Evaluators {
                                level: eval.level + 1,
                                tape: sub_tape,
                                interval: None,
                                float_slice: None,
                                grad: None,
                            },
                        ))
                    } else {
                        // Immediately return the spare tape
                        //
                        // TODO: reuse choices
                        self.spare_tapes[eval.level]
                            .give(sub_tape.take().unwrap());

                        // Alas, the sibling has been consumed, so we can't
                        // reuse it at all.
                        None
                    }
                }
            };
            (out, None) // prev_sibling is always consumed
        } else {
            // If we're not simplifying this tape, then keep the sibling around,
            // since it's still useful.
            (None, sibling)
        };

        // Return `data_interval` to our scratch buffer
        self.scratch.data_interval = data_interval;

        // At this point, only one of `sub_eval` and `prev_sibling` can be
        // `Some`; both could also be `None`, if we consumed `prev_sibling` then
        // realized that it didn't shorten the tape.
        //
        // There's probably a better abstraction here!

        // Recurse!
        let new_sibling = if let Some(&next_size) =
            self.config.tile_sizes.get(level + 1)
        {
            let n = tile_size / next_size;

            // If we didn't simplify, then we can attempt to reuse the sibling
            // passed into this function call, which is still valid.
            let mut sibling = if sub_eval.is_none() {
                prev_sibling.take()
            } else {
                assert!(prev_sibling.is_none());
                None
            };

            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        sibling = self.render_tile_recurse(
                            sub_eval.as_mut().map(|c| &mut c.1).unwrap_or(eval),
                            sibling,
                            level + 1,
                            self.config.new_tile([
                                tile.corner[0] + i * next_size,
                                tile.corner[1] + j * next_size,
                                tile.corner[2] + k * next_size,
                            ]),
                        );
                    }
                }
            }
            sibling
        } else {
            self.render_tile_pixels(
                sub_eval.as_mut().map(|c| &mut c.1).unwrap_or(eval),
                tile_size,
                tile,
            );
            // We never generate a sibling from pixel rendering, since we don't
            // simplify the tape any further, so return the previous sibling
            // (if one existed)
            None
        };

        // Since we simplified the tape here, the sibling is only valid for
        // calls that we're making here.  Alas, we are done with these calls, so
        // we have to recycle it here.
        if let Some((choices, sub)) = sub_eval {
            assert!(prev_sibling.is_none());
            if let Some((_choices, sibling)) = new_sibling {
                assert!(sibling.level == sub.level + 1);
                let tape = self.reclaim_storage(sibling);
                self.spare_tapes[sub.level].give(tape);
                // TODO: reuse _choices
            }
            // We return our own subtape, which can be a sibling subtape in
            // future calls.
            Some((choices, sub))
        } else if prev_sibling.is_some() {
            // new_sibling only exists in a branch where prev_sibling is
            // consumed to make it, so they can't coexist here.
            assert!(new_sibling.is_none());
            prev_sibling
        } else {
            // Otherwise, since we didn't simplify here, the sibling
            // evaluator (if present) is also valid for our parent
            new_sibling
        }
    }

    fn render_tile_pixels(
        &mut self,
        eval: &mut Evaluators<I>,
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

        // Reuse the FloatSliceFunc handle passed in, or build one if it
        // wasn't already available (which makes it available to siblings)
        let func = eval.float_slice.get_or_insert_with(|| {
            let storage = self.float_storage[eval.level].take().unwrap();
            eval.tape.new_float_slice_evaluator_with_storage(storage)
        });

        // Borrow the scratch data, returning it at the end of the function
        let mut data_float = std::mem::take(&mut self.scratch.data_float);
        let out = self.scratch.eval_s(func, size, &mut data_float);

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
            // Reuse the FloatSliceFunc handle passed in, or build one if it
            // wasn't already available (which makes it available to siblings)
            let func = eval.grad.get_or_insert_with(|| {
                let storage = self.grad_storage[eval.level].take().unwrap();
                eval.tape.new_grad_slice_evaluator_with_storage(storage)
            });
            let mut data_grad = std::mem::take(&mut self.scratch.data_grad);
            let out_grad = self.scratch.eval_g(func, grad, &mut data_grad);

            for (index, o) in self.scratch.columns[0..grad].iter().enumerate() {
                self.color[*o] =
                    out_grad[index].to_rgb().unwrap_or([255, 0, 0]);
            }
            self.scratch.data_grad = data_grad;
        }

        // Return the scratch data
        self.scratch.data_float = data_float;
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

fn worker<I: Family>(
    tape: &Tape<I>,
    queues: &[Queue<3>],
    mut index: usize,
    config: &AlignedRenderConfig<3>,
) -> BTreeMap<[usize; 2], Image> {
    let mut out = BTreeMap::new();

    // Calculate maximum evaluation buffer size
    let buf_size = *config.tile_sizes.last().unwrap();
    let scratch = Scratch::new(buf_size);
    let mut w: Worker<I> = Worker {
        scratch,
        depth: vec![],
        color: vec![],
        config,

        // Notice that these are all populated with Some(...)!
        float_storage: (0..=config.tile_sizes.len())
            .map(|_| Some(Default::default()))
            .collect(),
        grad_storage: (0..=config.tile_sizes.len())
            .map(|_| Some(Default::default()))
            .collect(),

        interval_storage: (0..config.tile_sizes.len())
            .map(|_| Some(Default::default()))
            .collect(),

        spare_tapes: (0..=config.tile_sizes.len())
            .map(|_| Some(tape.data().clone().into()))
            .collect(),
    };

    // Every thread has a set of tiles assigned to it, which are in Z-sorted
    // order (to encourage culling).  Once the thread finishes its tiles, it
    // begins stealing from other thread queues; if every single thread queue is
    // empty, then we return.
    let start = index;
    let i_handle = tape.new_interval_evaluator();
    loop {
        while let Some(tile) = queues[index].next() {
            let image = out
                .remove(&[tile.corner[0], tile.corner[1]])
                .unwrap_or_else(|| Image::new(config.tile_sizes[0]));

            // Prepare to render, allocating space for a tile
            w.depth = image.depth;
            w.color = image.color;
            let mut eval = Evaluators {
                level: 0,
                tape: i_handle.tape(),
                interval: Some(i_handle.clone()),
                float_slice: None,
                grad: None,
            };
            if let Some((_, e)) =
                w.render_tile_recurse(&mut eval, None, 0, tile)
            {
                if let Some(i) = e.interval {
                    w.interval_storage[1].give(i.take().unwrap());
                }
                if let Some(f) = e.float_slice {
                    w.float_storage[1].give(f.take().unwrap());
                }
                if let Some(g) = e.grad {
                    w.grad_storage[1].give(g.take().unwrap());
                }
                w.spare_tapes[0].give(e.tape.take().unwrap());
            }

            // Check our invariants, to make sure that everyone gave back their
            // storage data when complete.
            assert!(w.interval_storage.iter().all(Option::is_some));
            assert!(w.float_storage.iter().all(Option::is_some));
            assert!(w.grad_storage.iter().all(Option::is_some));
            assert!(w.spare_tapes.iter().all(Option::is_some));

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
/// This function is parameterized by both evaluator family, which determines
/// how we perform evaluation.
pub fn render<I: Family>(
    tape: Tape<I>,
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

    // Special-case for single-threaded operation, to give simpler backtraces
    let out = if config.threads == 1 {
        worker::<I>(&tape, tile_queues.as_slice(), 0, &config)
            .into_iter()
            .collect()
    } else {
        let config_ref = &config;
        std::thread::scope(|s| {
            let mut handles = vec![];
            let queues = tile_queues.as_slice();
            let tape_ref = &tape;
            for i in 0..config.threads {
                handles.push(s.spawn(move || {
                    worker::<I>(tape_ref, queues, i, config_ref)
                }));
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
