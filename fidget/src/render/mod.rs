//! 2D and 3D rendering
//!
//! To render something, build a configuration object then call its `run`
//! function, e.g. [`ImageRenderConfig::run`] and [`VoxelRenderConfig::run`].
use crate::{
    eval::{BulkEvaluator, Function, Trace, TracingEvaluator},
    shape::{Shape, ShapeTape},
};
use std::sync::Arc;

mod config;
mod region;
mod render2d;
mod render3d;
mod view;

pub use config::{ImageRenderConfig, ThreadCount, VoxelRenderConfig};
pub use region::{ImageSize, RegionSize, VoxelSize};
pub use view::{View2, View3};

use render2d::render as render2d;
use render3d::render as render3d;

pub use render2d::{
    BitRenderMode, DebugRenderMode, RenderMode, SdfPixelRenderMode,
    SdfRenderMode,
};

/// A `RenderHandle` contains lazily-populated tapes for rendering
///
/// The tapes are stored as `Arc<..>`, so it can be cheaply cloned.
///
/// The most recent simplification is cached for reuse (if the trace matches).
pub struct RenderHandle<F: Function> {
    shape: Shape<F>,

    i_tape: Option<Arc<ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape>>>,
    f_tape: Option<Arc<ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape>>>,
    g_tape: Option<Arc<ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape>>>,

    next: Option<(F::Trace, Box<Self>)>,
}

impl<F: Function> Clone for RenderHandle<F> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            i_tape: self.i_tape.clone(),
            f_tape: self.f_tape.clone(),
            g_tape: self.g_tape.clone(),
            next: None,
        }
    }
}

impl<F: Function> RenderHandle<F> {
    /// Build a new [`RenderHandle`] for the given shape
    ///
    /// None of the tapes are populated here.
    pub fn new(shape: Shape<F>) -> Self {
        Self {
            shape,
            i_tape: None,
            f_tape: None,
            g_tape: None,
            next: None,
        }
    }

    /// Returns a tape for tracing interval evaluation
    pub fn i_tape(
        &mut self,
        storage: &mut Vec<F::TapeStorage>,
    ) -> &ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape> {
        self.i_tape.get_or_insert_with(|| {
            Arc::new(
                self.shape.interval_tape(storage.pop().unwrap_or_default()),
            )
        })
    }

    /// Returns a tape for bulk float evaluation
    pub fn f_tape(
        &mut self,
        storage: &mut Vec<F::TapeStorage>,
    ) -> &ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape> {
        self.f_tape.get_or_insert_with(|| {
            Arc::new(
                self.shape
                    .float_slice_tape(storage.pop().unwrap_or_default()),
            )
        })
    }

    /// Returns a tape for bulk gradient evaluation
    pub fn g_tape(
        &mut self,
        storage: &mut Vec<F::TapeStorage>,
    ) -> &ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape> {
        self.g_tape.get_or_insert_with(|| {
            Arc::new(
                self.shape
                    .grad_slice_tape(storage.pop().unwrap_or_default()),
            )
        })
    }

    /// Simplifies the shape with the given trace
    ///
    /// As an internal optimization, this may reuse a previous simplification if
    /// the trace matches.
    pub fn simplify(
        &mut self,
        trace: &F::Trace,
        workspace: &mut F::Workspace,
        shape_storage: &mut Vec<F::Storage>,
        tape_storage: &mut Vec<F::TapeStorage>,
    ) -> &mut Self {
        // Free self.next if it doesn't match our new set of choices
        let mut trace_storage = if let Some(neighbor) = &self.next {
            if &neighbor.0 != trace {
                let (trace, neighbor) = self.next.take().unwrap();
                neighbor.recycle(shape_storage, tape_storage);
                Some(trace)
                // continue with simplification
            } else {
                None
            }
        } else {
            None
        };

        // Ordering is a little weird here, to persuade the borrow checker to be
        // happy about things.  At this point, `next` is empty if we can't reuse
        // it, and `Some(..)` if we can.
        if self.next.is_none() {
            let s = shape_storage.pop().unwrap_or_default();
            let next = self.shape.simplify(trace, s, workspace).unwrap();
            if next.size() >= self.shape.size() {
                // Optimization: if the simplified shape isn't any shorter, then
                // don't use it (this saves time spent generating tapes)
                shape_storage.extend(next.recycle());
                self
            } else {
                assert!(self.next.is_none());
                if let Some(t) = trace_storage.as_mut() {
                    t.copy_from(trace);
                } else {
                    trace_storage = Some(trace.clone());
                }
                self.next = Some((
                    trace_storage.unwrap(),
                    Box::new(RenderHandle {
                        shape: next,
                        i_tape: None,
                        f_tape: None,
                        g_tape: None,
                        next: None,
                    }),
                ));
                &mut self.next.as_mut().unwrap().1
            }
        } else {
            &mut self.next.as_mut().unwrap().1
        }
    }

    /// Recycles the entire handle into the given storage vectors
    pub fn recycle(
        mut self,
        shape_storage: &mut Vec<F::Storage>,
        tape_storage: &mut Vec<F::TapeStorage>,
    ) {
        // Recycle the child first, in case it borrowed from us
        if let Some((_trace, shape)) = self.next.take() {
            shape.recycle(shape_storage, tape_storage);
        }

        if let Some(i_tape) = self.i_tape.take() {
            if let Ok(i_tape) = Arc::try_unwrap(i_tape) {
                tape_storage.push(i_tape.recycle());
            }
        }
        if let Some(g_tape) = self.g_tape.take() {
            if let Ok(g_tape) = Arc::try_unwrap(g_tape) {
                tape_storage.push(g_tape.recycle());
            }
        }
        if let Some(f_tape) = self.f_tape.take() {
            if let Ok(f_tape) = Arc::try_unwrap(f_tape) {
                tape_storage.push(f_tape.recycle());
            }
        }

        // Do this step last because the evaluators may borrow the shape
        shape_storage.extend(self.shape.recycle());
    }
}
