//! 2D and 3D rendering
//!
//! The easiest way to render something is with
//! [`RenderConfig::run`](RenderConfig::run); you can also use the lower-level
//! functions ([`render2d`](render2d()) and [`render3d`](render3d())) for manual
//! control over the input tape.
use crate::eval::{BulkEvaluator, Shape, Tape, Trace, TracingEvaluator};
use std::sync::Arc;

mod config;
mod render2d;
mod render3d;

pub use config::RenderConfig;
pub use render2d::render as render2d;
pub use render3d::render as render3d;

pub use render2d::{BitRenderMode, DebugRenderMode, RenderMode, SdfRenderMode};

struct RenderHandle<S: Shape> {
    shape: S,

    i_tape: Option<Arc<<S::IntervalEval as TracingEvaluator>::Tape>>,
    f_tape: Option<<S::FloatSliceEval as BulkEvaluator>::Tape>,
    g_tape: Option<<S::GradSliceEval as BulkEvaluator>::Tape>,

    next: Option<(S::Trace, Box<Self>)>,
}

impl<S> RenderHandle<S>
where
    S: Shape,
{
    fn new(
        shape: S,
        i_tape: Arc<<S::IntervalEval as TracingEvaluator>::Tape>,
    ) -> Self {
        Self {
            shape,
            i_tape: Some(i_tape),
            f_tape: None,
            g_tape: None,
            next: None,
        }
    }
    fn i_tape(
        &mut self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::IntervalEval as TracingEvaluator>::Tape {
        self.i_tape.get_or_insert_with(|| {
            Arc::new(
                self.shape.interval_tape(storage.pop().unwrap_or_default()),
            )
        })
    }
    fn f_tape(
        &mut self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::FloatSliceEval as BulkEvaluator>::Tape {
        self.f_tape.get_or_insert_with(|| {
            self.shape
                .float_slice_tape(storage.pop().unwrap_or_default())
        })
    }
    fn g_tape(
        &mut self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::GradSliceEval as BulkEvaluator>::Tape {
        self.g_tape.get_or_insert_with(|| {
            self.shape
                .grad_slice_tape(storage.pop().unwrap_or_default())
        })
    }
    fn simplify(
        &mut self,
        trace: &S::Trace,
        workspace: &mut S::Workspace,
        shape_storage: &mut Vec<S::Storage>,
        tape_storage: &mut Vec<S::TapeStorage>,
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
            return &mut self.next.as_mut().unwrap().1;
        }
    }

    fn recycle(
        mut self,
        shape_storage: &mut Vec<S::Storage>,
        tape_storage: &mut Vec<S::TapeStorage>,
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
        tape_storage.extend(self.g_tape.map(Tape::recycle));
        tape_storage.extend(self.f_tape.map(Tape::recycle));

        // Do this step last because the evaluators may borrow the shape
        shape_storage.extend(self.shape.recycle());
    }
}
