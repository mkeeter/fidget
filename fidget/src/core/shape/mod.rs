//! Traits and data structures for shape evaluation
//!
//! There are a bunch of things in here, but the most important trait is
//! [`Shape`], followed by the evaluator traits ([`BulkEvaluator`] and
//! [`TracingEvaluator`]).
//!
//! ```rust
//! use fidget::vm::VmShape;
//! use fidget::context::Context;
//! use fidget::shape::{TracingEvaluator, Shape, MathShape, EzShape};
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let shape = VmShape::new(&ctx, x)?;
//!
//! // Let's build a single point evaluator:
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! let (value, _trace) = eval.eval(&tape, 0.25, 0.0, 0.0)?;
//! assert_eq!(value, 0.25);
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! Note that the traits here mirror the ones in ones in
//! [`fidget::eval`](crate::eval), but are specialized to operate on `x, y, z`
//! arguments (rather than taking arbitrary numbers of variables).  It is
//! recommended to import the traits from either one or the other, to avoid
//! ambiguity.

use crate::{
    context::{Context, Node},
    eval::{self, Trace},
    types::{Grad, Interval},
    Error,
};

mod bounds;
mod bulk;
mod tracing;
mod transform;

// Re-export a few things
pub use bounds::Bounds;
pub use bulk::BulkEvaluator;
pub use tracing::TracingEvaluator;
pub use transform::TransformedShape;

/// A shape represents an implicit surface
///
/// It is mostly agnostic to _how_ that surface is represented; we simply
/// require that the shape can generate evaluators of various kinds.
///
/// Shapes are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
pub trait Shape: Send + Sync + Clone {
    /// Associated type traces collected during tracing evaluation
    ///
    /// This type must implement [`Eq`] so that traces can be compared; calling
    /// [`Shape::simplify`] with traces that compare equal should produce an
    /// identical result and may be cached.
    type Trace: Clone + Eq + Send + Trace;

    /// Associated type for storage used by the shape itself
    type Storage: Default + Send;

    /// Associated type for workspace used during shape simplification
    type Workspace: Default + Send;

    /// Associated type for storage used by tapes
    ///
    /// For simplicity, we require that every tape use the same type for storage.
    /// This could change in the future!
    type TapeStorage: Default + Send;

    /// Associated type for single-point tracing evaluation
    type PointEval: TracingEvaluator<
            Data = f32,
            Trace = Self::Trace,
            TapeStorage = Self::TapeStorage,
        > + Send
        + Sync;

    /// Builds a new point evaluator
    fn new_point_eval() -> Self::PointEval {
        Self::PointEval::new()
    }

    /// Associated type for single interval tracing evaluation
    type IntervalEval: TracingEvaluator<
            Data = Interval,
            Trace = Self::Trace,
            TapeStorage = Self::TapeStorage,
        > + Send
        + Sync;

    /// Builds a new interval evaluator
    fn new_interval_eval() -> Self::IntervalEval {
        Self::IntervalEval::new()
    }

    /// Associated type for evaluating many points in one call
    type FloatSliceEval: BulkEvaluator<Data = f32, TapeStorage = Self::TapeStorage>
        + Send
        + Sync;

    /// Builds a new float slice evaluator
    fn new_float_slice_eval() -> Self::FloatSliceEval {
        Self::FloatSliceEval::new()
    }

    /// Associated type for evaluating many gradients in one call
    type GradSliceEval: BulkEvaluator<Data = Grad, TapeStorage = Self::TapeStorage>
        + Send
        + Sync;

    /// Builds a new gradient slice evaluator
    fn new_grad_slice_eval() -> Self::GradSliceEval {
        Self::GradSliceEval::new()
    }

    /// Returns an evaluation tape for a point evaluator
    fn point_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::PointEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for an interval evaluator
    fn interval_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn float_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn grad_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape;

    /// Computes a simplified tape using the given trace, and reusing storage
    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, Error>
    where
        Self: Sized;

    /// Attempt to reclaim storage from this shape
    ///
    /// This may fail, because shapes are `Clone` and are often implemented
    /// using an `Arc` around a heavier data structure.
    fn recycle(self) -> Option<Self::Storage>;

    /// Returns a size associated with this shape
    ///
    /// This is underspecified and only used for unit testing; for tape-based
    /// shapes, it's typically the length of the tape,
    fn size(&self) -> usize;

    /// Associated type returned when applying a transform
    ///
    /// This is normally [`TransformedShape<Self>`](TransformedShape), but if
    /// `Self` is already `TransformedShape`, then the transform is stacked
    /// (instead of creating a wrapped object).
    type TransformedShape: Shape;

    /// Returns a shape with the given transform applied
    fn apply_transform(
        self,
        mat: nalgebra::Matrix4<f32>,
    ) -> <Self as Shape>::TransformedShape;
}

/// Extension trait for working with a shape without thinking much about memory
///
/// All of the [`Shape`] functions that use significant amounts of memory
/// pedantically require you to pass in storage for reuse.  This trait allows
/// you to ignore that, at the cost of performance; we require that all storage
/// types implement [`Default`], so these functions do the boilerplate for you.
///
/// This trait is automatically implemented for every [`Shape`], but must be
/// imported separately as a speed-bump to using it everywhere.
pub trait EzShape: Shape {
    /// Returns an evaluation tape for a point evaluator
    fn ez_point_tape(&self) -> <Self::PointEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for an interval evaluator
    fn ez_interval_tape(
        &self,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn ez_float_slice_tape(
        &self,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn ez_grad_slice_tape(
        &self,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape;

    /// Computes a simplified tape using the given trace
    fn ez_simplify(&self, trace: &Self::Trace) -> Result<Self, Error>
    where
        Self: Sized;
}

impl<S: Shape> EzShape for S {
    fn ez_point_tape(&self) -> <Self::PointEval as TracingEvaluator>::Tape {
        self.point_tape(Default::default())
    }

    fn ez_interval_tape(
        &self,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape {
        self.interval_tape(Default::default())
    }

    fn ez_float_slice_tape(
        &self,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape {
        self.float_slice_tape(Default::default())
    }

    fn ez_grad_slice_tape(
        &self,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape {
        self.grad_slice_tape(Default::default())
    }

    fn ez_simplify(&self, trace: &Self::Trace) -> Result<Self, Error> {
        let mut workspace = Default::default();
        self.simplify(trace, Default::default(), &mut workspace)
    }
}

/// Hints for how to render this particular type
pub trait RenderHints {
    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> &'static [usize];

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> &'static [usize];

    /// Indicates whether we run tape simplification at the given cell depth
    /// during meshing.
    ///
    /// By default, this is always true; for evaluators where simplification is
    /// more expensive than evaluation (i.e. the JIT), it may only be true at
    /// certain depths.
    fn simplify_tree_during_meshing(_d: usize) -> bool {
        true
    }
}

/// A [`Shape`] which can be built from a math expression
pub trait MathShape {
    /// Builds a new shape from the given node with default (X, Y, Z) axes
    fn new(ctx: &mut Context, node: Node) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let axes = ctx.axes();
        Self::new_with_axes(ctx, node, axes)
    }

    /// Builds a new shape from the given context, node, and axes
    fn new_with_axes(
        ctx: &Context,
        node: Node,
        axes: [Node; 3],
    ) -> Result<Self, Error>
    where
        Self: Sized;

    /// Helper function to build a shape from a [`Tree`](crate::context::Tree)
    ///
    /// This function uses the default (X, Y, Z) axes
    fn from_tree(t: &crate::context::Tree) -> Self
    where
        Self: Sized,
    {
        let mut ctx = Context::new();
        let node = ctx.import(t);
        Self::new(&mut ctx, node).unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Wrapper to convert a [`Function`](fidget::eval::Function) into a [`Shape`]
/// for evaluation.
#[derive(Clone)]
pub struct FunctionShape<F> {
    /// Wrapped function
    f: F,

    /// Index of x, y, z axes within the function's variable list
    axes: [usize; 3],
}

impl<F: eval::Function + Clone> Shape for FunctionShape<F> {
    type Trace = <F as eval::Function>::Trace;
    type Storage = <F as eval::Function>::Storage;
    type Workspace = <F as eval::Function>::Workspace;
    type TapeStorage = <F as eval::Function>::TapeStorage;

    type PointEval = FunctionShapeTracingEval<<F as eval::Function>::PointEval>;
    type IntervalEval =
        FunctionShapeTracingEval<<F as eval::Function>::IntervalEval>;
    type FloatSliceEval =
        FunctionShapeBulkEval<<F as eval::Function>::FloatSliceEval>;
    type GradSliceEval =
        FunctionShapeBulkEval<<F as eval::Function>::GradSliceEval>;

    fn point_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::PointEval as TracingEvaluator>::Tape {
        self.f.point_tape(storage)
    }

    fn interval_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape {
        self.f.interval_tape(storage)
    }

    fn float_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape {
        self.f.float_slice_tape(storage)
    }

    fn grad_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape {
        self.f.grad_slice_tape(storage)
    }

    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let f = self.f.simplify(trace, storage, workspace)?;
        Ok(Self { f, axes: self.axes })
    }

    fn recycle(self) -> Option<Self::Storage> {
        self.f.recycle()
    }

    fn size(&self) -> usize {
        self.f.size()
    }

    type TransformedShape = TransformedShape<Self>;

    fn apply_transform(
        self,
        mat: nalgebra::Matrix4<f32>,
    ) -> <Self as Shape>::TransformedShape {
        TransformedShape::new(self, mat)
    }

    // todo
}

impl<F: eval::MathFunction> MathShape for FunctionShape<F> {
    fn new_with_axes(
        ctx: &Context,
        node: Node,
        axes: [Node; 3],
    ) -> Result<Self, Error> {
        let f = F::new(ctx, node)?; // TODO get a varmap here
        Ok(Self { f, axes: [0, 1, 2] })
    }
}

impl<F: RenderHints> RenderHints for FunctionShape<F> {
    fn tile_sizes_3d() -> &'static [usize] {
        F::tile_sizes_3d()
    }

    fn tile_sizes_2d() -> &'static [usize] {
        F::tile_sizes_2d()
    }

    fn simplify_tree_during_meshing(d: usize) -> bool {
        F::simplify_tree_during_meshing(d)
    }
}

/// Wrapper struct to convert from [`eval::TracingEvaluator`] to
/// [`shape::TracingEvaluator`](TracingEvaluator)
#[derive(Default)]
pub struct FunctionShapeTracingEval<E> {
    eval: E,

    /// Index of x, y, z axes within the function's variable list
    axes: [usize; 3],
}

impl<E: eval::TracingEvaluator> TracingEvaluator
    for FunctionShapeTracingEval<E>
{
    type Data = E::Data;
    type Tape = E::Tape;
    type TapeStorage = E::TapeStorage;
    type Trace = E::Trace;

    fn eval<F: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
    ) -> Result<(Self::Data, Option<&Self::Trace>), Error> {
        let mut vars = [None, None, None];
        vars[self.axes[0]] = Some(x.into());
        vars[self.axes[1]] = Some(y.into());
        vars[self.axes[2]] = Some(z.into());

        // TODO make this error?  Where do we maintain the `axes` invariants?
        let vars = vars.map(Option::unwrap);
        self.eval.eval(tape, vars.as_slice())
    }
    // todo
}

/// Wrapper struct to convert from [`eval::BulkEvaluator`] to
/// [`shape::TracingEvaluator`](BulkEvaluator)
#[derive(Default)]
pub struct FunctionShapeBulkEval<E> {
    eval: E,

    /// Index of x, y, z axes within the function's variable list
    axes: [usize; 3],
}

impl<E: eval::BulkEvaluator> BulkEvaluator for FunctionShapeBulkEval<E> {
    type Data = E::Data;
    type Tape = E::Tape;
    type TapeStorage = E::TapeStorage;

    fn new() -> Self {
        Self::default()
    }

    fn eval(
        &mut self,
        tape: &Self::Tape,
        x: &[Self::Data],
        y: &[Self::Data],
        z: &[Self::Data],
    ) -> Result<&[Self::Data], Error> {
        let mut vars = [None, None, None];
        vars[self.axes[0]] = Some(x);
        vars[self.axes[1]] = Some(y);
        vars[self.axes[2]] = Some(z);

        // TODO make this error?  Where do we maintain the `axes` invariants?
        let vars = vars.map(Option::unwrap);
        self.eval.eval(tape, &vars)
    }
}
