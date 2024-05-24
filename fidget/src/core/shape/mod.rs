//! Traits and data structures for shape evaluation
//!
//! There are a bunch of things in here, but the most important trait is
//! [`Shape`], followed by the evaluator traits ([`BulkEvaluator`] and
//! [`TracingEvaluator`]).
//!
//! ```rust
//! use fidget::vm::VmShape;
//! use fidget::context::Context;
//! use fidget::shape::EzShape;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let shape = VmShape::new(&mut ctx, x)?;
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
    context::{Context, Node, Tree},
    eval::{BulkEvaluator, Function, MathFunction, Tape, TracingEvaluator},
    types::{Grad, Interval},
    Error,
};
use nalgebra::{Matrix4, Point3};

mod bounds;

// Re-export a few things
pub use bounds::Bounds;

/// A shape represents an implicit surface
///
/// It is mostly agnostic to _how_ that surface is represented, wrapping a
/// [`Function`](Function) and a set of axes.
///
/// Shapes are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
#[derive(Clone)]
pub struct Shape<F> {
    /// Wrapped function
    f: F,

    /// Index of x, y, z axes within the function's variable list (if present)
    axes: [Option<usize>; 3],

    /// Optional transform to apply to the shape
    transform: Option<Matrix4<f32>>,
}

impl<F: Function + Clone> Shape<F> {
    /// Builds a new point evaluator
    pub fn new_point_eval() -> ShapeTracingEval<F::PointEval> {
        ShapeTracingEval {
            eval: F::PointEval::default(),
        }
    }

    /// Builds a new interval evaluator
    pub fn new_interval_eval() -> ShapeTracingEval<F::IntervalEval> {
        ShapeTracingEval {
            eval: F::IntervalEval::default(),
        }
    }

    /// Builds a new float slice evaluator
    pub fn new_float_slice_eval() -> ShapeBulkEval<F::FloatSliceEval> {
        ShapeBulkEval {
            eval: F::FloatSliceEval::default(),
            xs: vec![],
            ys: vec![],
            zs: vec![],
        }
    }

    /// Builds a new gradient slice evaluator
    pub fn new_grad_slice_eval() -> ShapeBulkEval<F::GradSliceEval> {
        ShapeBulkEval {
            eval: F::GradSliceEval::default(),
            xs: vec![],
            ys: vec![],
            zs: vec![],
        }
    }

    /// Returns an evaluation tape for a point evaluator
    pub fn point_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::PointEval as TracingEvaluator>::Tape> {
        ShapeTape {
            tape: self.f.point_tape(storage),
            axes: self.axes,
            transform: self.transform,
        }
    }

    /// Returns an evaluation tape for a interval evaluator
    pub fn interval_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape> {
        ShapeTape {
            tape: self.f.interval_tape(storage),
            axes: self.axes,
            transform: self.transform,
        }
    }

    /// Returns an evaluation tape for a float slice evaluator
    pub fn float_slice_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape> {
        ShapeTape {
            tape: self.f.float_slice_tape(storage),
            axes: self.axes,
            transform: self.transform,
        }
    }

    /// Returns an evaluation tape for a gradient slice evaluator
    pub fn grad_slice_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape> {
        ShapeTape {
            tape: self.f.grad_slice_tape(storage),
            axes: self.axes,
            transform: self.transform,
        }
    }

    /// Computes a simplified tape using the given trace, and reusing storage
    pub fn simplify(
        &self,
        trace: &F::Trace,
        storage: F::Storage,
        workspace: &mut F::Workspace,
    ) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let f = self.f.simplify(trace, storage, workspace)?;
        Ok(Self {
            f,
            axes: self.axes,
            transform: self.transform,
        })
    }

    /// Attempt to reclaim storage from this shape
    ///
    /// This may fail, because shapes are `Clone` and are often implemented
    /// using an `Arc` around a heavier data structure.
    pub fn recycle(self) -> Option<F::Storage> {
        self.f.recycle()
    }

    /// Returns a size associated with this shape
    ///
    /// This is underspecified and only used for unit testing; for tape-based
    /// shapes, it's typically the length of the tape,
    pub fn size(&self) -> usize {
        self.f.size()
    }

    /// Borrows the inner [`Function`](Function) object
    pub fn inner(&self) -> &F {
        &self.f
    }

    /// Borrows the inner axis mapping
    pub fn axes(&self) -> &[Option<usize>; 3] {
        &self.axes
    }

    /// Raw constructor
    pub fn new_raw(f: F, axes: [Option<usize>; 3]) -> Self {
        Self {
            f,
            axes,
            transform: None,
        }
    }
}

impl<F: RenderHints> Shape<F> {
    pub fn tile_sizes_3d() -> &'static [usize] {
        F::tile_sizes_3d()
    }

    pub fn tile_sizes_2d() -> &'static [usize] {
        F::tile_sizes_2d()
    }

    pub fn simplify_tree_during_meshing(d: usize) -> bool {
        F::simplify_tree_during_meshing(d)
    }
}

impl<F> Shape<F> {
    pub fn apply_transform(mut self, mat: Matrix4<f32>) -> Self {
        if let Some(prev) = self.transform.as_mut() {
            *prev *= mat;
        } else {
            self.transform = Some(mat);
        }
        self
    }
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
pub trait EzShape<F: Function> {
    /// Returns an evaluation tape for a point evaluator
    fn ez_point_tape(
        &self,
    ) -> ShapeTape<<F::PointEval as TracingEvaluator>::Tape>;

    /// Returns an evaluation tape for an interval evaluator
    fn ez_interval_tape(
        &self,
    ) -> ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape>;

    /// Returns an evaluation tape for a float slice evaluator
    fn ez_float_slice_tape(
        &self,
    ) -> ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape>;

    /// Returns an evaluation tape for a float slice evaluator
    fn ez_grad_slice_tape(
        &self,
    ) -> ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape>;

    /// Computes a simplified tape using the given trace
    fn ez_simplify(&self, trace: &F::Trace) -> Result<Self, Error>
    where
        Self: Sized;
}

impl<F: Function> EzShape<F> for Shape<F> {
    fn ez_point_tape(
        &self,
    ) -> ShapeTape<<F::PointEval as TracingEvaluator>::Tape> {
        self.point_tape(Default::default())
    }

    fn ez_interval_tape(
        &self,
    ) -> ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape> {
        self.interval_tape(Default::default())
    }

    fn ez_float_slice_tape(
        &self,
    ) -> ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape> {
        self.float_slice_tape(Default::default())
    }

    fn ez_grad_slice_tape(
        &self,
    ) -> ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape> {
        self.grad_slice_tape(Default::default())
    }

    fn ez_simplify(&self, trace: &F::Trace) -> Result<Self, Error> {
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

impl<F: MathFunction> Shape<F> {
    pub fn new_with_axes(
        ctx: &Context,
        node: Node,
        axes: [Node; 3],
    ) -> Result<Self, Error> {
        let (f, vs) = F::new(ctx, node)?;
        let x = ctx.var_name(axes[0])?.ok_or(Error::NotAVar)?;
        let y = ctx.var_name(axes[1])?.ok_or(Error::NotAVar)?;
        let z = ctx.var_name(axes[2])?.ok_or(Error::NotAVar)?;
        Ok(Self {
            f,
            axes: [x, y, z].map(|v| vs.get(v).cloned()),
            transform: None,
        })
    }

    /// Builds a new shape from the given node with default (X, Y, Z) axes
    pub fn new(ctx: &mut Context, node: Node) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let axes = ctx.axes();
        Self::new_with_axes(ctx, node, axes)
    }
}

impl<F: MathFunction> From<Tree> for Shape<F> {
    fn from(t: Tree) -> Self {
        let mut ctx = Context::new();
        let node = ctx.import(&t);
        Self::new(&mut ctx, node).unwrap()
    }
}

/// Wrapper struct to bind a generic tape to particular X, Y, Z axes
pub struct ShapeTape<T> {
    tape: T,

    /// Index of the X, Y, Z axes in the variables array
    axes: [Option<usize>; 3],

    /// Optional transform
    transform: Option<Matrix4<f32>>,
}

impl<T: Tape> ShapeTape<T> {
    /// Recycles the inner tape's storage for reuse
    pub fn recycle(self) -> T::Storage {
        self.tape.recycle()
    }
}

/// Wrapper struct to convert from [`TracingEvaluator`] to
/// [`shape::TracingEvaluator`](TracingEvaluator)
#[derive(Debug, Default)]
pub struct ShapeTracingEval<E> {
    eval: E,
}

impl<E: TracingEvaluator> ShapeTracingEval<E>
where
    <E as TracingEvaluator>::Data: Transformable,
{
    pub fn eval<F: Into<E::Data>>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
    ) -> Result<(E::Data, Option<&E::Trace>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let (x, y, z) = if let Some(mat) = tape.transform {
            Transformable::transform(x, y, z, mat)
        } else {
            (x, y, z)
        };

        let mut vars = [None, None, None];
        if let Some(a) = tape.axes[0] {
            vars[a] = Some(x.into());
        }
        if let Some(b) = tape.axes[1] {
            vars[b] = Some(y.into());
        }
        if let Some(c) = tape.axes[2] {
            vars[c] = Some(z.into());
        }
        let n = vars.iter().position(Option::is_none).unwrap_or(3);
        let vars = vars.map(|v| v.unwrap_or(0f32.into()));
        self.eval.eval(&tape.tape, &vars[..n])
    }

    #[cfg(test)]
    pub fn eval_x<J: Into<E::Data>>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: J,
    ) -> E::Data {
        self.eval(tape, x.into(), E::Data::from(0.0), E::Data::from(0.0))
            .unwrap()
            .0
    }
    #[cfg(test)]
    pub fn eval_xy<J: Into<E::Data>>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: J,
        y: J,
    ) -> E::Data {
        self.eval(tape, x.into(), y.into(), E::Data::from(0.0))
            .unwrap()
            .0
    }
}

/// Bulk evaluator for a shape
///
/// This wraps a generic [`BulkEvaluator`] and exposes an API that takes
/// `(x, y, z)` arguments instead.  In addition, it applies the transform
/// associated with the [`ShapeTape`].
#[derive(Debug, Default)]
pub struct ShapeBulkEval<E: BulkEvaluator> {
    eval: E,
    xs: Vec<E::Data>,
    ys: Vec<E::Data>,
    zs: Vec<E::Data>,
}

impl<E: BulkEvaluator> ShapeBulkEval<E>
where
    E::Data: From<f32> + Transformable,
{
    pub fn eval(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
    ) -> Result<&[E::Data], Error> {
        let (xs, ys, zs) = if let Some(mat) = tape.transform {
            if x.len() != y.len() || x.len() != z.len() {
                return Err(Error::MismatchedSlices);
            }
            let n = x.len();
            self.xs.resize(n, 0.0.into());
            self.ys.resize(n, 0.0.into());
            self.zs.resize(n, 0.0.into());
            for i in 0..n {
                let (x, y, z) = Transformable::transform(x[i], y[i], z[i], mat);
                self.xs[i] = x;
                self.ys[i] = y;
                self.zs[i] = z;
            }
            (self.xs.as_slice(), self.ys.as_slice(), self.zs.as_slice())
        } else {
            (x, y, z)
        };
        let mut vars = [None, None, None];
        if let Some(a) = tape.axes[0] {
            vars[a] = Some(xs);
        }
        if let Some(b) = tape.axes[1] {
            vars[b] = Some(ys);
        }
        if let Some(c) = tape.axes[2] {
            vars[c] = Some(zs);
        }
        let n = vars.iter().position(|v| v.is_none()).unwrap_or(3);
        let vars = if vars.iter().all(Option::is_some) {
            vars.map(Option::unwrap)
        } else if let Some(q) = vars.iter().find(|v| v.is_some()) {
            vars.map(|v| v.unwrap_or_else(|| q.unwrap()))
        } else {
            [[].as_slice(); 3]
        };

        self.eval.eval(&tape.tape, &vars[..n])
    }
}

pub trait Transformable {
    fn transform(
        x: Self,
        y: Self,
        z: Self,
        mat: Matrix4<f32>,
    ) -> (Self, Self, Self)
    where
        Self: Sized;
}

impl Transformable for f32 {
    fn transform(x: f32, y: f32, z: f32, mat: Matrix4<f32>) -> (f32, f32, f32) {
        let out = mat.transform_point(&Point3::new(x, y, z));
        (out.x, out.y, out.z)
    }
}

impl Transformable for Interval {
    fn transform(
        x: Interval,
        y: Interval,
        z: Interval,
        mat: Matrix4<f32>,
    ) -> (Interval, Interval, Interval) {
        let out = [0, 1, 2, 3].map(|i| {
            let row = mat.row(i);
            x * row[0] + y * row[1] + z * row[2] + Interval::from(row[3])
        });

        (out[0] / out[3], out[1] / out[3], out[2] / out[3])
    }
}

impl Transformable for Grad {
    fn transform(
        x: Grad,
        y: Grad,
        z: Grad,
        mat: Matrix4<f32>,
    ) -> (Grad, Grad, Grad) {
        let out = [0, 1, 2, 3].map(|i| {
            let row = mat.row(i);
            x * row[0] + y * row[1] + z * row[2] + Grad::from(row[3])
        });

        (out[0] / out[3], out[1] / out[3], out[2] / out[3])
    }
}
