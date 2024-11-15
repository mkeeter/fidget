//! Data structures for shape evaluation
//!
//! Types in this module are typically thin (generic) wrappers around objects
//! that implement traits in [`fidget::eval`](crate::eval).  The wrapper types
//! are specialized to operate on `x, y, z` arguments, rather than taking
//! arbitrary numbers of variables.
//!
//! For example, a [`Shape`] is a wrapper which makes it easier to treat a
//! [`Function`] as an implicit surface (with X, Y, Z axes and an optional
//! transform matrix).
//!
//! ```rust
//! use fidget::vm::VmShape;
//! use fidget::context::Context;
//! use fidget::shape::EzShape;
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

use crate::{
    context::{Context, Node, Tree},
    eval::{BulkEvaluator, Function, MathFunction, Tape, TracingEvaluator},
    types::{Grad, Interval},
    var::{Var, VarIndex, VarMap},
    Error,
};
use nalgebra::{Matrix4, Point3};
use std::collections::HashMap;

/// A shape represents an implicit surface
///
/// It is mostly agnostic to _how_ that surface is represented, wrapping a
/// [`Function`] and a set of axes.
///
/// Shapes are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
#[derive(Clone)]
pub struct Shape<F> {
    /// Wrapped function
    f: F,

    /// Variables representing x, y, z axes
    axes: [Var; 3],

    /// Optional transform to apply to the shape
    transform: Option<Matrix4<f32>>,
}

impl<F: Function + Clone> Shape<F> {
    /// Builds a new point evaluator
    pub fn new_point_eval() -> ShapeTracingEval<F::PointEval> {
        ShapeTracingEval {
            eval: F::PointEval::default(),
            scratch: vec![],
        }
    }

    /// Builds a new interval evaluator
    pub fn new_interval_eval() -> ShapeTracingEval<F::IntervalEval> {
        ShapeTracingEval {
            eval: F::IntervalEval::default(),
            scratch: vec![],
        }
    }

    /// Builds a new float slice evaluator
    pub fn new_float_slice_eval() -> ShapeBulkEval<F::FloatSliceEval> {
        ShapeBulkEval {
            eval: F::FloatSliceEval::default(),
            scratch: vec![],
        }
    }

    /// Builds a new gradient slice evaluator
    pub fn new_grad_slice_eval() -> ShapeBulkEval<F::GradSliceEval> {
        ShapeBulkEval {
            eval: F::GradSliceEval::default(),
            scratch: vec![],
        }
    }

    /// Returns an evaluation tape for a point evaluator
    pub fn point_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::PointEval as TracingEvaluator>::Tape> {
        let tape = self.f.point_tape(storage);
        let vars = tape.vars();
        let axes = self.axes.map(|v| vars.get(&v));
        ShapeTape {
            tape,
            axes,
            transform: self.transform,
        }
    }

    /// Returns an evaluation tape for a interval evaluator
    pub fn interval_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape> {
        let tape = self.f.interval_tape(storage);
        let vars = tape.vars();
        let axes = self.axes.map(|v| vars.get(&v));
        ShapeTape {
            tape,
            axes,
            transform: self.transform,
        }
    }

    /// Returns an evaluation tape for a float slice evaluator
    pub fn float_slice_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape> {
        let tape = self.f.float_slice_tape(storage);
        let vars = tape.vars();
        let axes = self.axes.map(|v| vars.get(&v));
        ShapeTape {
            tape,
            axes,
            transform: self.transform,
        }
    }

    /// Returns an evaluation tape for a gradient slice evaluator
    pub fn grad_slice_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape> {
        let tape = self.f.grad_slice_tape(storage);
        let vars = tape.vars();
        let axes = self.axes.map(|v| vars.get(&v));
        ShapeTape {
            tape,
            axes,
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
}

impl<F> Shape<F> {
    /// Borrows the inner [`Function`] object
    pub fn inner(&self) -> &F {
        &self.f
    }

    /// Borrows the inner axis mapping
    pub fn axes(&self) -> &[Var; 3] {
        &self.axes
    }

    /// Raw constructor
    pub fn new_raw(f: F, axes: [Var; 3]) -> Self {
        Self {
            f,
            axes,
            transform: None,
        }
    }
    /// Returns a shape with the given transform applied
    pub fn apply_transform(mut self, mat: Matrix4<f32>) -> Self {
        if let Some(prev) = self.transform.as_mut() {
            *prev *= mat;
        } else {
            self.transform = Some(mat);
        }
        self
    }
}

/// Variables bound to values for shape evaluation
///
/// Note that this cannot store `X`, `Y`, `Z` variables (which are passed in as
/// first-class arguments); it only stores [`Var::V`] values (identified by
/// their inner [`VarIndex`]).
pub struct ShapeVars<F>(HashMap<VarIndex, F>);

impl<F> Default for ShapeVars<F> {
    fn default() -> Self {
        Self(HashMap::default())
    }
}

impl<F> ShapeVars<F> {
    /// Builds a new, empty variable set
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the number of variables stored in the set
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Checks whether the variable set is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Inserts a new variable
    ///
    /// Returns the previous value (if present)
    pub fn insert(&mut self, v: VarIndex, f: F) -> Option<F> {
        self.0.insert(v, f)
    }
}

impl<'a, F> IntoIterator for &'a ShapeVars<F> {
    type Item = (&'a VarIndex, &'a F);
    type IntoIter = std::collections::hash_map::Iter<'a, VarIndex, F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
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

impl<F: MathFunction> Shape<F> {
    /// Builds a new shape from a math expression with the given axes
    pub fn new_with_axes(
        ctx: &Context,
        node: Node,
        axes: [Var; 3],
    ) -> Result<Self, Error> {
        let f = F::new(ctx, &[node])?;
        Ok(Self {
            f,
            axes,
            transform: None,
        })
    }

    /// Builds a new shape from the given node with default (X, Y, Z) axes
    pub fn new(ctx: &Context, node: Node) -> Result<Self, Error>
    where
        Self: Sized,
    {
        Self::new_with_axes(ctx, node, [Var::X, Var::Y, Var::Z])
    }
}

/// Converts a [`Tree`] to a [`Shape`] with the default axes
impl<F: MathFunction> From<Tree> for Shape<F> {
    fn from(t: Tree) -> Self {
        let mut ctx = Context::new();
        let node = ctx.import(&t);
        Self::new(&ctx, node).unwrap()
    }
}

/// Wrapper around a function tape, with axes and an optional transform matrix
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

    /// Returns a mapping from [`Var`] to evaluation index
    pub fn vars(&self) -> &VarMap {
        self.tape.vars()
    }
}

/// Wrapper around a [`TracingEvaluator`]
///
/// Unlike the raw tracing evaluator, a [`ShapeTracingEval`] knows about the
/// tape's X, Y, Z axes and optional transform matrix.
#[derive(Debug)]
pub struct ShapeTracingEval<E: TracingEvaluator> {
    eval: E,
    scratch: Vec<E::Data>,
}

impl<E: TracingEvaluator> Default for ShapeTracingEval<E> {
    fn default() -> Self {
        Self {
            eval: E::default(),
            scratch: vec![],
        }
    }
}

impl<E: TracingEvaluator> ShapeTracingEval<E>
where
    <E as TracingEvaluator>::Data: Transformable,
{
    /// Tracing evaluation of the given tape with X, Y, Z input arguments
    ///
    /// Before evaluation, the tape's transform matrix is applied (if present).
    ///
    /// If the tape has other variables, [`eval_v`](Self::eval_v) should be
    /// called instead (and this function will return an error.
    pub fn eval<F: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
    ) -> Result<(E::Data, Option<&E::Trace>), Error> {
        let h = ShapeVars::default();
        self.eval_v(tape, x, y, z, &h)
    }

    /// Tracing evaluation of a single sample
    ///
    /// Before evaluation, the tape's transform matrix is applied (if present).
    pub fn eval_v<F: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
        vars: &ShapeVars<F>,
    ) -> Result<(E::Data, Option<&E::Trace>), Error> {
        assert_eq!(
            tape.tape.output_count(),
            1,
            "ShapeTape has multiple outputs"
        );

        let x = x.into();
        let y = y.into();
        let z = z.into();
        let (x, y, z) = if let Some(mat) = tape.transform {
            Transformable::transform(x, y, z, mat)
        } else {
            (x, y, z)
        };

        let vs = tape.vars();
        let expected_vars = vs.len()
            - vs.get(&Var::X).is_some() as usize
            - vs.get(&Var::Y).is_some() as usize
            - vs.get(&Var::Z).is_some() as usize;
        if expected_vars != vars.len() {
            return Err(Error::BadVarSlice(vars.len(), expected_vars));
        }

        self.scratch.resize(tape.vars().len(), 0f32.into());
        if let Some(a) = tape.axes[0] {
            self.scratch[a] = x;
        }
        if let Some(b) = tape.axes[1] {
            self.scratch[b] = y;
        }
        if let Some(c) = tape.axes[2] {
            self.scratch[c] = z;
        }
        for (var, value) in vars {
            if let Some(i) = vs.get(&Var::V(*var)) {
                if i < self.scratch.len() {
                    self.scratch[i] = (*value).into();
                } else {
                    return Err(Error::BadVarIndex(i, self.scratch.len()));
                }
            } else {
                // Passing in Bonus Variables is allowed (for now)
            }
        }

        let (out, trace) = self.eval.eval(&tape.tape, &self.scratch)?;
        Ok((out[0], trace))
    }
}

/// Wrapper around a [`BulkEvaluator`]
///
/// Unlike the raw bulk evaluator, a [`ShapeBulkEval`] knows about the
/// tape's X, Y, Z axes and optional transform matrix.
#[derive(Debug, Default)]
pub struct ShapeBulkEval<E: BulkEvaluator> {
    eval: E,
    scratch: Vec<Vec<E::Data>>,
}

impl<E: BulkEvaluator> ShapeBulkEval<E>
where
    E::Data: From<f32> + Transformable,
{
    /// Bulk evaluation of many samples, without any variables
    ///
    /// If the shape includes variables other than `X`, `Y`, `Z`,
    /// [`eval_v`](Self::eval_v) should be used instead (and this function will
    /// return an error).
    ///
    /// Before evaluation, the tape's transform matrix is applied (if present).
    pub fn eval(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
    ) -> Result<&[E::Data], Error> {
        let h: HashMap<VarIndex, &[E::Data]> = HashMap::default();
        self.eval_v(tape, x, y, z, &h)
    }

    /// Bulk evaluation of many samples, with variables
    ///
    /// Before evaluation, the tape's transform matrix is applied (if present).
    pub fn eval_v<V: std::ops::Deref<Target = [E::Data]>>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        vars: &HashMap<VarIndex, V>,
    ) -> Result<&[E::Data], Error> {
        assert_eq!(
            tape.tape.output_count(),
            1,
            "ShapeTape has multiple outputs"
        );

        // Make sure our scratch arrays are big enough for this evaluation
        if x.len() != y.len() || x.len() != z.len() {
            return Err(Error::MismatchedSlices);
        }
        let n = x.len();
        if vars.values().any(|vs| vs.len() != n) {
            return Err(Error::MismatchedSlices);
        }
        let vs = tape.vars();
        let expected_vars = vs.len()
            - vs.get(&Var::X).is_some() as usize
            - vs.get(&Var::Y).is_some() as usize
            - vs.get(&Var::Z).is_some() as usize;
        if expected_vars != vars.len() {
            return Err(Error::BadVarSlice(vars.len(), expected_vars));
        }

        self.scratch.resize_with(vs.len(), Vec::new);
        for s in &mut self.scratch {
            s.resize(n, 0.0.into());
        }

        if let Some(mat) = tape.transform {
            self.scratch.resize_with(tape.vars().len(), Vec::new);
            for s in &mut self.scratch {
                s.resize(n, 0.0.into());
            }
            for i in 0..n {
                let (x, y, z) = Transformable::transform(x[i], y[i], z[i], mat);
                if let Some(a) = tape.axes[0] {
                    self.scratch[a][i] = x;
                }
                if let Some(b) = tape.axes[1] {
                    self.scratch[b][i] = y;
                }
                if let Some(c) = tape.axes[2] {
                    self.scratch[c][i] = z;
                }
            }
        } else {
            if let Some(a) = tape.axes[0] {
                self.scratch[a].copy_from_slice(x);
            }
            if let Some(b) = tape.axes[1] {
                self.scratch[b].copy_from_slice(y);
            }
            if let Some(c) = tape.axes[2] {
                self.scratch[c].copy_from_slice(z);
            }
            // TODO fast path if there are no extra vars, reusing slices
        };

        for (var, value) in vars {
            if let Some(i) = vs.get(&Var::V(*var)) {
                if i < self.scratch.len() {
                    self.scratch[i].copy_from_slice(value);
                } else {
                    return Err(Error::BadVarIndex(i, self.scratch.len()));
                }
            } else {
                // Passing in Bonus Variables is allowed (for now)
            }
        }

        let out = self.eval.eval(&tape.tape, &self.scratch)?;
        Ok(out.borrow(0))
    }
}

/// Trait for types that can be transformed by a 4x4 homogeneous transform matrix
pub trait Transformable {
    /// Apply the given transform to an `(x, y, z)` position
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::vm::VmShape;

    #[test]
    fn shape_vars() {
        let v = Var::new();
        let s = Tree::x() + Tree::y() + v;

        let mut ctx = Context::new();
        let s = ctx.import(&s);

        let s = VmShape::new(&ctx, s).unwrap();
        let vs = s.inner().vars();
        assert_eq!(vs.len(), 3);

        assert!(vs.get(&Var::X).is_some());
        assert!(vs.get(&Var::Y).is_some());
        assert!(vs.get(&Var::Z).is_none());
        assert!(vs.get(&v).is_some());

        let mut seen = [false; 3];
        for v in [Var::X, Var::Y, v] {
            seen[vs[&v]] = true;
        }
        assert!(seen.iter().all(|i| *i));
    }
}
