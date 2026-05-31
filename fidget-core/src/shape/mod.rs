//! Data structures for shape evaluation
//!
//! Types in this module are typically thin (generic) wrappers around objects
//! that implement traits in [`fidget_core::eval`](crate::eval).  The wrapper types
//! are specialized to operate on `x, y, z` arguments, rather than taking
//! arbitrary numbers of variables.
//!
//! For example, a [`Shape`] is a wrapper which makes it easier to treat a
//! [`Function`] as an implicit surface (with X, Y, Z axes and functions that
//! accept a transform matrix).
//!
//! ```rust
//! use fidget_core::vm::VmShape;
//! use fidget_core::context::Context;
//! use fidget_core::shape::EzShape;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let shape = VmShape::new(&ctx, x)?;
//!
//! // Let's build a single point evaluator:
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! let (value, _trace) = eval.eval_with_transform(
//!     &tape, 0.25, 0.0, 0.0, &nalgebra::Matrix4::identity()
//! )?;
//! assert_eq!(value, 0.25);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    context::{BadNode, Context, Node, Tree},
    eval::{
        BulkEvalError, BulkEvaluator, Function, MathFunction, Tape,
        TracingEvalError, TracingEvaluator,
    },
    types::{Grad, Interval},
    var::{BulkArgError, TracingArgError, Var, VarIndex, VarMap},
    vm::BadTrace,
};
use nalgebra::{Matrix4, Point3};
use std::collections::HashMap;

/// A shape represents an implicit surface with X/Y/Z variables
///
/// It is mostly agnostic to _how_ that surface is represented, wrapping a
/// [`Function`] and a set of axes.
///
/// Shapes are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
pub struct Shape<F> {
    /// Wrapped function
    f: F,
}

impl<F: Clone> Clone for Shape<F> {
    fn clone(&self) -> Self {
        Self { f: self.f.clone() }
    }
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
    #[inline]
    pub fn point_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::PointEval as TracingEvaluator>::Tape> {
        let tape = self.f.point_tape(storage);
        ShapeTape { tape }
    }

    /// Returns an evaluation tape for a interval evaluator
    #[inline]
    pub fn interval_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape> {
        let tape = self.f.interval_tape(storage);
        ShapeTape { tape }
    }

    /// Returns an evaluation tape for a float slice evaluator
    #[inline]
    pub fn float_slice_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::FloatSliceEval as BulkEvaluator>::Tape> {
        let tape = self.f.float_slice_tape(storage);
        ShapeTape { tape }
    }

    /// Returns an evaluation tape for a gradient slice evaluator
    #[inline]
    pub fn grad_slice_tape(
        &self,
        storage: F::TapeStorage,
    ) -> ShapeTape<<F::GradSliceEval as BulkEvaluator>::Tape> {
        let tape = self.f.grad_slice_tape(storage);
        ShapeTape { tape }
    }

    /// Computes a simplified tape using the given trace, and reusing storage
    #[inline]
    pub fn simplify(
        &self,
        trace: &F::Trace,
        storage: F::Storage,
        workspace: &mut F::Workspace,
    ) -> Result<Self, BadTrace>
    where
        Self: Sized,
    {
        let f = self.f.simplify(trace, storage, workspace)?;
        Ok(Self { f })
    }

    /// Attempt to reclaim storage from this shape
    ///
    /// This may fail, because shapes are `Clone` and are often implemented
    /// using an `Arc` around a heavier data structure.
    #[inline]
    pub fn recycle(self) -> Option<F::Storage> {
        self.f.recycle()
    }

    /// Returns a size associated with this shape
    ///
    /// This is underspecified and only used for unit testing; for tape-based
    /// shapes, it's typically the length of the tape,
    #[inline]
    pub fn size(&self) -> usize {
        self.f.size()
    }
}

impl<F> Shape<F> {
    /// Borrows the inner [`Function`] object
    pub fn inner(&self) -> &F {
        &self.f
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
        Self(HashMap::default())
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

    /// Iterates over values
    pub fn values(&self) -> impl Iterator<Item = &F> {
        self.0.values()
    }

    /// Looks up a value by [`VarIndex`]
    pub fn get(&self, i: VarIndex) -> Option<&F> {
        self.0.get(&i)
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
    fn ez_simplify(&self, trace: &F::Trace) -> Result<Self, BadTrace>
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

    fn ez_simplify(&self, trace: &F::Trace) -> Result<Self, BadTrace> {
        let mut workspace = Default::default();
        self.simplify(trace, Default::default(), &mut workspace)
    }
}

impl<F: MathFunction> Shape<F> {
    /// Builds a new shape from a math expression with default (X, Y, Z) axes
    pub fn new(ctx: &Context, node: Node) -> Result<Self, BadNode> {
        let f = F::new(ctx, &[node])?;
        Ok(Self { f })
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

/// Wrapper around a single-output function tape
#[derive(Clone)]
pub struct ShapeTape<T> {
    tape: T,
}

impl<T: Tape> ShapeTape<T> {
    /// Recycles the inner tape's storage for reuse
    pub fn recycle(self) -> Option<T::Storage> {
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
/// tape's X, Y, Z axes and its evaluators take a transform matrix.
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

/// A [`VarIndex`] variable is missing
#[derive(thiserror::Error, Debug, PartialEq)]
#[error("variable {var:?} must be provided")]
pub struct MissingVar {
    /// Missing variable index
    pub var: VarIndex,
}

/// Error type for shape tracing evaluation
#[derive(thiserror::Error, Debug)]
pub enum ShapeTracingEvalError {
    /// Missing variable index
    #[error(transparent)]
    MissingVar(#[from] MissingVar),
}

/// Error type for shape bulk evaluation
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ShapeBulkEvalError {
    /// Missing variable index
    #[error(transparent)]
    MissingVar(#[from] MissingVar),

    /// Mismatched slice length
    #[error(
        "slice lengths are mismatched: \
        slice for {var_a} has {length_a} items; \
        slice for {var_b} has {length_b} items;"
    )]
    MismatchedVarSlices {
        /// A specific variable
        var_a: Var,
        /// The length of the slice for `var_a`
        length_a: usize,
        /// A second variable
        var_b: Var,
        /// The length of the slice for `var_b`
        length_b: usize,
    },
}

impl<E: TracingEvaluator> ShapeTracingEval<E>
where
    <E as TracingEvaluator>::Data: Transformable,
{
    /// Tracing evaluation of the given tape with X, Y, Z input arguments
    #[inline]
    pub fn eval<F: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
    ) -> Result<(E::Data, Option<&E::Trace>), ShapeTracingEvalError> {
        let h = ShapeVars::<f32>::new();
        self.eval_raw(tape, x, y, z, None, &h)
    }

    /// Tracing evaluation with X, Y, Z arguments and a transform
    #[inline]
    pub fn eval_with_transform<F: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
        transform: &Matrix4<f32>,
    ) -> Result<(E::Data, Option<&E::Trace>), ShapeTracingEvalError> {
        let h = ShapeVars::<f32>::new();
        self.eval_raw(tape, x, y, z, Some(transform), &h)
    }

    /// Tracing evaluation with X, Y, Z arguments, transform, and vars
    #[inline]
    pub fn eval_with_transform_and_vars<
        F: Into<E::Data> + Copy,
        V: Into<E::Data> + Copy,
    >(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
        transform: &Matrix4<f32>,
        vars: &ShapeVars<V>,
    ) -> Result<(E::Data, Option<&E::Trace>), ShapeTracingEvalError> {
        self.eval_raw(tape, x, y, z, Some(transform), vars)
    }

    /// Tracing evaluation with X, Y, Z arguments and vars
    #[inline]
    pub fn eval_with_vars<F: Into<E::Data> + Copy, V: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
        vars: &ShapeVars<V>,
    ) -> Result<(E::Data, Option<&E::Trace>), ShapeTracingEvalError> {
        self.eval_raw(tape, x, y, z, None, vars)
    }

    /// Innermost evaluation function
    #[inline]
    pub fn eval_raw<F: Into<E::Data> + Copy, V: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: F,
        y: F,
        z: F,
        transform: Option<&Matrix4<f32>>,
        vars: &ShapeVars<V>,
    ) -> Result<(E::Data, Option<&E::Trace>), ShapeTracingEvalError> {
        assert_eq!(
            tape.tape.output_count(),
            1,
            "ShapeTape has multiple outputs"
        );

        let x = x.into();
        let y = y.into();
        let z = z.into();
        let (x, y, z) = if let Some(t) = transform {
            Transformable::transform(x, y, z, t)
        } else {
            (x, y, z)
        };

        let vs = tape.vars();
        self.scratch.resize(vs.len(), 0f32.into());
        for (var, index) in vs.iter() {
            match var {
                Var::X => self.scratch[index] = x,
                Var::Y => self.scratch[index] = y,
                Var::Z => self.scratch[index] = z,
                Var::V(i) => {
                    let Some(value) = vars.get(i) else {
                        return Err(MissingVar { var: i }.into());
                    };
                    self.scratch[index] = (*value).into();
                }
            }
        }
        let (out, trace) = match self.eval.eval(&tape.tape, &self.scratch) {
            Ok((out, trace)) => (out, trace),
            Err(TracingEvalError(TracingArgError::BadVarSlice(..))) => {
                unreachable!() // we resized `scratch` above
            }
        };
        Ok((out[0], trace))
    }
}

/// Wrapper around a [`BulkEvaluator`]
///
/// Unlike the raw bulk evaluator, a [`ShapeBulkEval`] knows about the
/// tape's X, Y, Z axes and accepts a transform matrix.
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
    /// [`eval_with_vars`](Self::eval_with_vars) or
    /// [`eval_with_var_arrays`](Self::eval_with_var_arrays) should be used
    /// instead (and this function will return an error).
    #[inline]
    pub fn eval(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
    ) -> Result<&[E::Data], ShapeBulkEvalError> {
        self.eval_raw(tape, x, y, z, None, Self::no_vars)
    }

    /// Bulk evaluation of many samples with a transform
    #[inline]
    pub fn eval_with_transform(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        transform: &Matrix4<f32>,
    ) -> Result<&[E::Data], ShapeBulkEvalError> {
        self.eval_raw(tape, x, y, z, Some(transform), Self::no_vars)
    }

    /// Bulk evaluation of many samples, with slices of variables
    ///
    /// Each variable is a slice (or `Vec`) of values, which must be the same
    /// length as the `x`, `y`, `z` slices.  This is in contrast with
    /// [`eval_with_vars`](Self::eval_with_vars), where variables have a single
    /// value used for every position in the `x`, `y,` `z` slices.
    ///
    /// Before evaluation, the transform matrix is applied to input coordinates.
    #[inline]
    pub fn eval_with_var_arrays<
        V: std::ops::Deref<Target = [G]>,
        G: Into<E::Data> + Copy,
    >(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        vars: &ShapeVars<V>,
    ) -> Result<&[E::Data], ShapeBulkEvalError> {
        self.eval_raw(tape, x, y, z, None, Self::var_array(vars))
    }

    /// Bulk evaluation of many transformed samples, with slices of variables
    #[inline]
    pub fn eval_with_transform_and_var_arrays<
        V: std::ops::Deref<Target = [G]>,
        G: Into<E::Data> + Copy,
    >(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        transform: &Matrix4<f32>,
        vars: &ShapeVars<V>,
    ) -> Result<&[E::Data], ShapeBulkEvalError> {
        self.eval_raw(tape, x, y, z, Some(transform), Self::var_array(vars))
    }

    /// Bulk evaluation of many samples, with fixed variables
    ///
    /// Each variable has a single value, which is used for every position in
    /// the `x`, `y`, `z` slices.  This is in contrast with
    /// [`eval_with_var_arrays`](Self::eval_with_var_arrays), where variables
    /// can be different for every position in the `x`, `y,` `z` slices.
    #[inline]
    pub fn eval_with_vars<G: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        vars: &ShapeVars<G>,
    ) -> Result<&[E::Data], ShapeBulkEvalError> {
        self.eval_raw(tape, x, y, z, None, Self::var_value(vars))
    }

    /// Bulk evaluation of many transformed samples, with fixed variables
    #[inline]
    pub fn eval_with_transform_and_vars<G: Into<E::Data> + Copy>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        transform: &Matrix4<f32>,
        vars: &ShapeVars<G>,
    ) -> Result<&[E::Data], ShapeBulkEvalError> {
        self.eval_raw(tape, x, y, z, Some(transform), Self::var_value(vars))
    }

    /// Helper function for evaluation without variables
    #[inline]
    fn no_vars(
        _: &mut [E::Data],
        var: VarIndex,
    ) -> Result<(), ShapeBulkEvalError> {
        Err(MissingVar { var }.into())
    }

    /// Helper to bind to a multi-value variable map
    ///
    /// This is a building block for working with [`eval_raw`](Self::eval_raw),
    /// and you probably don't need it unless you're deep in the weeds.
    #[inline]
    pub fn var_array<
        V: std::ops::Deref<Target = [G]>,
        G: Into<E::Data> + Copy,
    >(
        vars: &ShapeVars<V>,
    ) -> impl Fn(&mut [E::Data], VarIndex) -> Result<(), ShapeBulkEvalError>
    {
        |data: &mut [E::Data], i: VarIndex| {
            let vars = vars.get(i).ok_or(MissingVar { var: i })?;
            if vars.len() != data.len() {
                return Err(ShapeBulkEvalError::MismatchedVarSlices {
                    var_a: Var::X,
                    length_a: data.len(),
                    var_b: Var::V(i),
                    length_b: vars.len(),
                });
            }
            for (a, b) in data.iter_mut().zip(vars.deref().iter()) {
                *a = (*b).into();
            }
            Ok(())
        }
    }

    /// Helper to bind to a single-variable map
    ///
    /// This is a building block for working with [`eval_raw`](Self::eval_raw),
    /// and you probably don't need it unless you're deep in the weeds.
    #[inline]
    pub fn var_value<G: Into<E::Data> + Copy>(
        vars: &ShapeVars<G>,
    ) -> impl Fn(&mut [E::Data], VarIndex) -> Result<(), ShapeBulkEvalError>
    {
        |data: &mut [E::Data], i: VarIndex| {
            let value = vars.get(i).ok_or(MissingVar { var: i })?;
            data.fill((*value).into());
            Ok(())
        }
    }

    /// Core implementation of evaluation
    ///
    /// `copy_vars` is a way to populate variable slices (other than the X, Y, Z
    /// inputs), and may be generated from a helper like
    /// [`var_value`](Self::var_value) or [`var_array`](Self::var_array).
    #[inline]
    pub fn eval_raw<F>(
        &mut self,
        tape: &ShapeTape<E::Tape>,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
        transform: Option<&Matrix4<f32>>,
        copy_vars: F,
    ) -> Result<&[E::Data], ShapeBulkEvalError>
    where
        F: Fn(&mut [E::Data], VarIndex) -> Result<(), ShapeBulkEvalError>,
    {
        assert_eq!(
            tape.tape.output_count(),
            1,
            "ShapeTape has multiple outputs" // enforced by ShapeTape
        );

        // Make sure our scratch arrays are big enough for this evaluation
        if x.len() != y.len() {
            return Err(ShapeBulkEvalError::MismatchedVarSlices {
                var_a: Var::X,
                length_a: x.len(),
                var_b: Var::Y,
                length_b: y.len(),
            });
        }
        if x.len() != z.len() {
            return Err(ShapeBulkEvalError::MismatchedVarSlices {
                var_a: Var::X,
                length_a: x.len(),
                var_b: Var::Z,
                length_b: z.len(),
            });
        }
        let n = x.len();

        // We need at least one item in the scratch array to set evaluation
        // size; otherwise, evaluating a single constant will return []
        let vs = tape.vars();
        self.scratch.resize_with(vs.len().max(1), Vec::new);
        for s in &mut self.scratch {
            s.resize(n, 0.0.into());
        }

        let mut axes = [None; 3];
        for (var, index) in vs.iter() {
            match var {
                Var::X => axes[0] = Some(index),
                Var::Y => axes[1] = Some(index),
                Var::Z => axes[2] = Some(index),
                Var::V(i) => {
                    copy_vars(&mut self.scratch[index], i)?;
                }
            }
        }

        for i in 0..n {
            let (x, y, z) = if let Some(t) = transform {
                Transformable::transform(x[i], y[i], z[i], t)
            } else {
                (x[i], y[i], z[i])
            };
            if let Some(a) = axes[0] {
                self.scratch[a][i] = x;
            }
            if let Some(b) = axes[1] {
                self.scratch[b][i] = y;
            }
            if let Some(c) = axes[2] {
                self.scratch[c][i] = z;
            }
        }

        let out = match self.eval.eval(&tape.tape, &self.scratch) {
            Ok(out) => out,
            Err(BulkEvalError(e)) => match e {
                // All of these conditions should be handled by `setup`
                BulkArgError::BadVarSlice(..)
                | BulkArgError::MismatchedSlices(..) => unreachable!(),
            },
        };
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
        mat: &Matrix4<f32>,
    ) -> (Self, Self, Self)
    where
        Self: Sized;
}

impl Transformable for f32 {
    fn transform(
        x: f32,
        y: f32,
        z: f32,
        mat: &Matrix4<f32>,
    ) -> (f32, f32, f32) {
        let out = mat.transform_point(&Point3::new(x, y, z));
        (out.x, out.y, out.z)
    }
}

impl Transformable for Interval {
    fn transform(
        x: Interval,
        y: Interval,
        z: Interval,
        mat: &Matrix4<f32>,
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
        mat: &Matrix4<f32>,
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

    #[test]
    fn shape_eval_bulk_size() {
        let s = Tree::constant(1.0);
        let mut ctx = Context::new();
        let s = ctx.import(&s);

        let s = VmShape::new(&ctx, s).unwrap();
        let tape = s.ez_float_slice_tape();
        let mut eval = VmShape::new_float_slice_eval();
        let out = eval
            .eval(&tape, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
            .unwrap();
        assert_eq!(out, [1.0, 1.0, 1.0]);
    }
}
