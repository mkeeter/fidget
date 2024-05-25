use crate::{
    eval::Tape,
    shape::{BulkEvaluator, RenderHints, Shape, TracingEvaluator},
    types::{Grad, Interval},
    Error,
};
use nalgebra::{Matrix4, Point3, Vector3};

/// A generic [`Shape`] that has been transformed by a 4x4 transform matrix
#[derive(Clone)]
pub struct TransformedShape<S> {
    shape: S,
    mat: Matrix4<f32>,
}

impl<S> TransformedShape<S> {
    /// Builds a new [`TransformedShape`] with the identity transform
    pub fn new(shape: S, mat: Matrix4<f32>) -> Self {
        Self { shape, mat }
    }

    /// Appends a translation to the transformation matrix
    pub fn translate(&mut self, offset: Vector3<f32>) {
        self.mat.append_translation_mut(&offset);
    }

    /// Appends a uniform scale to the transformation matrix
    pub fn scale(&mut self, scale: f32) {
        self.mat.append_scaling_mut(scale);
    }

    /// Resets to the identity transform matrix
    pub fn reset(&mut self) {
        self.mat = Matrix4::identity();
    }

    /// Sets the transform matrix
    pub fn set_transform(&mut self, mat: Matrix4<f32>) {
        self.mat = mat;
    }
}

/// A generic [`Tape`] with an associated 4x4 transform matrix
pub struct TransformedTape<T> {
    tape: T,
    mat: Matrix4<f32>,
}

impl<T: Tape> Tape for TransformedTape<T> {
    type Storage = <T as Tape>::Storage;
    fn recycle(self) -> Self::Storage {
        self.tape.recycle()
    }
}

/// A generic [`TracingEvaluator`] which applies a transform matrix
#[derive(Default)]
pub struct TransformedTracingEval<E> {
    eval: E,
}

trait Transformable {
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

impl<T: TracingEvaluator> TracingEvaluator for TransformedTracingEval<T>
where
    <T as TracingEvaluator>::Data: Transformable,
{
    type Data = <T as TracingEvaluator>::Data;
    type Tape = TransformedTape<<T as TracingEvaluator>::Tape>;
    type TapeStorage = <T as TracingEvaluator>::TapeStorage;
    type Trace = <T as TracingEvaluator>::Trace;
    fn eval<F: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
    ) -> Result<(Self::Data, Option<&Self::Trace>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let (x, y, z) = Transformable::transform(x, y, z, tape.mat);
        self.eval.eval(&tape.tape, x, y, z)
    }
}

/// A generic [`BulkEvaluator`] which applies a transform matrix
pub struct TransformedBulkEval<E: BulkEvaluator> {
    eval: E,
    xs: Vec<E::Data>,
    ys: Vec<E::Data>,
    zs: Vec<E::Data>,
}

impl<E: BulkEvaluator> Default for TransformedBulkEval<E> {
    fn default() -> Self {
        Self {
            eval: E::default(),
            xs: vec![],
            ys: vec![],
            zs: vec![],
        }
    }
}

impl<E: BulkEvaluator> BulkEvaluator for TransformedBulkEval<E>
where
    <E as BulkEvaluator>::Data: Transformable,
{
    type Data = <E as BulkEvaluator>::Data;
    type Tape = TransformedTape<<E as BulkEvaluator>::Tape>;
    type TapeStorage = <E as BulkEvaluator>::TapeStorage;
    fn eval(
        &mut self,
        tape: &Self::Tape,
        x: &[E::Data],
        y: &[E::Data],
        z: &[E::Data],
    ) -> Result<&[Self::Data], Error> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(Error::MismatchedSlices);
        }
        let n = x.len();
        self.xs.resize(n, E::Data::from(0.0));
        self.ys.resize(n, E::Data::from(0.0));
        self.zs.resize(n, E::Data::from(0.0));
        for i in 0..x.len() {
            let (x, y, z) =
                Transformable::transform(x[i], y[i], z[i], tape.mat);
            self.xs[i] = x;
            self.ys[i] = y;
            self.zs[i] = z;
        }
        self.eval.eval(&tape.tape, &self.xs, &self.ys, &self.zs)
    }
}

impl<S: Shape> Shape for TransformedShape<S> {
    type Trace = <S as Shape>::Trace;
    type Storage = <S as Shape>::Storage;
    type Workspace = <S as Shape>::Workspace;
    type TapeStorage = <S as Shape>::TapeStorage;
    type PointEval = TransformedTracingEval<<S as Shape>::PointEval>;
    type IntervalEval = TransformedTracingEval<<S as Shape>::IntervalEval>;
    type FloatSliceEval = TransformedBulkEval<<S as Shape>::FloatSliceEval>;
    type GradSliceEval = TransformedBulkEval<<S as Shape>::GradSliceEval>;
    fn size(&self) -> usize {
        self.shape.size()
    }
    fn recycle(self) -> Option<Self::Storage> {
        self.shape.recycle()
    }
    fn point_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> TransformedTape<<<S as Shape>::PointEval as TracingEvaluator>::Tape>
    {
        TransformedTape {
            tape: self.shape.point_tape(storage),
            mat: self.mat,
        }
    }
    fn interval_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> TransformedTape<<<S as Shape>::IntervalEval as TracingEvaluator>::Tape>
    {
        TransformedTape {
            tape: self.shape.interval_tape(storage),
            mat: self.mat,
        }
    }
    fn float_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> TransformedTape<<<S as Shape>::FloatSliceEval as BulkEvaluator>::Tape>
    {
        TransformedTape {
            tape: self.shape.float_slice_tape(storage),
            mat: self.mat,
        }
    }
    fn grad_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> TransformedTape<<<S as Shape>::GradSliceEval as BulkEvaluator>::Tape>
    {
        TransformedTape {
            tape: self.shape.grad_slice_tape(storage),
            mat: self.mat,
        }
    }
    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, Error> {
        let shape = self.shape.simplify(trace, storage, workspace)?;
        Ok(Self {
            shape,
            mat: self.mat,
        })
    }

    type TransformedShape = Self;
    fn apply_transform(mut self, mat: Matrix4<f32>) -> Self::TransformedShape {
        self.mat *= mat;
        self
    }
}

impl<S: RenderHints> RenderHints for TransformedShape<S> {
    fn tile_sizes_2d() -> &'static [usize] {
        S::tile_sizes_2d()
    }
    fn tile_sizes_3d() -> &'static [usize] {
        S::tile_sizes_3d()
    }
}
