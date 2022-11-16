//! Single-point evaluation
use crate::eval::{tape::Tape, Choice, Eval};

/// Function handle for `f32` evaluation
pub trait PointEvalT<R> {
    fn new(tape: &Tape<R>) -> Self;
    fn eval_p(&mut self, x: f32, y: f32, z: f32, c: &mut [Choice]) -> f32;
}

/// Function handle for point evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `PointEval` objects, which actually do evaluation.
pub struct PointEval<E: Eval> {
    tape: Tape<E>,
    choices: Vec<Choice>,
    eval: E::PointEval,
}

impl<E: Eval> PointEval<E> {
    pub fn new(tape: Tape<E>) -> Self {
        let eval = E::PointEval::new(&tape);
        let choices = vec![Choice::Unknown; tape.choice_count()];
        Self {
            tape,
            choices,
            eval,
        }
    }
    /// Calculates a simplified [`Tape`](crate::tape::Tape) based on the last
    /// evaluation.
    pub fn simplify(&self) -> Tape<E> {
        self.tape.simplify(&self.choices).unwrap()
    }

    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }

    /// Resets the internal choice array to `Choice::Unknown`
    fn reset_choices(&mut self) {
        self.choices.fill(Choice::Unknown);
    }

    /// Performs point evaluation
    pub fn eval_p(&mut self, x: f32, y: f32, z: f32) -> f32 {
        self.reset_choices();
        let out = self.eval.eval_p(x, y, z, self.choices.as_mut_slice());
        out
    }
}

////////////////////////////////////////////////////////////////////////////////

// This module exports a standard test suite for any point evaluator, which can
// be included as `point_tests!(ty)`.
#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::context::Context;

    pub fn test_constant<I: Eval>() {
        let mut ctx = Context::new();
        let p = ctx.constant(1.5);
        let tape = ctx.get_tape(p);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(0.0, 0.0, 0.0), 1.5);
    }

    pub fn test_constant_push<I: Eval>() {
        let mut ctx = Context::new();
        let a = ctx.constant(1.5);
        let x = ctx.x();
        let min = ctx.min(a, x).unwrap();
        let tape = ctx.get_tape(min);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(2.0, 0.0, 0.0), 1.5);

        let next = eval.simplify();
        assert_eq!(next.len(), 1);
        let mut eval = I::new_point_evaluator(next);
        assert_eq!(eval.eval_p(2.0, 0.0, 0.0), 1.5);
        assert_eq!(eval.eval_p(1.0, 0.0, 0.0), 1.5);
    }

    pub fn test_circle<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let circle = ctx.sub(radius, 1.0).unwrap();

        let tape = ctx.get_tape(circle);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(0.0, 0.0, 0.0), -1.0);
        assert_eq!(eval.eval_p(1.0, 0.0, 0.0), 0.0);
    }

    pub fn test_p_min<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(0.0, 0.0, 0.0), 0.0);
        assert_eq!(eval.choices(), &[Choice::Both]);

        assert_eq!(eval.eval_p(0.0, 1.0, 0.0), 0.0);
        assert_eq!(eval.choices(), &[Choice::Left]);

        assert_eq!(eval.eval_p(2.0, 0.0, 0.0), 0.0);
        assert_eq!(eval.choices(), &[Choice::Right]);

        let v = eval.eval_p(std::f32::NAN, 0.0, 0.0);
        assert!(v.is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);

        let v = eval.eval_p(0.0, std::f32::NAN, 0.0);
        assert!(v.is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);
    }

    pub fn test_p_max<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = ctx.get_tape(max);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(0.0, 0.0, 0.0), 0.0);
        assert_eq!(eval.choices(), &[Choice::Both]);

        assert_eq!(eval.eval_p(0.0, 1.0, 0.0), 1.0);
        assert_eq!(eval.choices(), &[Choice::Right]);

        assert_eq!(eval.eval_p(2.0, 0.0, 0.0), 2.0);
        assert_eq!(eval.choices(), &[Choice::Left]);

        let v = eval.eval_p(std::f32::NAN, 0.0, 0.0);
        assert!(v.is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);

        let v = eval.eval_p(0.0, std::f32::NAN, 0.0);
        assert!(v.is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);
    }

    pub fn basic_interpreter<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sum = ctx.add(x, 1.0).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let tape = ctx.get_tape(min);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(1.0, 3.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(3.0, 3.5, 0.0), 3.5);
    }

    pub fn test_push<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min);
        let mut eval = I::new_point_evaluator(tape.clone());
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]).unwrap();
        let mut eval = I::new_point_evaluator(t);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]).unwrap();
        let mut eval = I::new_point_evaluator(t);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 2.0);

        let min = ctx.min(x, 1.0).unwrap();
        let tape = ctx.get_tape(min);
        let mut eval = I::new_point_evaluator(tape.clone());
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]).unwrap();
        let mut eval = I::new_point_evaluator(t);
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]).unwrap();
        let mut eval = I::new_point_evaluator(t);
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 1.0);
    }

    pub fn test_basic<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let y2 = ctx.mul(y, 2.5).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let tape = ctx.get_tape(sum);
        let mut eval = I::new_point_evaluator(tape);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 6.0);
    }

    #[macro_export]
    macro_rules! point_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::point::eval_tests::$i::<$t>()
            }
        };
    }

    #[macro_export]
    macro_rules! point_tests {
        ($t:ty) => {
            $crate::point_test!(test_constant, $t);
            $crate::point_test!(test_constant_push, $t);
            $crate::point_test!(test_circle, $t);
            $crate::point_test!(test_p_max, $t);
            $crate::point_test!(test_p_min, $t);
            $crate::point_test!(basic_interpreter, $t);
            $crate::point_test!(test_push, $t);
            $crate::point_test!(test_basic, $t);
        };
    }
}
