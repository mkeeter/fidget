use crate::{
    eval::{Choice, EvalFamily},
    tape::Tape,
};

/// Function handle for `f32` evaluation
pub trait PointEvalT: From<Tape> {
    type Family: EvalFamily;
    fn eval_p(&mut self, x: f32, y: f32, z: f32, c: &mut [Choice]) -> f32;
}

/// Function handle for point evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `PointEval` objects, which actually do evaluation.
pub struct PointEval<E> {
    tape: Tape,
    choices: Vec<Choice>,
    eval: E,
}

impl<E: PointEvalT> From<Tape> for PointEval<E> {
    fn from(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            choices: vec![Choice::Unknown; tape.choice_count()],
            eval: E::from(tape),
        }
    }
}

impl<E: PointEvalT> PointEval<E> {
    /// Calculates a simplified [`Tape`](crate::tape::Tape) based on the last
    /// evaluation.
    pub fn simplify(&self, reg_limit: u8) -> Tape {
        self.tape.simplify_with_reg_limit(&self.choices, reg_limit)
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

    pub fn test_circle<I: PointEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let one = ctx.constant(1.0);
        let circle = ctx.sub(radius, one).unwrap();

        let tape = ctx.get_tape(circle, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape);
        assert_eq!(eval.eval_p(0.0, 0.0, 0.0), -1.0);
        assert_eq!(eval.eval_p(1.0, 0.0, 0.0), 0.0);
    }

    pub fn test_p_min<I: PointEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape);
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

    pub fn test_p_max<I: PointEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = ctx.get_tape(max, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape);
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

    pub fn basic_interpreter<I: PointEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let one = ctx.constant(1.0);
        let sum = ctx.add(x, one).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let tape = ctx.get_tape(min, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(1.0, 3.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(3.0, 3.5, 0.0), 3.5);
    }

    pub fn test_push<I: PointEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape.clone());
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = PointEval::<I>::from(t);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = PointEval::<I>::from(t);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let tape = ctx.get_tape(min, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape.clone());
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = PointEval::<I>::from(t);
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = PointEval::<I>::from(t);
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 1.0);
    }

    pub fn test_basic<I: PointEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let two = ctx.constant(2.5);
        let y2 = ctx.mul(y, two).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let tape = ctx.get_tape(sum, I::Family::REG_LIMIT);
        let mut eval = PointEval::<I>::from(tape);
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
            $crate::point_test!(test_circle, $t);
            $crate::point_test!(test_p_max, $t);
            $crate::point_test!(test_p_min, $t);
            $crate::point_test!(basic_interpreter, $t);
            $crate::point_test!(test_push, $t);
            $crate::point_test!(test_basic, $t);
        };
    }
}
