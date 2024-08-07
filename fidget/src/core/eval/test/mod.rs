//! Test suites for each evaluator type
pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;

// Internal-only tests
mod symbolic_deriv;

use crate::{
    context::{Context, IntoNode, Node},
    eval::Tape,
    var::Var,
};

/// Builds a function which stresses the register allocator and function caller
pub fn build_stress_fn(n: usize) -> (Context, Node) {
    let mut inputs = vec![];
    let mut ctx = Context::new();
    let mut sum = ctx.constant(0.0);
    let x = ctx.x();
    let y = ctx.y();
    let z = ctx.z();

    // Build up the sum (x * 1) + (y * 2) + (z * 3) + (x * 4) + ...
    //
    // We're going to both send these operations into a sin(..) node, then add
    // them afterwards (in reverse order), meaning their allocations must
    // persist beyond the sine.
    for i in 1..=n {
        let d = ctx.mul(i as f32, [x, y, z][i % 3]).unwrap();
        inputs.push(d);
        sum = ctx.add(sum, d).unwrap();
    }

    sum = ctx.sin(sum).unwrap();
    for i in inputs.into_iter().rev() {
        sum = ctx.add(sum, i).unwrap();
    }

    (ctx, sum)
}

/// Pick a bunch of arguments, some of which are spicy
fn test_args_n(n: i64) -> Vec<f32> {
    let mut args = (-n..=n)
        .map(|i| std::f32::consts::PI * 2.0 * i as f32 / (n as f32))
        .collect::<Vec<_>>();
    args.push(1.0);
    args.push(5.0);
    args.push(0.5);
    args.push(1.5);
    args.push(10.0);
    args.push(std::f32::consts::PI);
    args.push(std::f32::consts::FRAC_PI_2);
    args.push(std::f32::consts::FRAC_1_PI);
    args.push(std::f32::consts::SQRT_2);
    args.push(f32::NAN);
    args
}

fn test_args() -> Vec<f32> {
    test_args_n(32)
}

fn bind_xy<T: Tape, V, G: Into<V>>(tape: &T) -> Box<dyn Fn(G, G) -> [V; 2]> {
    let vars = tape.vars();
    let ix = vars[&Var::X];
    let iy = vars[&Var::Y];
    assert_ne!(ix, iy);
    Box::new(move |x, y| {
        let mut out = [None, None];
        out[ix] = Some(x.into());
        out[iy] = Some(y.into());
        out.map(Option::unwrap)
    })
}

fn bind_xyz<T: Tape, V, G: Into<V>>(
    tape: &T,
) -> Box<dyn Fn(G, G, G) -> [V; 3]> {
    let vars = tape.vars();
    let ix = vars[&Var::X];
    let iy = vars[&Var::Y];
    let iz = vars[&Var::Z];
    assert_ne!(ix, iy);
    assert_ne!(iy, iz);
    assert_ne!(ix, iz);
    Box::new(move |x, y, z| {
        let mut out = [None, None, None];
        out[ix] = Some(x.into());
        out[iy] = Some(y.into());
        out[iz] = Some(z.into());
        out.map(Option::unwrap)
    })
}

/// Trait for canonical evaluation testing of unary operations
pub trait CanonicalUnaryOp {
    const NAME: &'static str;
    fn build(ctx: &mut Context, arg: Node) -> Node;
    fn eval_f32(arg: f32) -> f32;
    fn eval_f64(arg: f64) -> f64;

    /// Returns true if there is a bidirectional discontinuity at a position
    ///
    /// This means that we should skip gradient checking, because we can't
    /// accurately estimate the gradient on either side.
    fn discontinuous_at(_lhs: f32) -> bool {
        false
    }
}

/// Trait for canonical evaluation testing of binary operations
pub trait CanonicalBinaryOp {
    const NAME: &'static str;
    fn build<A: IntoNode, B: IntoNode>(
        ctx: &mut Context,
        lhs: A,
        rhs: B,
    ) -> Node;
    fn eval_reg_reg_f32(lhs: f32, rhs: f32) -> f32;
    fn eval_reg_imm_f32(lhs: f32, rhs: f32) -> f32;
    fn eval_imm_reg_f32(lhs: f32, rhs: f32) -> f32;
    fn eval_reg_reg_f64(lhs: f64, rhs: f64) -> f64;
    fn eval_reg_imm_f64(lhs: f64, rhs: f64) -> f64;
    fn eval_imm_reg_f64(lhs: f64, rhs: f64) -> f64;

    /// Returns true if there is a bidirectional discontinuity at a position
    ///
    /// This means that we should skip gradient checking, because we can't
    /// accurately estimate the gradient on either side.
    fn discontinuous_at(_lhs: f32, _rhs: f32) -> bool {
        false
    }
}

macro_rules! declare_canonical_unary {
    (Context::$i:ident, |$a:ident| $t:expr, |$b:ident| $u:expr) => {
        pub struct $i;
        impl CanonicalUnaryOp for $i {
            const NAME: &'static str = stringify!($i);
            fn build(ctx: &mut Context, arg: Node) -> Node {
                Context::$i(ctx, arg).unwrap()
            }
            fn eval_f32($a: f32) -> f32 {
                $t
            }
            fn eval_f64($a: f64) -> f64 {
                $t
            }
            fn discontinuous_at($b: f32) -> bool {
                $u
            }
        }
    };
    (Context::$i:ident, |$lhs:ident| $t:expr) => {
        declare_canonical_unary!(Context::$i, |$lhs| $t, |_a| false);
    };
}

macro_rules! declare_canonical_binary {
    (Context::$i:ident, |$lhs:ident, $rhs:ident| $t:expr, |$lhs2:ident, $rhs2: ident| $d:expr) => {
        pub struct $i;
        impl CanonicalBinaryOp for $i {
            const NAME: &'static str = stringify!($i);
            fn build<A: IntoNode, B: IntoNode>(
                ctx: &mut Context,
                lhs: A,
                rhs: B,
            ) -> Node {
                let lhs = lhs.into_node(ctx).unwrap();
                let rhs = rhs.into_node(ctx).unwrap();
                Context::$i(ctx, lhs, rhs).unwrap()
            }
            fn eval_reg_reg_f32($lhs: f32, $rhs: f32) -> f32 {
                $t
            }
            fn eval_reg_imm_f32($lhs: f32, $rhs: f32) -> f32 {
                $t
            }
            fn eval_imm_reg_f32($lhs: f32, $rhs: f32) -> f32 {
                $t
            }
            fn eval_reg_reg_f64($lhs: f64, $rhs: f64) -> f64 {
                $t
            }
            fn eval_reg_imm_f64($lhs: f64, $rhs: f64) -> f64 {
                $t
            }
            fn eval_imm_reg_f64($lhs: f64, $rhs: f64) -> f64 {
                $t
            }
            fn discontinuous_at($lhs2: f32, $rhs2: f32) -> bool {
                $d
            }
        }
    };
    (Context::$i:ident, |$lhs:ident, $rhs:ident| $t:expr) => {
        declare_canonical_binary!(Context::$i, |$lhs, $rhs| $t, |_a, _b| false);
    };
}

macro_rules! declare_canonical_binary_full {
    (Context::$i:ident,
     |$lhs_reg_reg:ident, $rhs_reg_reg:ident| $f_reg_reg:expr,
     |$lhs_reg_imm:ident, $rhs_reg_imm:ident| $f_reg_imm:expr,
     |$lhs_imm_reg:ident, $rhs_imm_reg:ident| $f_imm_reg:expr,
     ) => {
        pub struct $i;
        impl CanonicalBinaryOp for $i {
            const NAME: &'static str = stringify!($i);
            fn build<A: IntoNode, B: IntoNode>(
                ctx: &mut Context,
                lhs: A,
                rhs: B,
            ) -> Node {
                let lhs = lhs.into_node(ctx).unwrap();
                let rhs = rhs.into_node(ctx).unwrap();
                Context::$i(ctx, lhs, rhs).unwrap()
            }
            fn eval_reg_reg_f32($lhs_reg_reg: f32, $rhs_reg_reg: f32) -> f32 {
                $f_reg_reg
            }
            fn eval_reg_imm_f32($lhs_reg_imm: f32, $rhs_reg_imm: f32) -> f32 {
                $f_reg_imm
            }
            fn eval_imm_reg_f32($lhs_imm_reg: f32, $rhs_imm_reg: f32) -> f32 {
                $f_imm_reg
            }
            fn eval_reg_reg_f64($lhs_reg_reg: f64, $rhs_reg_reg: f64) -> f64 {
                $f_reg_reg
            }
            fn eval_reg_imm_f64($lhs_reg_imm: f64, $rhs_reg_imm: f64) -> f64 {
                $f_reg_imm
            }
            fn eval_imm_reg_f64($lhs_imm_reg: f64, $rhs_imm_reg: f64) -> f64 {
                $f_imm_reg
            }
        }
    };
}

#[allow(non_camel_case_types, clippy::useless_conversion)]
pub mod canonical {
    use super::*;

    declare_canonical_unary!(Context::neg, |a| -a);
    declare_canonical_unary!(Context::recip, |a| 1.0 / a);
    declare_canonical_unary!(Context::abs, |a| a.abs());
    declare_canonical_unary!(Context::sin, |a| a.sin());
    declare_canonical_unary!(Context::cos, |a| a.cos());
    declare_canonical_unary!(Context::tan, |a| a.tan());
    declare_canonical_unary!(Context::asin, |a| a.asin());
    declare_canonical_unary!(Context::acos, |a| a.acos());
    declare_canonical_unary!(Context::atan, |a| a.atan());
    declare_canonical_unary!(Context::exp, |a| a.exp());
    declare_canonical_unary!(Context::ln, |a| a.ln());
    declare_canonical_unary!(Context::square, |a| a * a);
    declare_canonical_unary!(Context::sqrt, |a| a.sqrt());
    declare_canonical_unary!(Context::floor, |a| a.floor());
    declare_canonical_unary!(Context::ceil, |a| a.ceil());
    declare_canonical_unary!(Context::round, |a| a.round());
    declare_canonical_unary!(Context::not, |a| (a == 0.0).into(), |a| a == 0.0);

    declare_canonical_binary!(Context::add, |a, b| a + b);
    declare_canonical_binary!(Context::sub, |a, b| a - b);
    declare_canonical_binary_full!(
        Context::mul,
        |a, b| a * b,
        |a, imm| if imm == 0.0 { imm } else { a * imm },
        |imm, b| if imm == 0.0 { imm } else { imm * b },
    );
    declare_canonical_binary_full!(
        Context::div,
        |a, b| a / b,
        |a, imm| a / imm,
        |imm, b| if imm == 0.0 { imm } else { imm / b },
    );
    declare_canonical_binary!(
        Context::min,
        |a, b| if a.is_nan() || b.is_nan() {
            f32::NAN.into()
        } else {
            a.min(b)
        }
    );
    declare_canonical_binary!(
        Context::max,
        |a, b| if a.is_nan() || b.is_nan() {
            f32::NAN.into()
        } else {
            a.max(b)
        }
    );
    declare_canonical_binary!(
        Context::compare,
        |a, b| match a.partial_cmp(&b) {
            None => f32::NAN.into(),
            Some(v) => (v as i8).into(),
        },
        |a, b| a == b
    );
    declare_canonical_binary!(
        Context::modulo,
        |a, b| a.rem_euclid(b),
        |a, b| {
            let v = a / b;
            (v.round() - v).abs() < 1e-9
        }
    );
    declare_canonical_binary!(
        Context::and,
        |a, b| if a == 0.0 { a } else { b },
        |a, _b| a == 0.0 // discontinuity, because either side snaps to b
    );
    declare_canonical_binary!(
        Context::or,
        |a, b| if a != 0.0 { a } else { b },
        |a, _b| a == 0.0 // discontinuity, because either side snaps to a
    );
    declare_canonical_binary!(Context::atan2, |y, x| y.atan2(x));
}

#[macro_export]
macro_rules! one_unary_test {
    ($tester:ty, $i:ident) => {
        #[test]
        fn $i() {
            <$tester>::test_unary::<$crate::eval::test::canonical::$i>()
        }
    };
}

#[macro_export]
macro_rules! one_binary_test {
    ($tester:ty, $i:ident) => {
        #[test]
        fn $i() {
            <$tester>::test_binary::<$crate::eval::test::canonical::$i>()
        }
    };
}

#[macro_export]
macro_rules! all_unary_tests {
    ($tester:ty) => {
        $crate::one_unary_test!($tester, neg);
        $crate::one_unary_test!($tester, recip);
        $crate::one_unary_test!($tester, abs);
        $crate::one_unary_test!($tester, sin);
        $crate::one_unary_test!($tester, cos);
        $crate::one_unary_test!($tester, tan);
        $crate::one_unary_test!($tester, asin);
        $crate::one_unary_test!($tester, acos);
        $crate::one_unary_test!($tester, atan);
        $crate::one_unary_test!($tester, exp);
        $crate::one_unary_test!($tester, ln);
        $crate::one_unary_test!($tester, not);
        $crate::one_unary_test!($tester, square);
        $crate::one_unary_test!($tester, floor);
        $crate::one_unary_test!($tester, ceil);
        $crate::one_unary_test!($tester, round);
        $crate::one_unary_test!($tester, sqrt);
    };
}

#[macro_export]
macro_rules! all_binary_tests {
    ($tester:ty) => {
        $crate::one_binary_test!($tester, add);
        $crate::one_binary_test!($tester, sub);
        $crate::one_binary_test!($tester, mul);
        $crate::one_binary_test!($tester, div);
        $crate::one_binary_test!($tester, atan2);
        $crate::one_binary_test!($tester, min);
        $crate::one_binary_test!($tester, max);
        $crate::one_binary_test!($tester, compare);
        $crate::one_binary_test!($tester, modulo);
        $crate::one_binary_test!($tester, and);
        $crate::one_binary_test!($tester, or);
    };
}
