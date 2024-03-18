//! Test suites for each evaluator type
pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;

use crate::context::{Context, Node};

/// Builds a function which stresses the register allocator and function caller
pub(crate) fn build_stress_fn(n: usize) -> (Context, Node) {
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
    let mut args = (-n..n).map(|i| i as f32 / (n as f32)).collect::<Vec<_>>();
    args.push(0.0);
    args.push(1.0);
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
