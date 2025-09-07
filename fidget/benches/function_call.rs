use criterion::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use fidget_core::{
    context::{Context, Node},
    eval::{Function, MathFunction},
    shape::{EzShape, Shape},
};

pub fn run_bench<F: Function + MathFunction>(
    c: &mut Criterion,
    ctx: Context,
    node: Node,
    test_name: &'static str,
    name: &'static str,
) {
    let shape_vm = &Shape::<F>::new(&ctx, node).unwrap();

    let mut eval = Shape::<F>::new_float_slice_eval();
    let tape = shape_vm.ez_float_slice_tape();

    let mut group = c.benchmark_group(test_name);
    for n in [10, 100, 1000] {
        let data = (0..n).map(|i| i as f32 / n as f32).collect::<Vec<f32>>();
        let t = &tape;
        group.bench_function(BenchmarkId::new(name, n), |b| {
            b.iter(|| {
                black_box(eval.eval(t, &data, &data, &data).unwrap());
            })
        });
    }
}

pub fn test_single_fn<F: Function + MathFunction>(
    c: &mut Criterion,
    name: &'static str,
) {
    let mut ctx = Context::new();
    let x = ctx.x();
    let f = ctx.sin(x).unwrap();

    run_bench::<F>(c, ctx, f, "single function", name);
}

pub fn test_many_fn<F: Function + MathFunction>(
    c: &mut Criterion,
    name: &'static str,
) {
    let mut ctx = Context::new();
    let x = ctx.x();
    let f = ctx.sin(x).unwrap();
    let y = ctx.y();
    let g = ctx.cos(y).unwrap();
    let z = ctx.z();
    let h = ctx.exp(z).unwrap();

    let out = ctx.add(f, g).unwrap();
    let out = ctx.add(out, h).unwrap();

    run_bench::<F>(c, ctx, out, "many functions", name);
}

pub fn test_single_fns(c: &mut Criterion) {
    test_single_fn::<fidget::vm::VmFunction>(c, "vm");
    #[cfg(feature = "jit")]
    test_single_fn::<fidget::jit::JitFunction>(c, "jit");
}

pub fn test_many_fns(c: &mut Criterion) {
    test_many_fn::<fidget::vm::VmFunction>(c, "vm");
    #[cfg(feature = "jit")]
    test_many_fn::<fidget::jit::JitFunction>(c, "jit");
}

criterion_group!(benches, test_single_fns, test_many_fns);
criterion_main!(benches);
