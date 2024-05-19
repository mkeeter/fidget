use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use fidget::{
    context::{Context, Node},
    shape::{BulkEvaluator, EzShape, MathShape, Shape},
};

pub fn run_bench<S: Shape + MathShape>(
    c: &mut Criterion,
    ctx: Context,
    node: Node,
    test_name: &'static str,
    name: &'static str,
) {
    let shape_vm = &S::new(&ctx, node).unwrap();

    let mut eval = S::new_float_slice_eval();
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

pub fn test_single_fn<S: Shape + MathShape>(
    c: &mut Criterion,
    name: &'static str,
) {
    let mut ctx = Context::new();
    let x = ctx.x();
    let f = ctx.sin(x).unwrap();

    run_bench::<S>(c, ctx, f, "single function", name);
}

pub fn test_many_fn<S: Shape + MathShape>(
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

    run_bench::<S>(c, ctx, out, "many functions", name);
}

pub fn test_single_fns(c: &mut Criterion) {
    test_single_fn::<fidget::vm::VmShape>(c, "vm");
    #[cfg(feature = "jit")]
    test_single_fn::<fidget::jit::JitShape>(c, "jit");
}

pub fn test_many_fns(c: &mut Criterion) {
    test_many_fn::<fidget::vm::VmShape>(c, "vm");
    #[cfg(feature = "jit")]
    test_many_fn::<fidget::jit::JitShape>(c, "jit");
}

criterion_group!(benches, test_single_fns, test_many_fns);
criterion_main!(benches);
