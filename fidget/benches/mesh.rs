use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};

const COLONNADE: &str = include_str!("../../models/colonnade.vm");

pub fn colonnade_octree_thread_sweep(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(COLONNADE.as_bytes()).unwrap();
    let tape_jit = &ctx.get_tape::<fidget::jit::Eval>(root).unwrap();
    let tape_vm = &ctx.get_tape::<fidget::vm::Eval>(root).unwrap();

    let mut group =
        c.benchmark_group("speed vs threads (colonnade, octree) (depth 6)");
    for threads in [0, 4, 5, 6, 7, 8] {
        let cfg = &fidget::mesh::Settings {
            min_depth: 6,
            max_depth: 6,
            threads,
        };
        group.bench_function(BenchmarkId::new("jit", threads), move |b| {
            b.iter(|| {
                let cfg = *cfg;
                black_box(fidget::mesh::Octree::build(tape_jit, cfg))
            })
        });
        group.bench_function(BenchmarkId::new("vm", threads), move |b| {
            b.iter(|| {
                let cfg = *cfg;
                black_box(fidget::mesh::Octree::build(tape_vm, cfg))
            })
        });
    }
}

pub fn colonnade_mesh(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(COLONNADE.as_bytes()).unwrap();
    let tape_vm = &ctx.get_tape::<fidget::vm::Eval>(root).unwrap();
    let cfg = fidget::mesh::Settings {
        min_depth: 8,
        max_depth: 8,
        threads: 8,
    };
    let octree = &fidget::mesh::Octree::build(tape_vm, cfg);

    let mut group =
        c.benchmark_group("speed vs threads (colonnade, meshing) (depth 8)");
    for threads in [0, 4, 5, 6, 7, 8] {
        let cfg = &fidget::mesh::Settings { threads, ..cfg };
        group.bench_function(
            BenchmarkId::new("walk_dual", threads),
            move |b| {
                let cfg = *cfg;
                b.iter(|| black_box(octree.walk_dual(cfg)))
            },
        );
    }
}

criterion_group!(benches, colonnade_octree_thread_sweep, colonnade_mesh);
criterion_main!(benches);
