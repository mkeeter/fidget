use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const COLONNADE: &str = include_str!("../../models/colonnade.vm");

pub fn colonnade_octree_thread_sweep(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(COLONNADE.as_bytes()).unwrap();
    let shape_vm = &fidget::vm::VmShape::new(&ctx, root).unwrap();
    #[cfg(feature = "jit")]
    let shape_jit = &fidget::jit::JitShape::new(&ctx, root).unwrap();

    let mut group =
        c.benchmark_group("speed vs threads (colonnade, octree) (depth 6)");

    for threads in [None, Some(1), Some(4), Some(8)] {
        let pool = threads.map(|n| {
            fidget::render::ThreadPool::Custom(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap(),
            )
        });
        let cfg = &fidget::mesh::Settings {
            depth: 6,
            threads: pool.as_ref(),
            ..Default::default()
        };
        let threads = threads.unwrap_or(0);

        #[cfg(feature = "jit")]
        group.bench_function(BenchmarkId::new("jit", threads), move |b| {
            b.iter(|| black_box(fidget::mesh::Octree::build(shape_jit, cfg)))
        });
        group.bench_function(BenchmarkId::new("vm", threads), move |b| {
            b.iter(|| black_box(fidget::mesh::Octree::build(shape_vm, cfg)))
        });
    }
}

pub fn colonnade_mesh(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(COLONNADE.as_bytes()).unwrap();
    let shape_vm = &fidget::vm::VmShape::new(&ctx, root).unwrap();
    let cfg = fidget::mesh::Settings {
        depth: 8,
        ..Default::default()
    };
    let octree = &fidget::mesh::Octree::build(shape_vm, &cfg).unwrap();

    let mut group = c.benchmark_group("speed (colonnade, meshing) (depth 8)");
    group
        .bench_function(BenchmarkId::new("walk_dual", "colonnade"), move |b| {
            b.iter(|| black_box(octree.walk_dual()))
        });
}

criterion_group!(benches, colonnade_octree_thread_sweep, colonnade_mesh);
criterion_main!(benches);
