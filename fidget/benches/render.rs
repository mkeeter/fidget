use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};

const PROSPERO: &str = include_str!("../../models/prospero.vm");

use fidget::eval::{MathShape, Shape};

pub fn prospero_size_sweep(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(PROSPERO.as_bytes()).unwrap();
    let shape_vm = &fidget::vm::VmShape::new(&ctx, root).unwrap();

    #[cfg(feature = "jit")]
    let shape_jit = &fidget::jit::JitShape::new(&ctx, root).unwrap();

    let mut group =
        c.benchmark_group("speed vs image size (prospero, 2d) (8 threads)");
    for size in [256, 512, 768, 1024, 1280, 1546, 1792, 2048] {
        let cfg = &fidget::render::RenderConfig {
            image_size: size,
            tile_sizes: fidget::vm::VmShape::tile_sizes_2d().to_vec(),
            threads: 8.try_into().unwrap(),
            ..Default::default()
        };
        group.bench_function(BenchmarkId::new("vm", size), move |b| {
            b.iter(|| {
                let tape = shape_vm.clone();
                black_box(fidget::render::render2d(
                    tape,
                    cfg,
                    &fidget::render::BitRenderMode,
                ))
            })
        });

        #[cfg(feature = "jit")]
        {
            let cfg = &fidget::render::RenderConfig {
                image_size: size,
                tile_sizes: fidget::jit::JitShape::tile_sizes_2d().to_vec(),
                threads: 8.try_into().unwrap(),
                ..Default::default()
            };
            group.bench_function(BenchmarkId::new("jit", size), move |b| {
                b.iter(|| {
                    let tape = shape_jit.clone();
                    black_box(fidget::render::render2d(
                        tape,
                        cfg,
                        &fidget::render::BitRenderMode,
                    ))
                })
            });
        }
    }
}

pub fn prospero_thread_sweep(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(PROSPERO.as_bytes()).unwrap();
    let shape_vm = &fidget::vm::VmShape::new(&ctx, root).unwrap();

    #[cfg(feature = "jit")]
    let shape_jit = &fidget::jit::JitShape::new(&ctx, root).unwrap();

    let mut group =
        c.benchmark_group("speed vs threads (prospero, 2d) (1024 x 1024)");
    for threads in [1, 2, 4, 8, 16] {
        let cfg = &fidget::render::RenderConfig {
            image_size: 1024,
            tile_sizes: fidget::vm::VmShape::tile_sizes_2d().to_vec(),
            threads: threads.try_into().unwrap(),
            ..Default::default()
        };
        group.bench_function(BenchmarkId::new("vm", threads), move |b| {
            b.iter(|| {
                let tape = shape_vm.clone();
                black_box(fidget::render::render2d(
                    tape,
                    cfg,
                    &fidget::render::BitRenderMode,
                ))
            })
        });
        #[cfg(feature = "jit")]
        {
            let cfg = &fidget::render::RenderConfig {
                image_size: 1024,
                tile_sizes: fidget::jit::JitShape::tile_sizes_2d().to_vec(),
                threads: threads.try_into().unwrap(),
                ..Default::default()
            };
            group.bench_function(BenchmarkId::new("jit", threads), move |b| {
                b.iter(|| {
                    let tape = shape_jit.clone();
                    black_box(fidget::render::render2d(
                        tape,
                        cfg,
                        &fidget::render::BitRenderMode,
                    ))
                })
            });
        }
    }
}

criterion_group!(benches, prospero_size_sweep, prospero_thread_sweep);
criterion_main!(benches);
