use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use fidget::{
    render::{ImageSize, ThreadCount},
    shape::RenderHints,
};

const PROSPERO: &str = include_str!("../../models/prospero.vm");

pub fn prospero_size_sweep(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(PROSPERO.as_bytes()).unwrap();
    let shape_vm = &fidget::vm::VmShape::new(&ctx, root).unwrap();

    #[cfg(feature = "jit")]
    let shape_jit = &fidget::jit::JitShape::new(&ctx, root).unwrap();

    let mut group =
        c.benchmark_group("speed vs image size (prospero, 2d) (8 threads)");
    for size in [256, 512, 768, 1024, 1280, 1546, 1792, 2048] {
        let cfg = &fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(size),
            tile_sizes: fidget::vm::VmFunction::tile_sizes_2d(),
            ..Default::default()
        };
        group.bench_function(BenchmarkId::new("vm", size), move |b| {
            b.iter(|| {
                let tape = shape_vm.clone();
                black_box(cfg.run::<_, fidget::render::BitRenderMode>(tape))
            })
        });

        #[cfg(feature = "jit")]
        {
            let cfg = &fidget::render::ImageRenderConfig {
                image_size: fidget::render::ImageSize::from(size),
                tile_sizes: fidget::jit::JitFunction::tile_sizes_2d(),
                ..Default::default()
            };
            group.bench_function(BenchmarkId::new("jit", size), move |b| {
                b.iter(|| {
                    let tape = shape_jit.clone();
                    black_box(cfg.run::<_, fidget::render::BitRenderMode>(tape))
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
    for threads in std::iter::once(ThreadCount::One).chain(
        [1, 2, 4, 8, 16].map(|i| ThreadCount::Many(i.try_into().unwrap())),
    ) {
        let cfg = &fidget::render::ImageRenderConfig {
            image_size: ImageSize::from(1024),
            tile_sizes: fidget::vm::VmFunction::tile_sizes_2d(),
            threads,
            ..Default::default()
        };
        group.bench_function(BenchmarkId::new("vm", threads), move |b| {
            b.iter(|| {
                let tape = shape_vm.clone();
                black_box(cfg.run::<_, fidget::render::BitRenderMode>(tape))
            })
        });
        #[cfg(feature = "jit")]
        {
            let cfg = &fidget::render::ImageRenderConfig {
                image_size: ImageSize::from(1024),
                tile_sizes: fidget::jit::JitFunction::tile_sizes_2d(),
                threads,
                ..Default::default()
            };
            group.bench_function(BenchmarkId::new("jit", threads), move |b| {
                b.iter(|| {
                    let tape = shape_jit.clone();
                    black_box(cfg.run::<_, fidget::render::BitRenderMode>(tape))
                })
            });
        }
    }
}

criterion_group!(benches, prospero_size_sweep, prospero_thread_sweep);
criterion_main!(benches);
