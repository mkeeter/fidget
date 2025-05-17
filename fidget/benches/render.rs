use criterion::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use fidget::render::{ImageSize, RenderHints, ThreadPool};

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
    let pools = [1, 2, 4, 8, 16].map(|i| {
        Some(ThreadPool::Custom(
            rayon::ThreadPoolBuilder::new()
                .num_threads(i)
                .build()
                .unwrap(),
        ))
    });
    for threads in [None, Some(ThreadPool::Global)].into_iter().chain(pools) {
        let threads = threads.as_ref();
        let name = match &threads {
            None => "-".to_string(),
            Some(ThreadPool::Custom(i)) => i.current_num_threads().to_string(),
            Some(ThreadPool::Global) => "N".to_string(),
        };
        let cfg = &fidget::render::ImageRenderConfig {
            image_size: ImageSize::from(1024),
            tile_sizes: fidget::vm::VmFunction::tile_sizes_2d(),
            threads,
            ..Default::default()
        };
        group.bench_function(BenchmarkId::new("vm", &name), move |b| {
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
            group.bench_function(BenchmarkId::new("jit", &name), move |b| {
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
