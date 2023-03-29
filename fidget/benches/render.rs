use criterion::{black_box, criterion_group, criterion_main, Criterion};

const PROSPERO: &str = include_str!("../../models/prospero.vm");

use fidget::{eval::Family, jit::Eval};

pub fn criterion_benchmark(c: &mut Criterion) {
    let cfg = fidget::render::RenderConfig {
        image_size: 1024,
        tile_sizes: Eval::tile_sizes_2d().to_vec(),
        threads: 8,

        mat: nalgebra::Transform2::identity(),
    };
    // fooo
    let (ctx, root) = fidget::Context::from_text(PROSPERO.as_bytes()).unwrap();
    let tape = ctx.get_tape::<Eval>(root).unwrap();

    c.bench_function("render 1024x1024", move |b| {
        let tape_ref = &tape;
        let cfg_ref = &cfg;
        b.iter(|| {
            let tape = tape_ref.clone();
            black_box(fidget::render::render2d(
                tape,
                cfg_ref,
                &fidget::render::BitRenderMode,
            ))
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
