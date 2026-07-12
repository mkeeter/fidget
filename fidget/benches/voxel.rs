use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const COLONNADE: &str = include_str!("../../models/colonnade.vm");

pub fn colonnade_voxel_exemplary(c: &mut Criterion) {
    let (ctx, root) = fidget::Context::from_text(COLONNADE.as_bytes()).unwrap();
    let vars = Default::default();
    let shape_vm = &fidget::vm::VmShape::new(&ctx, root).unwrap();
    let shape_vm = shape_vm.bind(&vars).unwrap();

    let mut group = c.benchmark_group("colonnade-voxel-exemplary");

    // Pick the standard scale / pitch / roll to generate a nice image
    let s = 1.0 / 0.7;
    let scale = nalgebra::Scale3::new(s, s, s);
    let pitch = nalgebra::Rotation3::new(
        nalgebra::Vector3::x() * 60.0 * std::f32::consts::PI / 180.0,
    );
    let roll = nalgebra::Rotation3::new(
        nalgebra::Vector3::z() * 30.0 * std::f32::consts::PI / 180.0,
    );
    let perspective = 0.3;
    let mut camera = nalgebra::Transform3::identity();
    *camera.matrix_mut().get_mut((3, 2)).unwrap() = perspective;
    let t = roll.to_homogeneous()
        * pitch.to_homogeneous()
        * scale.to_homogeneous()
        * camera.to_homogeneous();

    let cfg = &fidget::raster::voxel::RenderConfig {
        world_to_model: t,
        image_size: fidget::render::VoxelSize::from(1024),
        ..Default::default()
    };
    group.bench_function("vm", move |b| {
        b.iter(|| {
            let tape = shape_vm.clone();
            black_box(cfg.run(tape))
        })
    });

    #[cfg(feature = "jit")]
    {
        let shape_jit = fidget::jit::JitShape::new(&ctx, root).unwrap();
        let shape_jit = shape_jit.bind(&vars).unwrap();
        group.bench_function("jit", move |b| {
            b.iter(|| {
                let tape = shape_jit.clone();
                black_box(cfg.run(tape))
            })
        });
    }
}

criterion_group!(benches, colonnade_voxel_exemplary);
criterion_main!(benches);
