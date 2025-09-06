//! Integration test for 3D rendering with VM and JIT evaluators
use fidget::{
    context::Tree,
    eval::{Function, MathFunction},
    gui::View3,
    raster::VoxelRenderConfig,
    render::VoxelSize,
    shape::{Shape, ShapeVars},
    var::Var,
};
use nalgebra::Vector3;

fn sphere_var<F: Function + MathFunction>() {
    let (x, y, z) = Tree::axes();
    let v = Var::new();
    let c = Tree::from(v);
    let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
    let shape = Shape::<F>::from(sphere);

    let size = 32;
    for scale in [1.0, 0.5] {
        let cfg = VoxelRenderConfig {
            image_size: VoxelSize::from(size),
            world_to_model: View3::from_center_and_scale(
                Vector3::zeros(),
                scale,
            )
            .world_to_model(),
            ..Default::default()
        };
        let m = cfg.image_size.screen_to_world();

        for r in [0.5, 0.75] {
            let mut vars = ShapeVars::new();
            vars.insert(v.index().unwrap(), r);
            let image = cfg.run_with_vars::<_>(shape.clone(), &vars).unwrap();

            // Handwavey calculation: ±1 split into `size` voxels, max error
            // of two voxels (top to bottom), and dividing for `scale` for
            // bonus corrections.
            let epsilon = 2.0 / size as f32 / scale * 2.0;
            for (i, p) in image.iter().enumerate() {
                let p = p.depth;
                if p == size {
                    // Skip saturated voxels
                    continue;
                }
                let size = size as i32;
                let i = i as i32;
                let x = (i % size) as f32;
                let y = (i / size) as f32;
                let z = p as f32;
                let pos =
                    m.transform_point(&nalgebra::Point3::new(x, y, z)) * scale;
                if p == 0 {
                    let v = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
                    assert!(
                        v + epsilon > r,
                        "got z = 0 inside the sphere ({x}, {y}, {z}); \
                             radius is {v}"
                    );
                } else {
                    let v =
                        (pos.x.powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt();
                    let err = (r - v).abs();
                    assert!(
                        err < epsilon,
                        "too much error {err} at ({x}, {y}, {z}) ({pos}) \
                             (scale = {scale}); radius is {v}, expected {r}"
                    );
                }
            }
        }
    }
}

macro_rules! render_tests {
    ($i:ident, $ty:ty) => {
        mod $i {
            #[test]
            fn render_sphere_var() {
                super::sphere_var::<$ty>();
            }
        }
    };
}

render_tests!(vm, fidget::vm::VmFunction);
render_tests!(vm3, fidget::vm::GenericVmFunction<3>);

#[cfg(feature = "jit")]
render_tests!(jit, fidget::jit::JitFunction);
