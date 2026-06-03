//! Integration test for 3D rendering with VM and JIT evaluators
use fidget::{
    context::Tree,
    eval::{Function, MathFunction},
    gui::View3,
    raster::voxel::{Image, RenderConfig, RenderSize},
    render::RenderHints,
    shape::{Shape, ShapeVars},
    var::Var,
};
use nalgebra::Vector3;

fn sphere_var<F: Function + MathFunction + RenderHints>() {
    let (x, y, z) = Tree::axes();
    let v = Var::new();
    let c = Tree::from(v);
    let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
    let shape = Shape::<F>::from(sphere);

    let size = 32;
    for scale in [1.0, 0.5] {
        let cfg = RenderConfig {
            image_size: RenderSize::from(size),
            world_to_model: View3::from_center_and_scale(
                Vector3::zeros(),
                scale,
            )
            .world_to_model(),
            ..Default::default()
        };
        for r in [0.5, 0.75] {
            let mut vars = ShapeVars::new();
            vars.insert(v.index().unwrap(), r);
            let image = cfg
                .run_with_vars::<_>(shape.clone(), &vars)
                .expect("rendering should not fail")
                .expect("rendering should not be cancelled");

            check_sphere(image, size, scale, r);
        }
    }
}

fn check_sphere(image: Image, size: u32, scale: f32, r: f32) {
    // Handwavey calculation: ±1 split into `size` voxels, max error
    // of two voxels (top to bottom), and dividing for `scale` for
    // bonus corrections.
    let epsilon = 2.0 / size as f32 / scale * 2.0;
    let m = RenderSize::from(size).screen_to_world();
    for (i, p) in image.iter().enumerate() {
        let p = p.depth;
        if p == size as f32 {
            // Skip saturated voxels
            continue;
        }
        let size = size as i32;
        let i = i as i32;
        let x = (i % size) as f32;
        let y = (i / size) as f32;
        let z = p;
        let pos = m.transform_point(&nalgebra::Point3::new(x, y, z)) * scale;
        if p == 0.0 {
            let v = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
            assert!(
                v + epsilon > r,
                "got z = 0 inside the sphere ({x}, {y}, {z}); \
                             radius is {v}"
            );
        } else {
            let v = (pos.x.powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt();
            let err = (r - v).abs();
            assert!(
                err < epsilon,
                "too much error {err} at ({x}, {y}, {z}) ({pos}) \
                             (scale = {scale}); radius is {v}, expected {r}"
            );
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

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;

    #[test]
    fn sphere_wgpu() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        // Build a WGPU instance without any special features
        // (e.g. no timestamp queries)
        use fidget_wgpu::wgpu;
        let instance = wgpu::Instance::default();
        let (device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap()
        });

        let (x, y, z) = Tree::axes();
        let ctx = fidget_wgpu::voxel::Context::new(device, queue);

        let size = 32;
        let image_size = RenderSize::from(size);
        let mut buf = ctx.buffers(image_size).unwrap();
        for scale in [1.0, 0.5] {
            for r in [0.5, 0.75] {
                let sphere = (x.square() + y.square() + z.square()).sqrt()
                    - Tree::constant(r);
                let shape =
                    ctx.shape(&fidget::vm::VmShape::from(sphere)).unwrap();
                let image = ctx
                    .run(
                        &shape,
                        &mut buf,
                        fidget::wgpu::voxel::RenderConfig {
                            world_to_model: View3::from_center_and_scale(
                                Vector3::zeros(),
                                scale,
                            )
                            .world_to_model(),
                        },
                    )
                    .unwrap();

                check_sphere(image, size, scale, r as f32);
            }
        }
    }

    #[test]
    fn sphere_wgpu_vars() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        // Build a WGPU instance without any special features
        // (e.g. no timestamp queries)
        use fidget_wgpu::wgpu;
        let instance = wgpu::Instance::default();
        let (device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap()
        });

        let (x, y, z) = Tree::axes();
        let v = Var::new();
        let c = Tree::from(v);
        let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
        let shape = fidget::vm::VmShape::from(sphere);
        let ctx = fidget_wgpu::voxel::Context::new(device, queue);
        let render_shape = ctx.shape(&shape).unwrap();

        let size = 32;
        let image_size = RenderSize::from(size);
        let mut buf = ctx.buffers(image_size).unwrap();
        for scale in [1.0, 0.5] {
            for r in [0.5, 0.75] {
                let mut vars = ShapeVars::new();
                vars.insert(v.index().unwrap(), r);
                let image = ctx
                    .run_with_vars(
                        &render_shape,
                        &vars,
                        &mut buf,
                        fidget::wgpu::voxel::RenderConfig {
                            world_to_model: View3::from_center_and_scale(
                                Vector3::zeros(),
                                scale,
                            )
                            .world_to_model(),
                        },
                    )
                    .unwrap();

                check_sphere(image, size, scale, r);
            }
        }
    }
}
