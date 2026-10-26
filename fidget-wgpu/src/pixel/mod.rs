//! 2D (pixel) rendering
pub use fidget_raster::pixel::RenderSize;
pub mod color;
pub mod merge;
pub mod render;

const DISTANCE_PIXEL_SHADER: &str = include_str!("shaders/distance_pixel.wgsl");
const TRANSFORM_INPUT: &str = include_str!("shaders/transform_input.wgsl");

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Context, Gpu, color::ShapeColor};
    use fidget_core::{context::Tree, render::ImageSize, vm::VmShape};
    use fidget_raster::pixel::Image;

    struct RenderOutput {
        distance: Image,
        color: fidget_raster::Image<u32, ImageSize>,
    }

    fn render(
        shapes: &[(Tree, ShapeColor<Tree>)],
        image_size: ImageSize,
        vars: &fidget_core::shape::ShapeVars<f32>,
    ) -> RenderOutput {
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let pixel_ctx = render::Context::new(&gpu);
        let merge_ctx = merge::Context::new(&gpu);
        let color_ctx = color::Context::new(&gpu);

        let mut workspace = pixel_ctx.workspace();
        let mut buf = pixel_ctx.buffers(image_size).unwrap();
        let mut merge_workspace = merge_ctx.workspace();
        let mut merge_buf = merge_ctx.buffers(image_size).unwrap();

        // Render and accumulate each shape
        for (shape, _) in shapes {
            let shape = gpu
                .build_render_shape(&VmShape::from(shape.clone()))
                .unwrap();
            pixel_ctx
                .submit(
                    &gpu,
                    render::Args { shape: &shape },
                    render::Settings {
                        vars,
                        ..render::Settings::default()
                    },
                    &mut workspace,
                    &mut buf,
                )
                .unwrap();
            merge_ctx
                .submit(&buf, true, &mut merge_workspace, &mut merge_buf)
                .unwrap();
        }
        let shape_colors = shapes
            .iter()
            .map(|(_, c)| match c {
                ShapeColor::Rgb { r, g, b } => ShapeColor::Rgb {
                    r: VmShape::from(r.clone()),
                    g: VmShape::from(g.clone()),
                    b: VmShape::from(b.clone()),
                },
                ShapeColor::Hsl { h, s, l } => ShapeColor::Hsl {
                    h: VmShape::from(h.clone()),
                    s: VmShape::from(s.clone()),
                    l: VmShape::from(l.clone()),
                },
            })
            .collect::<Vec<_>>();
        let shape_colors = gpu.build_render_color(&shape_colors).unwrap();
        let mut color_workspace = color_ctx.workspace();

        // Compute per-pixel colors
        color_ctx
            .submit_with_vars(
                &mut merge_buf,
                color::ColorSettings {
                    z: 0.0,
                    only_filled: false,
                    world_to_model: nalgebra::Matrix3::identity(),
                },
                &shape_colors,
                &mut color_workspace,
                vars,
            )
            .unwrap();

        let mut distance_out = gpu.read_buffer_for(merge_buf.output_distance());
        let mut color_out =
            gpu.read_buffer_for(merge_buf.output_color().unwrap());
        gpu.copy(merge_buf.output_distance(), &mut distance_out)
            .unwrap();
        gpu.copy(merge_buf.output_color().unwrap(), &mut color_out)
            .unwrap();

        RenderOutput {
            color: gpu.map_image(&mut color_out).image(),
            distance: gpu.map_image(&mut distance_out).image(),
        }
    }

    #[test]
    fn pixel_pipeline() {
        crate::require_gpu!();

        let (x, y, _z) = Tree::axes();
        let circle = (x.square() + y.square()).sqrt() - Tree::constant(0.5);

        // Test a variety of image sizes for correctness
        for image_size in [
            RenderSize::new(64, 64),
            RenderSize::new(128, 64),
            RenderSize::new(64, 128),
            RenderSize::new(27, 51),
        ] {
            let out = render(
                &[(
                    circle.clone(),
                    ShapeColor::Rgb {
                        r: Tree::x(),
                        g: Tree::y(),
                        b: Tree::constant(0.5),
                    },
                )],
                image_size,
                &Default::default(),
            );
            assert_eq!(out.color.size(), image_size);
            assert_eq!(out.distance.size(), image_size);

            // Basic circle inside/outside check
            let mat = image_size.screen_to_world();
            for j in 0..image_size.height() {
                for i in 0..image_size.width() {
                    let pos = mat.transform_point(&nalgebra::Point2::new(
                        i as f32, j as f32,
                    ));
                    let p = out.distance[(j as usize, i as usize)];
                    let r = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
                    if r < 0.5 {
                        assert!(
                            p.inside(),
                            "pixel should be inside at pixel ({i}, {j}) \
                            (pos {pos}) with radius {r}"
                        );
                    } else {
                        assert!(
                            !p.inside(),
                            "pixel should be outside at pixel ({i}, {j}) \
                            (pos {pos}) with radius {r}"
                        );
                    }
                }
            }

            for j in 0..image_size.height() {
                for i in 0..image_size.width() {
                    let pos = mat.transform_point(&nalgebra::Point2::new(
                        i as f32, j as f32,
                    ));
                    let p = out.color[(j as usize, i as usize)];
                    let r = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
                    let alpha = if r < 0.5 { 255 } else { 0 };
                    let expected_color = u32::from_ne_bytes([
                        (pos.x.clamp(0.0, 1.0) * 255.0) as u8,
                        (pos.y.clamp(0.0, 1.0) * 255.0) as u8,
                        127,
                        alpha,
                    ]);
                    assert_eq!(
                        p, expected_color,
                        "color mismatch at {i}, {j} ({pos})"
                    );
                }
            }
        }
    }

    #[test]
    fn pixel_multiple_images() {
        crate::require_gpu!();

        let circle_a = ((Tree::x() - 0.5).square() + Tree::y().square()).sqrt()
            - Tree::constant(0.25);
        let circle_b = ((Tree::x() + 0.5).square() + Tree::y().square()).sqrt()
            - Tree::constant(0.25);

        // Test a variety of image sizes for correctness
        let image_size = RenderSize::new(64, 64);
        let out = render(
            &[
                (
                    circle_a,
                    ShapeColor::Rgb {
                        r: Tree::constant(0.0),
                        g: Tree::constant(0.0),
                        b: Tree::constant(1.0),
                    },
                ),
                (
                    circle_b,
                    ShapeColor::Rgb {
                        r: Tree::constant(0.0),
                        g: Tree::constant(1.0),
                        b: Tree::constant(0.0),
                    },
                ),
            ],
            image_size,
            &Default::default(),
        );
        assert_eq!(out.color.size(), image_size);
        assert_eq!(out.distance.size(), image_size);

        let mut pixels = String::new();
        for j in 0..image_size.height() {
            for i in 0..image_size.width() {
                let p = out.color[(j as usize, i as usize)];
                let c = match p.to_ne_bytes() {
                    [0, 0, 255, 0] => "b",
                    [0, 0, 255, 255] => "B",
                    [0, 255, 0, 0] => "g",
                    [0, 255, 0, 255] => "G",
                    _ => panic!("invalid color {p:?}"),
                };
                pixels += c;
            }
            pixels += "\n";
        }

        assert_eq!(
            pixels,
            "\
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            gggggggggggggGGGGGGGgggggggggggggbbbbbbbbbbbbBBBBBBBbbbbbbbbbbbb
            gggggggggggGGGGGGGGGGGgggggggggggbbbbbbbbbbBBBBBBBBBBBbbbbbbbbbb
            ggggggggggGGGGGGGGGGGGGggggggggggbbbbbbbbbBBBBBBBBBBBBBbbbbbbbbb
            ggggggggggGGGGGGGGGGGGGggggggggggbbbbbbbbbBBBBBBBBBBBBBbbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggggggggbbBBBBBBBBBBBBBBBgggggggg
            gggggggggGGGGGGGGGGGGGGGgggggggggggggggbbBBBBBBBBBBBBBBBbggggggg
            ggggggggggGGGGGGGGGGGGGggggggggggggggggbbbBBBBBBBBBBBBBbbggggggg
            ggggggggggGGGGGGGGGGGGGggggggggggggggggbbbBBBBBBBBBBBBBbbggggggg
            gggggggggggGGGGGGGGGGGgggggggggggggggggbbbbBBBBBBBBBBBbbbggggggg
            gggggggggggggGGGGGGGgggggggggggggggggggbbbbbbBBBBBBBbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            "
            .replace(" ", "")
        );
    }

    #[test]
    fn pixel_vars() {
        crate::require_gpu!();

        let va = fidget_core::var::Var::new();
        let vb = fidget_core::var::Var::new();
        let vc = fidget_core::var::Var::new();
        let vd = fidget_core::var::Var::new();
        let circle_a = ((Tree::x() - 0.5).square() + Tree::y().square()).sqrt()
            - Tree::from(va);
        let circle_b = ((Tree::x() + va).square() + Tree::y().square()).sqrt()
            - Tree::from(vb);

        // Test a variety of image sizes for correctness
        let image_size = RenderSize::new(64, 64);
        let mut vars = fidget_core::shape::ShapeVars::new();
        vars.insert(va.index().unwrap(), 0.25);
        vars.insert(vb.index().unwrap(), 0.55);
        vars.insert(vc.index().unwrap(), 0.75);
        vars.insert(vd.index().unwrap(), 0.5);
        let out = render(
            &[
                (
                    circle_a,
                    ShapeColor::Rgb {
                        r: Tree::constant(0.0),
                        g: Tree::constant(0.0),
                        b: Tree::from(vc),
                    },
                ),
                (
                    circle_b,
                    ShapeColor::Rgb {
                        r: Tree::constant(0.0),
                        g: Tree::from(vd),
                        b: Tree::constant(0.0),
                    },
                ),
            ],
            image_size,
            &vars,
        );
        assert_eq!(out.color.size(), image_size);
        assert_eq!(out.distance.size(), image_size);

        let mut pixels = String::new();
        for j in 0..image_size.height() {
            for i in 0..image_size.width() {
                let p = out.color[(j as usize, i as usize)];
                let c = match p.to_ne_bytes() {
                    [0, 0, 191, 0] => "b",
                    [0, 0, 191, 255] => "B",
                    [0, 127, 0, 0] => "g",
                    [0, 127, 0, 255] => "G",
                    _ => panic!("invalid color {:?}", p.to_ne_bytes()),
                };
                pixels += c;
            }
            pixels += "\n";
        }

        assert_eq!(
            pixels,
            "\
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            ggggggggggggggggggggGGGGGGGGGggggggggggggggggggggggggggggggggggg
            gggggggggggggggggGGGGGGGGGGGGGGGgggggggggggggbbbbbbbbbbbbggggggg
            gggggggggggggggGGGGGGGGGGGGGGGGGGGgggggggggggbbbbbbbbbbbbggggggg
            ggggggggggggggGGGGGGGGGGGGGGGGGGGGGgggggggggbbbbbbbbbbbbbggggggg
            gggggggggggggGGGGGGGGGGGGGGGGGGGGGGGggggggggbbbbbbbbbbbbbggggggg
            ggggggggggggGGGGGGGGGGGGGGGGGGGGGGGGGgggggggbbbbbbbbbbbbbggggggg
            gggggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGgggggbbbbbbbbbbbbbbggggggg
            ggggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGggggbbbbbbbbbbbbbbggggggg
            gggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgggbbbbbbbbbbbbbbggggggg
            gggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGggbbbbbbbbbbbbbbbbbbbbbb
            ggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgbbbBBBBBBBbbbbbbbbbbbb
            ggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgbBBBBBBBBBBBbbbbbbbbbb
            ggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgBBBBBBBBBBBBBbbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBbbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBbbbbbbbb
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBgggggggg
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBbggggggg
            gggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBbbggggggg
            ggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgBBBBBBBBBBBBBbbggggggg
            ggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgbBBBBBBBBBBBbbbggggggg
            ggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGgbbbBBBBBBBbbbbbggggggg
            gggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGggbbbbbbbbbbbbbbbggggggg
            gggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGggbbbbbbbbbbbbbbbggggggg
            ggggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGGGggggggggggggggggggggggggg
            gggggggggggGGGGGGGGGGGGGGGGGGGGGGGGGGGgggggggggggggggggggggggggg
            ggggggggggggGGGGGGGGGGGGGGGGGGGGGGGGGggggggggggggggggggggggggggg
            gggggggggggggGGGGGGGGGGGGGGGGGGGGGGGgggggggggggggggggggggggggggg
            ggggggggggggggGGGGGGGGGGGGGGGGGGGGGggggggggggggggggggggggggggggg
            gggggggggggggggGGGGGGGGGGGGGGGGGGGgggggggggggggggggggggggggggggg
            gggggggggggggggggGGGGGGGGGGGGGGGgggggggggggggggggggggggggggggggg
            ggggggggggggggggggggGGGGGGGGGggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
            "
            .replace(" ", "")
        );
    }

    #[test]
    fn pixel_hsl() {
        crate::require_gpu!();

        let circle_a = ((Tree::x() - 0.5).square() + Tree::y().square()).sqrt()
            - Tree::constant(0.25);
        let circle_b = ((Tree::x() + 0.5).square() + Tree::y().square()).sqrt()
            - Tree::constant(0.25);

        // Test a variety of image sizes for correctness
        let image_size = RenderSize::new(64, 64);
        let out = render(
            &[
                (
                    circle_a,
                    ShapeColor::Hsl {
                        h: Tree::constant(0.0),
                        s: Tree::constant(1.0),
                        l: Tree::constant(0.5),
                    },
                ),
                (
                    circle_b,
                    ShapeColor::Hsl {
                        h: Tree::constant(0.5),
                        s: Tree::constant(1.0),
                        l: Tree::constant(0.5),
                    },
                ),
            ],
            image_size,
            &Default::default(),
        );
        assert_eq!(out.color.size(), image_size);
        assert_eq!(out.distance.size(), image_size);

        let mut pixels = String::new();
        for j in 0..image_size.height() {
            for i in 0..image_size.width() {
                let p = out.color[(j as usize, i as usize)];
                let c = match p.to_ne_bytes() {
                    [255, 0, 0, 0] => "r",
                    [255, 0, 0, 255] => "R",
                    [0, 255, 255, 0] => "c",
                    [0, 255, 255, 255] => "C",
                    _ => panic!("invalid color {:?}", p.to_ne_bytes()),
                };
                pixels += c;
            }
            pixels += "\n";
        }

        assert_eq!(
            pixels,
            "\
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
            cccccccccccccCCCCCCCcccccccccccccrrrrrrrrrrrrRRRRRRRrrrrrrrrrrrr
            cccccccccccCCCCCCCCCCCcccccccccccrrrrrrrrrrRRRRRRRRRRRrrrrrrrrrr
            ccccccccccCCCCCCCCCCCCCccccccccccrrrrrrrrrRRRRRRRRRRRRRrrrrrrrrr
            ccccccccccCCCCCCCCCCCCCccccccccccrrrrrrrrrRRRRRRRRRRRRRrrrrrrrrr
            cccccccccCCCCCCCCCCCCCCCcccccccccrrrrrrrrRRRRRRRRRRRRRRRrrrrrrrr
            cccccccccCCCCCCCCCCCCCCCcccccccccrrrrrrrrRRRRRRRRRRRRRRRrrrrrrrr
            cccccccccCCCCCCCCCCCCCCCcccccccccrrrrrrrrRRRRRRRRRRRRRRRrrrrrrrr
            cccccccccCCCCCCCCCCCCCCCcccccccccrrrrrrrrRRRRRRRRRRRRRRRrrrrrrrr
            cccccccccCCCCCCCCCCCCCCCcccccccccrrrrrrrrRRRRRRRRRRRRRRRrrrrrrrr
            cccccccccCCCCCCCCCCCCCCCcccccccccccccccrrRRRRRRRRRRRRRRRcccccccc
            cccccccccCCCCCCCCCCCCCCCcccccccccccccccrrRRRRRRRRRRRRRRRrccccccc
            ccccccccccCCCCCCCCCCCCCccccccccccccccccrrrRRRRRRRRRRRRRrrccccccc
            ccccccccccCCCCCCCCCCCCCccccccccccccccccrrrRRRRRRRRRRRRRrrccccccc
            cccccccccccCCCCCCCCCCCcccccccccccccccccrrrrRRRRRRRRRRRrrrccccccc
            cccccccccccccCCCCCCCcccccccccccccccccccrrrrrrRRRRRRRrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccrrrrrrrrrrrrrrrrrrccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            "
            .replace(" ", "")
        );
    }
}
