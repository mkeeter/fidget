//! Voxel (3D) evaluaton and post-processing
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub mod color;
pub mod heightmap;
pub mod merge;
pub mod render;
pub mod shade;
pub mod ssao;

const TRANSFORM_INPUT: &str = include_str!("shaders/transform_input.wgsl");
const PACKED_VOXEL: &str = include_str!("shaders/packed_voxel.wgsl");

/// Packed voxel structure used on the GPU
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
pub struct PackedVoxel {
    /// XY components of the normal (normalized to a length of 127)
    ///
    /// The Z component is implied and positive
    ///
    /// An invalid normal is represented by `[-128, -128]`.
    pub normal: [i8; 2],

    /// Shape index
    pub index: u16,

    /// Depth of the voxel
    ///
    /// If this is 0, then the voxel is not populated
    pub z: u32,
}

#[cfg(test)]
mod test {
    use super::{PackedVoxel, *};
    use crate::{Context as _, Gpu, color::ShapeColor};

    use fidget_core::{context::Tree, render::VoxelSize, vm::VmShape};
    use std::collections::HashSet;
    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

    #[test]
    fn packed_voxel_size() {
        assert_eq!(std::mem::size_of::<PackedVoxel>(), 8);
    }

    #[derive(
        Copy,
        Clone,
        Debug,
        IntoBytes,
        Immutable,
        FromBytes,
        KnownLayout,
        PartialEq,
        Eq,
        Hash,
    )]
    #[repr(C)]
    struct Rgba {
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    }

    struct RenderOutput {
        merged: Vec<PackedVoxel>,
        colors: Vec<Rgba>,
        shaded: fidget_raster::Image<u32, VoxelSize>,
        heightmap: fidget_raster::Image<u32, VoxelSize>,
    }

    fn render(
        shapes: &[(Tree, ShapeColor<Tree>)],
        size: VoxelSize,
        settings: render::Settings,
    ) -> RenderOutput {
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let voxel_ctx = render::Context::new(&gpu);
        let merge_ctx = merge::Context::new(&gpu);
        let shade_ctx = shade::Context::new(&gpu);
        let heightmap_ctx = heightmap::Context::new(&gpu);
        let color_ctx = color::Context::new(&gpu);
        let ssao_ctx = ssao::Context::new(&gpu);

        let mut workspace = voxel_ctx.workspace();
        let mut buffers = voxel_ctx.buffers(size).unwrap();
        let mut merge_workspace = merge_ctx.workspace();
        let mut merge_buf = merge_ctx.buffers(size).unwrap();
        let mut shade_workspace = shade_ctx.workspace();
        let mut shade_buf = shade_ctx.buffers(size).unwrap();
        let mut heightmap_workspace = heightmap_ctx.workspace();

        // Render and accumulate each shape
        for (shape, _) in shapes {
            let shape = gpu
                .build_render_shape(&VmShape::from(shape.clone()))
                .unwrap();
            voxel_ctx
                .submit(&gpu, &shape, settings, &mut workspace, &mut buffers)
                .unwrap();
            merge_ctx
                .submit(
                    &gpu,
                    &buffers,
                    Default::default(),
                    &mut merge_workspace,
                    &mut merge_buf,
                )
                .unwrap();
        }
        let merged = gpu.read_vec(merge_buf.output());
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

        // Compute SSAO buffer
        let mut ssao_buf = ssao_ctx.buffers(size.into()).unwrap();
        let mut ssao_workspace = ssao_ctx.workspace();
        ssao_ctx
            .submit(&gpu, &merge_buf, (), &mut ssao_workspace, &mut ssao_buf)
            .unwrap();

        // Compute per-pixel colors
        color_ctx
            .submit(
                &gpu,
                color::Args {
                    image: &merge_buf,
                    color: &shape_colors,
                },
                color::Settings {
                    world_to_model: settings.world_to_model,
                    ..Default::default()
                },
                &mut color_workspace,
                &mut shade_buf,
            )
            .unwrap();

        // At this point, we should have either transparent (white) or opaque
        // colored pixels in the color output buffer
        let raw_colors = gpu.read_vec(shade_buf.output());
        let colors = <[Rgba]>::ref_from_bytes(raw_colors.as_bytes())
            .unwrap()
            .to_vec();

        // Compute shaded image (with color, overwriting shade_buf)
        let mut shade_out = gpu.read_buffer_for(shade_buf.output());
        shade_ctx
            .submit(
                &gpu,
                &merge_buf,
                Some(&ssao_buf),
                &mut shade_workspace,
                &mut shade_buf,
            )
            .unwrap();
        gpu.copy(shade_buf.output(), &mut shade_out).unwrap();
        let shaded = gpu.map_image(&mut shade_out).image();

        // Recompute per-pixel colors
        color_ctx
            .submit(
                &gpu,
                color::Args {
                    image: &merge_buf,
                    color: &shape_colors,
                },
                color::Settings {
                    world_to_model: settings.world_to_model,
                    ..Default::default()
                },
                &mut color_workspace,
                &mut shade_buf,
            )
            .unwrap();
        // Compute heightmap
        heightmap_ctx
            .submit(
                &gpu,
                &merge_buf,
                (),
                &mut heightmap_workspace,
                &mut shade_buf,
            )
            .unwrap();
        gpu.copy(shade_buf.output(), &mut shade_out).unwrap();
        let heightmap = gpu.map_image(&mut shade_out).image();

        RenderOutput {
            merged,
            colors,
            shaded,
            heightmap,
        }
    }

    #[test]
    fn voxel_pipeline() {
        crate::require_gpu!();
        let size = 128;

        let (x, y, z) = Tree::axes();
        let x_ = x.clone() - 0.2;
        let sphere1 = (x_.square() + y.square() + z.square()).sqrt()
            - Tree::constant(0.5);
        let x_ = x + 0.2;
        let sphere2 = (x_.square() + y.square() + z.square()).sqrt()
            - Tree::constant(0.5);
        let spheres = sphere1.min(sphere2);
        let out = render(
            &[(
                spheres,
                ShapeColor::Rgb {
                    r: Tree::constant(0.0),
                    g: Tree::constant(1.0),
                    b: Tree::constant(0.25),
                },
            )],
            size.into(),
            Default::default(),
        );

        let color_set = out.colors.iter().cloned().collect::<HashSet<_>>();
        assert!(
            color_set.contains(&Rgba {
                a: 0,
                r: 0xFF,
                g: 0xFF,
                b: 0xFF
            }) && color_set.contains(&Rgba {
                a: 0xFF,
                r: 0x00,
                g: 0xFF,
                b: 0x3F
            }),
            "invalid color set {color_set:X?} in {} pixels",
            out.colors.len()
        );

        let (shaded, img_size) = out.shaded.take();
        assert_eq!(img_size.width(), size);
        assert_eq!(img_size.height(), size);

        for (m, c) in out.merged.iter().zip(out.colors.iter()) {
            let p = m.z;
            if p == 0 {
                assert_eq!(c.a, 0);
            } else {
                assert_eq!(c.a, 0xFF);
            }
        }
        for (m, c) in out.merged.iter().zip(shaded.iter()) {
            let p = m.z;
            if p == 0 {
                assert_eq!(*c, 0);
            } else {
                // Check that the alpha channel is fully opaque
                assert_eq!(c & (0xFF << 24), (0xFF << 24));
                // Check that the other colors are approximately correct
                let g = (c >> 8) & 0xFF;
                let b = (c >> 16) & 0xFF;
                assert!(
                    (g / 4).abs_diff(b) < 2,
                    "invalid G/B ratio {g}/{b} (expected 4)"
                );
            }
        }
        for (m, c) in out.merged.iter().zip(out.heightmap.iter()) {
            let p = m.z;
            if p == 0 {
                assert_eq!(*c, 0);
            } else {
                // Check that the alpha channel is fully opaque
                assert_eq!(c & (0xFF << 24), (0xFF << 24));
                // Check that the other colors are approximately correct
                let g = (c >> 8) & 0xFF;
                let b = (c >> 16) & 0xFF;
                assert!(
                    (g / 4).abs_diff(b) < 2,
                    "invalid G/B ratio {g}/{b} (expected 4)"
                );
            }
        }
        // Make sure that the heightmap and shaded images are different!
        assert!(out.heightmap.iter().zip(shaded.iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn voxel_transform() {
        crate::require_gpu!();
        let (x, y, z) = Tree::axes();
        let sphere =
            (x.square() + y.square() + z.square()).sqrt() - Tree::constant(0.5);

        let size = 64;
        let cfg = render::Settings {
            world_to_model: nalgebra::Translation3::new(-0.5, 0.0, 0.0)
                .to_homogeneous(),
            ..Default::default()
        };
        let out = render(
            &[(
                sphere.clone(),
                ShapeColor::Rgb {
                    r: Tree::x() + 0.5, // offset to avoid saturation at 0
                    g: Tree::y() + 0.5, // offset to avoid saturation at 0
                    b: Tree::z(),
                },
            )],
            size.into(),
            cfg,
        );

        let mut iter = out.colors.iter();
        let mut pixels = String::with_capacity(out.colors.len());
        for _y in 0..size {
            for _x in 0..size {
                pixels += if iter.next().unwrap().a == 0 {
                    "."
                } else {
                    "#"
                };
            }
            pixels += "\n";
        }
        assert_eq!(
            pixels,
            "\
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ...........................................###########..........
                .........................................###############........
                .......................................###################......
                ......................................#####################.....
                .....................................#######################....
                ....................................#########################...
                ...................................###########################..
                ...................................###########################..
                ..................................#############################.
                ..................................#############################.
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                .................................###############################
                ..................................#############################.
                ..................................#############################.
                ...................................###########################..
                ...................................###########################..
                ....................................#########################...
                .....................................#######################....
                ......................................#####################.....
                .......................................###################......
                .........................................###############........
                ...........................................###########..........
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
            "
            .replace(" ", "")
        );

        let (_out, img_size) = out.shaded.take();
        assert_eq!(img_size.width(), size);
        assert_eq!(img_size.height(), size);

        for (m, c) in out.merged.iter().zip(out.colors.iter()) {
            let p = m.z;
            if p == 0 {
                assert_eq!(c.a, 0);
            } else {
                assert_eq!(c.a, 0xFF);
            }
        }

        // Every pixel should have a unique color, plus the background color
        let color_set = out.colors.iter().cloned().collect::<HashSet<_>>();
        assert_eq!(
            color_set.len(),
            pixels.chars().filter(|c| *c == '#').count() + 1,
            "invalid color set {color_set:X?} in {} pixels",
            out.colors.len()
        );

        ////////////////////////////////////////////////////////////////////////
        // Now, run the same test with a different transform.  This lets us
        // confirm that colors are pinned to the correct space corodinates
        let cfg = render::Settings {
            world_to_model: nalgebra::Translation3::new(0.5, 0.0, 0.0)
                .to_homogeneous(),
            ..Default::default()
        };
        let out = render(
            &[(
                sphere,
                ShapeColor::Rgb {
                    r: Tree::x() + 0.5, // offset to avoid saturation at 0
                    g: Tree::y() + 0.5, // offset to avoid saturation at 0
                    b: Tree::z(),
                },
            )],
            size.into(),
            cfg,
        );

        let mut iter = out.colors.iter();
        let mut pixels = String::with_capacity(out.colors.len());
        for _y in 0..size {
            for _x in 0..size {
                pixels += if iter.next().unwrap().a == 0 {
                    "."
                } else {
                    "#"
                };
            }
            pixels += "\n";
        }
        assert_eq!(
            pixels,
            "\
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ...........###########..........................................
                .........###############........................................
                .......###################......................................
                ......#####################.....................................
                .....#######################....................................
                ....#########################...................................
                ...###########################..................................
                ...###########################..................................
                ..#############################.................................
                ..#############################.................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                .###############################................................
                ..#############################.................................
                ..#############################.................................
                ...###########################..................................
                ...###########################..................................
                ....#########################...................................
                .....#######################....................................
                ......#####################.....................................
                .......###################......................................
                .........###############........................................
                ...........###########..........................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
                ................................................................
            "
            .replace(" ", "")
        );

        let (_out, img_size) = out.shaded.take();
        assert_eq!(img_size.width(), size);
        assert_eq!(img_size.height(), size);

        for (m, c) in out.merged.iter().zip(out.colors.iter()) {
            let p = m.z;
            if p == 0 {
                assert_eq!(c.a, 0);
            } else {
                assert_eq!(c.a, 0xFF);
            }
        }

        // Every pixel should have a unique color, plus the background color
        let color_set = out.colors.iter().cloned().collect::<HashSet<_>>();
        assert_eq!(
            color_set.len(),
            pixels.chars().filter(|c| *c == '#').count() + 1,
            "invalid color set {color_set:X?} in {} pixels",
            out.colors.len()
        );
    }

    #[test]
    fn voxel_hsl() {
        crate::require_gpu!();
        let size = 128;
        let (x, y, z) = Tree::axes();
        let x_ = x.clone() - 0.2;
        let sphere1 = (x_.square() + y.square() + z.square()).sqrt()
            - Tree::constant(0.5);
        let x_ = x + 0.2;
        let sphere2 = (x_.square() + y.square() + z.square()).sqrt()
            - Tree::constant(0.5);
        let out = render(
            &[
                (
                    sphere1,
                    ShapeColor::Hsl {
                        h: Tree::constant(0.0),
                        s: Tree::constant(1.0),
                        l: Tree::constant(0.5),
                    },
                ),
                (
                    sphere2,
                    ShapeColor::Hsl {
                        h: Tree::constant(0.5),
                        s: Tree::constant(0.4),
                        l: Tree::constant(0.7),
                    },
                ),
            ],
            size.into(),
            Default::default(),
        );

        let color_set = out.colors.iter().cloned().collect::<HashSet<_>>();
        assert_eq!(color_set.len(), 3);
        assert!(
            color_set.contains(&Rgba {
                r: 0xFF,
                g: 0xFF,
                b: 0xFF,
                a: 0x00,
            }) && color_set.contains(&Rgba {
                r: 0xFF,
                g: 0x00,
                b: 0x00,
                a: 0xFF,
            }) && color_set.contains(&Rgba {
                r: 0x93,
                g: 0xD1,
                b: 0xD1,
                a: 0xFF
            }),
            "invalid color set {color_set:X?} in {} pixels",
            out.colors.len()
        );
        println!("{color_set:#x?}");
    }
}
