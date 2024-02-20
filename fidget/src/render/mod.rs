//! 2D and 3D rendering
//!
//! The easiest way to render something is with
//! [`RenderConfig::run`](RenderConfig::run); you can also use the lower-level
//! functions ([`render2d`](render2d()) and [`render3d`](render3d())) for manual
//! control over the input tape.
mod config;
mod render2d;
mod render3d;

pub use config::RenderConfig;
pub use render2d::render as render2d;
pub use render3d::render as render3d;

pub use render2d::{BitRenderMode, DebugRenderMode, RenderMode, SdfRenderMode};

#[cfg(test)]
mod test {
    use super::*;
    use crate::{eval::MathShape, vm::VmShape, Context};

    const HI: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/hi.vm"));
    #[test]
    fn render_hi_vm() {
        let (ctx, root) = Context::from_text(HI.as_bytes()).unwrap();
        let shape = VmShape::new(&ctx, root).unwrap();
        let cfg = RenderConfig::<2> {
            image_size: 32,
            ..RenderConfig::default()
        };
        let out = cfg.run(shape, &BitRenderMode).unwrap();
        let mut img_str = String::new();
        for (i, b) in out.iter().enumerate() {
            if i % 32 == 0 {
                img_str += "\n            ";
            }
            img_str.push(if *b { 'X' } else { '.' });
        }
        const EXPECTED: &str = "
            .................X..............
            .................X..............
            .................X..............
            .................X..........XX..
            .................X..........XX..
            .................X..............
            .................X..............
            .................XXXXXX.....XX..
            .................XXX..XX....XX..
            .................XX....XX...XX..
            .................X......X...XX..
            .................X......X...XX..
            .................X......X...XX..
            .................X......X...XX..
            .................X......X...XX..
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................
            ................................";
        if img_str != EXPECTED {
            println!("image mismatch detected!");
            println!("Expected:\n{EXPECTED}\nGot:\n{img_str}");
            panic!("image mismatch");
        }
    }
}
