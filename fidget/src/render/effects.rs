//! Post-processing effects for rendered images

use super::{ColorImage, DepthImage, NormalImage};
use nalgebra::{Vector3, Vector4};

/// Combines two images with shading
///
/// # Panics
/// If the images have different widths or heights
pub fn apply_shading(depth: &DepthImage, norm: &NormalImage) -> ColorImage {
    assert_eq!(depth.width(), norm.width());
    assert_eq!(depth.height(), norm.height());

    let mut out = ColorImage::new(depth.width(), depth.height());
    for y in 0..depth.height() {
        for x in 0..depth.width() {
            let pos = (y, x);
            if depth[pos] > 0 {
                out[pos] = shade_pixel(depth, norm, pos);
            }
        }
    }
    out
}

/// Compute shading for a single pixel
fn shade_pixel(
    depth: &DepthImage,
    norm: &NormalImage,
    pos: (usize, usize),
) -> [u8; 3] {
    let [nx, ny, nz] = norm[pos];
    let n = Vector3::new(nx, ny, nz).normalize();

    // Convert from pixel to world coordinates
    // XXX we're missing the actual depth scale
    let p = Vector3::new(
        2.0 * (pos.1 as f32 / depth.width() as f32 - 0.5),
        2.0 * (pos.0 as f32 / depth.height() as f32 - 0.5),
        2.0 * (depth[pos] as f32 / depth.width() as f32 - 0.5),
    );

    let lights = [
        Vector4::new(5.0, -5.0, 10.0, 0.5),
        Vector4::new(-5.0, 0.0, 10.0, 0.15),
        Vector4::new(0.0, -5.0, 10.0, 0.15),
    ];
    let mut accum = 0.2; // ambient
    for light in lights {
        let light_dir = (light.xyz() - p).normalize();
        accum += light_dir.dot(&n).max(0.0) * light.w;
    }

    // TODO SSAO dimming

    accum = accum.clamp(0.0, 1.0);

    let c = (accum * 255.0) as u8;
    [c, c, c]
}
