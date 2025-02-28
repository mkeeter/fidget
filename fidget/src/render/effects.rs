//! Post-processing effects for rendered images

use super::{ColorImage, DepthImage, Image, NormalImage, ThreadPool};
use nalgebra::{
    Const, Matrix3, MatrixXx2, MatrixXx3, OMatrix, RowVector2, RowVector3,
    Vector3, Vector4,
};

/// Combines two images with shading
///
/// # Panics
/// If the images have different widths or heights
pub fn apply_shading(
    depth: &DepthImage,
    norm: &NormalImage,
    threads: Option<ThreadPool>,
) -> ColorImage {
    assert_eq!(depth.width(), norm.width());
    assert_eq!(depth.height(), norm.height());

    let mut out = ColorImage::new(depth.width(), depth.height());
    out.apply_effect(
        |x, y| {
            if depth[(y, x)] > 0 {
                shade_pixel(depth, norm, x, y)
            } else {
                [0u8; 3]
            }
        },
        threads,
    );
    out
}

/// Computes SSAO occlusion at each pixel in an image
///
/// # Panics
/// If the images have different widths or heights
pub fn compute_ssao(
    depth: &DepthImage,
    norm: &NormalImage,
    threads: Option<ThreadPool>,
) -> Image<f32> {
    // TODO make an object that is a bound Depth + Normal image with conditions
    // already checked, maybe GBuffer?
    assert_eq!(depth.width(), norm.width());
    assert_eq!(depth.height(), norm.height());
    let (ssao_kernel, ssao_noise) = ssao_kernel(32);

    let mut out = Image::<f32>::new(depth.width(), depth.height());
    out.apply_effect(
        |x, y| {
            if depth[(y, x)] > 0 {
                compute_pixel_ssao(depth, norm, x, y, &ssao_kernel, &ssao_noise)
            } else {
                f32::NAN
            }
        },
        threads,
    );

    out
}

/// Blurs the given image
pub fn compute_blur(
    ssao: &Image<f32>,
    threads: Option<ThreadPool>,
) -> Image<f32> {
    let mut out = Image::<f32>::new(ssao.width(), ssao.height());
    out.apply_effect(
        |x, y| {
            if ssao[(y, x)].is_nan() {
                f32::NAN
            } else {
                compute_pixel_blur(ssao, x, y)
            }
        },
        threads,
    );
    out
}

/// Compute shading for a single pixel
fn shade_pixel(
    depth: &DepthImage,
    norm: &NormalImage,
    x: usize,
    y: usize,
) -> [u8; 3] {
    let [nx, ny, nz] = norm[(y, x)];
    let n = Vector3::new(nx, ny, nz).normalize();

    // Convert from pixel to world coordinates
    // XXX we're missing the actual depth scale
    let p = Vector3::new(
        2.0 * (x as f32 / depth.width() as f32 - 0.5),
        2.0 * (y as f32 / depth.height() as f32 - 0.5),
        2.0 * (depth[(y, x)] as f32 / depth.width() as f32 - 0.5),
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

    accum = accum.clamp(0.0, 1.0);

    let c = (accum * 255.0) as u8;
    [c, c, c]
}

/// Compute an SSAO shading factor for a single pixel
///
/// Returns NAN if the pixel is empty (i.e. its depth is 0)
fn compute_pixel_ssao(
    depth: &DepthImage,
    norm: &NormalImage,
    x: usize,
    y: usize,
    kernel: &OMatrix<f32, nalgebra::Dyn, Const<3>>,
    noise: &OMatrix<f32, nalgebra::Dyn, Const<2>>,
) -> f32 {
    let pos = (y, x);
    let [nx, ny, nz] = norm[pos];
    let d = depth[pos];

    if d == 0 {
        return f32::NAN;
    }

    // XXX this is guessing at the depth
    let p = Vector3::new(
        (((x as f32) / depth.width() as f32) + 0.5) * 2.0,
        (((y as f32) / depth.height() as f32) + 0.5) * 2.0,
        (((d as f32) / depth.width() as f32) + 0.5) * 2.0,
    );

    // Get normal from the image
    let n = Vector3::new(nx, ny, nz).normalize();

    // Get a rotation vector based on pixel index, and add a Z coordinate
    let idx = pos.0 + pos.1 * depth.width();
    let rvec = noise.row(idx % noise.nrows()).transpose();
    let rvec = Vector3::new(rvec.x, rvec.y, 0.0);

    // Build our transform matrix, using the Gram-Schmidt process
    let tangent = (rvec - n * rvec.dot(&n)).normalize();
    let bitangent = n.cross(&tangent);
    let tbn = Matrix3::from_columns(&[tangent, bitangent, n]);

    const RADIUS: f32 = 0.1;
    let mut occlusion = 0.0;
    for i in 0..kernel.nrows() {
        let sample_pos = tbn * kernel.row(i).transpose() * RADIUS + p;
        // XXX this distorts samples for non-square images
        let px = ((sample_pos.x / 2.0) + 0.5) * depth.width() as f32;
        let py = ((sample_pos.y / 2.0) + 0.5) * depth.height() as f32;

        // Get depth from heightmap
        let actual_h = if px < depth.width() as f32
            && py < depth.height() as f32
            && px > 0.0
            && py > 0.0
        {
            depth[(py as usize, px as usize)]
        } else {
            0
        };

        // XXX same caveat about image depth here
        let actual_z = (((actual_h as f32) / depth.width() as f32) + 0.5) * 2.0;

        let dz = sample_pos.z - actual_z;
        if dz < RADIUS {
            occlusion += (sample_pos.z <= actual_z) as usize as f32;
        } else if dz < RADIUS * 2.0 && sample_pos.z <= actual_z {
            occlusion += ((RADIUS - (dz - RADIUS)) / RADIUS).powi(2);
        }
    }
    1.0 - (occlusion / kernel.nrows() as f32)
}

/// Computes a single blurred pixel
fn compute_pixel_blur(ssao: &Image<f32>, x: usize, y: usize) -> f32 {
    let mut best = 1000000.0;
    let mut value = 0.0;
    const BLUR_RADIUS: isize = 2;
    let x = x as isize;
    let y = y as isize;
    let mut run = |xmin, ymin| {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..=BLUR_RADIUS {
            for j in 0..=BLUR_RADIUS {
                let tx = x + xmin + i;
                let ty = y + ymin + j;
                if tx >= 0
                    && ty >= 0
                    && (tx as usize) < ssao.width()
                    && (ty as usize) < ssao.height()
                {
                    let s = ssao[(ty as usize, tx as usize)];
                    if !s.is_nan() {
                        sum += s;
                        count += 1;
                    }
                }
            }
        }
        let mean = sum / count as f32;
        let mut stdev = 0.0;
        for i in 0..=BLUR_RADIUS {
            for j in 0..=BLUR_RADIUS {
                let tx = x + xmin + i;
                let ty = y + ymin + j;
                if tx >= 0
                    && ty >= 0
                    && (tx as usize) < ssao.width()
                    && (ty as usize) < ssao.height()
                {
                    let s = ssao[(ty as usize, tx as usize)];
                    if !s.is_nan() {
                        stdev += (mean - s).powi(2);
                    }
                }
            }
        }
        let stdev = ((stdev / count as f32) - 1.0).sqrt();
        if stdev < best {
            best = stdev;
            value = mean;
        }
    };
    run(0, 0);
    run(-BLUR_RADIUS, 0);
    run(0, -BLUR_RADIUS);
    run(-BLUR_RADIUS, -BLUR_RADIUS);

    value
}

/// Returns a tuple of `(kernel, noise)` matrices for SSAO
///
/// - `kernel` is a set of points in a hemisphere, used for sampling the depth
///   buffer; it should be oriented based on surface normal
/// - `noise` is a list of random XY rotations, which can be applied to the
///   kernel to reduce banding
pub fn ssao_kernel(
    n: usize,
) -> (
    OMatrix<f32, nalgebra::Dyn, Const<3>>,
    OMatrix<f32, nalgebra::Dyn, Const<2>>,
) {
    // Based on http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
    use rand::prelude::*;

    let mut kernel = MatrixXx3::<f32>::zeros(n);
    let mut noise = MatrixXx2::<f32>::zeros(n);
    let mut rng = thread_rng();
    let xy_range = rand::distributions::Uniform::new_inclusive(-1.0, 1.0);
    let z_range = rand::distributions::Uniform::new_inclusive(0.0, 1.0);

    for i in 0..64 {
        let row = RowVector3::<f32>::new(
            rng.sample(xy_range),
            rng.sample(xy_range),
            rng.sample(z_range),
        );
        // Scale to keep most samples near the center
        let scale =
            ((i as f32) / (kernel.nrows() as f32 - 1.0)).powi(2) * 0.9 + 0.1;
        kernel.set_row(i, &(row * scale / row.norm()));

        let row =
            RowVector2::<f32>::new(rng.sample(xy_range), rng.sample(xy_range));
        noise.set_row(i, &(row / row.norm()));
    }
    (kernel, noise)
}
