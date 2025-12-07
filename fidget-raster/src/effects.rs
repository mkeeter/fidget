//! Post-processing effects for rendered images

use super::{
    ColorImage, DistancePixel, GeometryBuffer, GeometryPixel, Image, ImageSize,
    ThreadPool,
};
use nalgebra::{
    Const, Matrix3, MatrixXx2, MatrixXx3, OMatrix, RowVector2, RowVector3,
    Vector3, Vector4,
};
use rand::prelude::*;

/// Denoise normals by replacing back-facing normals with their average neighbor
///
/// # Panics
/// If the images have different widths or heights
pub fn denoise_normals(
    image: &GeometryBuffer,
    threads: Option<&ThreadPool>,
) -> GeometryBuffer {
    let radius = 2;
    let mut out = GeometryBuffer::new(image.size());
    out.apply_effect(
        |x, y| {
            let depth = image[(y, x)].depth;
            let normal = if depth > 0.0 {
                denoise_pixel(image, x, y, radius)
            } else {
                [0.0; 3]
            };
            GeometryPixel { depth, normal }
        },
        threads,
    );
    out
}

/// Combines two images with shading
///
/// # Panics
/// If the images have different widths or heights
pub fn apply_shading(
    image: &GeometryBuffer,
    ssao: bool,
    threads: Option<&ThreadPool>,
) -> ColorImage {
    let ssao = if ssao {
        let ssao = compute_ssao(image, threads);
        Some(blur_ssao(&ssao, threads))
    } else {
        None
    };

    let size = image.size();
    let mut out = ColorImage::new(ImageSize::new(size.width(), size.height()));
    out.apply_effect(
        |x, y| {
            if image[(y, x)].depth > 0.0 {
                shade_pixel(image, ssao.as_ref(), x, y)
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
    image: &GeometryBuffer,
    threads: Option<&ThreadPool>,
) -> Image<f32> {
    let ssao_kernel = ssao_kernel(64);
    let ssao_noise = ssao_noise(16 * 16);

    let size = image.size();
    let mut out =
        Image::<f32>::new(ImageSize::new(size.width(), size.height()));
    out.apply_effect(
        |x, y| {
            if image[(y, x)].depth > 0.0 {
                compute_pixel_ssao(image, x, y, &ssao_kernel, &ssao_noise)
            } else {
                f32::NAN
            }
        },
        threads,
    );

    out
}

/// Blurs the given image, which is expected to be an SSAO occlusion map
pub fn blur_ssao(
    ssao: &Image<f32>,
    threads: Option<&ThreadPool>,
) -> Image<f32> {
    let mut out = Image::<f32>::new(ssao.size());
    let blur_radius = 2;
    out.apply_effect(
        |x, y| {
            if ssao[(y, x)].is_nan() {
                f32::NAN
            } else {
                compute_pixel_blur(ssao, x, y, blur_radius)
            }
        },
        threads,
    );
    out
}

/// Compute shading for a single pixel
fn shade_pixel(
    image: &GeometryBuffer,
    ssao: Option<&Image<f32>>,
    x: usize,
    y: usize,
) -> [u8; 3] {
    let [nx, ny, nz] = image[(y, x)].normal;
    let n = Vector3::new(nx, ny, nz).normalize();

    // Convert from pixel to world coordinates
    let p = Vector3::new(
        2.0 * (x as f32 / image.width() as f32 - 0.5),
        2.0 * (y as f32 / image.height() as f32 - 0.5),
        2.0 * (image[(y, x)].depth / image.depth() as f32 - 0.5),
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
    if let Some(ssao) = ssao {
        accum *= ssao[(y, x)] * 0.6 + 0.4;
    }

    accum = accum.clamp(0.0, 1.0);

    let c = (accum * 255.0) as u8;
    [c, c, c]
}

/// Compute an SSAO shading factor for a single pixel
///
/// Returns NAN if the pixel is empty (i.e. its depth is 0)
fn compute_pixel_ssao(
    image: &GeometryBuffer,
    x: usize,
    y: usize,
    kernel: &OMatrix<f32, nalgebra::Dyn, Const<3>>,
    noise: &OMatrix<f32, nalgebra::Dyn, Const<2>>,
) -> f32 {
    let pos = (y, x);
    let GeometryPixel {
        normal: [nx, ny, nz],
        depth: d,
    } = image[pos];

    if d == 0.0 {
        return f32::NAN;
    }

    // XXX The implementation in libfive-cuda adds a 0.5 pixel offset
    let p = Vector3::new(
        (((x as f32) / image.width() as f32) - 0.5) * 2.0,
        (((y as f32) / image.height() as f32) - 0.5) * 2.0,
        ((d / image.depth() as f32) - 0.5) * 2.0,
    );

    // Get normal from the image
    let n = Vector3::new(nx, ny, nz).normalize();

    // Get a rotation vector based on pixel index, and add a Z coordinate
    let idx = pos.0 + pos.1 * image.width();
    let rvec = noise.row((idx * 19) % noise.nrows()).transpose();
    let rvec = Vector3::new(rvec.x, rvec.y, 0.0);

    // Build our transform matrix, using the Gram-Schmidt process
    let tangent = (rvec - n * rvec.dot(&n)).normalize();
    let bitangent = n.cross(&tangent);
    let tbn = Matrix3::from_columns(&[tangent, bitangent, n]);

    const RADIUS: f32 = 0.1;
    let mut occlusion = 0.0;
    for i in 0..kernel.nrows() {
        // position in world coordinates
        let sample_pos = tbn * kernel.row(i).transpose() * RADIUS + p;

        // convert to pixel coordinates
        // XXX this distorts samples for non-square images
        let px = ((sample_pos.x / 2.0) + 0.5) * image.width() as f32;
        let py = ((sample_pos.y / 2.0) + 0.5) * image.height() as f32;

        // Get depth from heightmap
        let actual_h = if px < image.width() as f32
            && py < image.height() as f32
            && px > 0.0
            && py > 0.0
        {
            image[(py as usize, px as usize)].depth
        } else {
            0.0
        };

        let actual_z = ((actual_h / image.depth() as f32) - 0.5) * 2.0;

        let dz = sample_pos.z - actual_z;
        if dz < RADIUS {
            occlusion += (sample_pos.z <= actual_z) as usize as f32;
        } else if dz < RADIUS * 2.0 && sample_pos.z <= actual_z {
            occlusion += ((RADIUS - (dz - RADIUS)) / RADIUS).powi(2);
        }
    }
    1.0 - (occlusion / kernel.nrows() as f32)
}

/// If the pixel has a back-facing normal, then pick a normal from neighbors
fn denoise_pixel(
    image: &GeometryBuffer,
    x: usize,
    y: usize,
    denoise_radius: isize,
) -> [f32; 3] {
    let n = image[(y, x)].normal;
    if n[2] > 0.0 {
        return n;
    }
    let x = x as isize;
    let y = y as isize;
    [
        (0, 0),
        (-denoise_radius, 0),
        (0, -denoise_radius),
        (-denoise_radius, -denoise_radius),
    ]
    .into_iter()
    .flat_map(|(xmin, ymin)| {
        let mut sum = Vector3::zeros();
        let mut count = 0;
        for i in 0..=denoise_radius {
            for j in 0..=denoise_radius {
                let tx = x + xmin + i;
                let ty = y + ymin + j;
                if tx >= 0
                    && ty >= 0
                    && (tx as usize) < image.width()
                    && (ty as usize) < image.height()
                {
                    let pos = (ty as usize, tx as usize);
                    if image[pos].depth != 0.0 {
                        let n = image[pos].normal;
                        sum += Vector3::new(n[0], n[1], n[2]);
                        count += 1;
                    }
                }
            }
        }
        if count == 0 {
            return None;
        }
        let mean = sum / count as f32;
        let mut score = 0.0;
        for i in 0..=denoise_radius {
            for j in 0..=denoise_radius {
                let tx = x + xmin + i;
                let ty = y + ymin + j;
                if tx >= 0
                    && ty >= 0
                    && (tx as usize) < image.width()
                    && (ty as usize) < image.height()
                {
                    let pos = (ty as usize, tx as usize);
                    if image[pos].depth != 0.0 {
                        let n = image[pos].normal;
                        score += Vector3::new(n[0], n[1], n[2]).dot(&mean);
                    }
                }
            }
        }
        Some((score, mean.into()))
    })
    .max_by_key(|(score, _mean)| ordered_float::OrderedFloat(*score))
    .unwrap_or((0.0, n))
    .1
}

/// Computes a single blurred pixel
fn compute_pixel_blur(
    ssao: &Image<f32>,
    x: usize,
    y: usize,
    blur_radius: isize,
) -> f32 {
    let x = x as isize;
    let y = y as isize;
    [
        (0, 0),
        (-blur_radius, 0),
        (0, -blur_radius),
        (-blur_radius, -blur_radius),
    ]
    .into_iter()
    .flat_map(|(xmin, ymin)| {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..=blur_radius {
            for j in 0..=blur_radius {
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
        if count == 0 {
            return None;
        }
        let mean = sum / count as f32;
        let mut stdev = 0.0;
        for i in 0..=blur_radius {
            for j in 0..=blur_radius {
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
        Some((stdev / count as f32, mean))
    })
    .min_by_key(|(stdev, _mean)| ordered_float::OrderedFloat(*stdev))
    .unwrap_or_else(|| (0.0, ssao[(y as usize, x as usize)]))
    .1
}

/// Returns a SSAO kernel matrix
///
/// The resulting matrix is a list of row vectors representing points in a
/// hemisphere with a maximum radius of 1.0, used for sampling the depth buffer.
///
/// It should be reoriented based on the surface normal.
pub fn ssao_kernel(n: usize) -> OMatrix<f32, nalgebra::Dyn, Const<3>> {
    // Based on http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
    use rand::prelude::*;

    let mut kernel = MatrixXx3::<f32>::zeros(n);
    let mut rng = rand::rng();
    let xy_range = rand::distr::Uniform::new_inclusive(-1.0, 1.0).unwrap();
    let z_range = rand::distr::Uniform::new_inclusive(0.0, 1.0).unwrap();

    for i in 0..n {
        let row = RowVector3::<f32>::new(
            rng.sample(xy_range),
            rng.sample(xy_range),
            rng.sample(z_range),
        );
        // Scale to keep most samples near the center
        let scale =
            ((i as f32) / (kernel.nrows() as f32 - 1.0)).powi(2) * 0.9 + 0.1;
        kernel.set_row(i, &(row * scale / row.norm()));
    }
    kernel
}

/// Returns an SSAO noise matrix
///
/// The noise matrix is a list of row vectors representing random XY rotations,
/// which can be applied to the kernel vectors to reduce banding.
pub fn ssao_noise(n: usize) -> OMatrix<f32, nalgebra::Dyn, Const<2>> {
    let mut noise = MatrixXx2::<f32>::zeros(n);
    let mut rng = rand::rng();
    let xy_range = rand::distr::Uniform::new_inclusive(-1.0, 1.0).unwrap();
    for i in 0..n {
        let row =
            RowVector2::<f32>::new(rng.sample(xy_range), rng.sample(xy_range));
        noise.set_row(i, &(row / row.norm()));
    }
    noise
}

/// Converts a [`DistancePixel`] image into an RGBA bitmap
///
/// Filled pixels are `[255u8; 4]`; empty pixels are `[0u8; 4]` (i.e.
/// transparent) if `transparent` is set or `[0, 0, 0, 255]` otherwise.
pub fn to_rgba_bitmap(
    image: Image<DistancePixel>,
    transparent: bool,
    threads: Option<&ThreadPool>,
) -> Image<[u8; 4]> {
    let mut out = Image::new(image.size());
    out.apply_effect(
        |x, y| {
            let p = image[(y, x)];
            if p.inside() {
                [255u8; 4]
            } else if transparent {
                [0u8; 4]
            } else {
                [0, 0, 0, 255]
            }
        },
        threads,
    );
    out
}

/// Converts a [`DistancePixel`] image into a debug visualization
pub fn to_debug_bitmap(
    image: Image<DistancePixel>,
    threads: Option<&ThreadPool>,
) -> Image<[u8; 4]> {
    let mut out = Image::new(image.size());
    out.apply_effect(
        |x, y| match image[(y, x)].distance() {
            Ok(v) => {
                if v < 0.0 {
                    [255u8; 4]
                } else {
                    [0, 0, 0, 255]
                }
            }
            // Debug color scheme for fill regions
            Err(f) => match (f.depth, f.inside) {
                (0, true) => [255, 0, 0, 255],
                (0, false) => [50, 0, 0, 255],
                (1, true) => [0, 255, 0, 255],
                (1, false) => [0, 50, 0, 255],
                (2, true) => [0, 0, 255, 255],
                (2, false) => [0, 0, 50, 255],
                (_, true) => [255, 255, 0, 255],
                (_, false) => [50, 50, 0, 255],
            },
        },
        threads,
    );
    out
}

/// Converts a [`DistancePixel`] image to a SDF rendering
///
/// The shading style is pervasive on Shadertoy, so I'm not sure where it
/// originated; I borrowed it from
/// [Inigo Quilez](https://www.shadertoy.com/view/t3X3z4)
///
/// `NAN` distance pixels are shown in pure red (`[255, 0, 0]`).
pub fn to_rgba_distance(
    image: Image<DistancePixel>,
    threads: Option<&ThreadPool>,
) -> Image<[u8; 4]> {
    let mut out = Image::new(image.size());
    out.apply_effect(
        |x, y| match image[(y, x)].distance() {
            Ok(f) if f.is_nan() => [255, 0, 0, 255],
            Ok(f) => {
                let r = 1.0 - 0.1f32.copysign(f);
                let g = 1.0 - 0.4f32.copysign(f);
                let b = 1.0 - 0.7f32.copysign(f);

                let dim = 1.0 - (-4.0 * f.abs()).exp(); // dimming near 0
                let bands = 0.8 + 0.2 * (140.0 * f).cos(); // banding

                let smoothstep = |edge0: f32, edge1: f32, x: f32| {
                    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };
                let mix = |x: f32, y: f32, a: f32| x * (1.0 - a) + y * a;

                let run = |v: f32| {
                    let mut v = v * dim * bands;
                    v = mix(v, 1.0, 1.0 - smoothstep(0.0, 0.015, f.abs()));
                    v = mix(v, 1.0, 1.0 - smoothstep(0.0, 0.005, f.abs()));
                    (v.clamp(0.0, 1.0) * 255.0) as u8
                };

                [run(r), run(g), run(b), 255]
            }
            // Pick a single orange / blue for fill regions
            Err(f) => {
                if f.inside {
                    [184, 235, 255, 255]
                } else {
                    [217, 144, 72, 255]
                }
            }
        },
        threads,
    );
    out
}
