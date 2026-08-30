//! Pixel post-processing pipelines
//!
//! There is a notable difference between voxel and pixel post-processing:
//! instead of producing a fully-populated RGBA image to be drawn to the screen,
//! we produce two images: one containing [`RawDistancePixel`] values, and one
//! containing RGBA values (as 4-byte words).
//!
//! This is because – when the final shader draws to the screen – we can improve
//! visual fidelity by interpolating between distance values (typically by the
//! texture unit).  If we baked the final RGBA image, then drawing it at a
//! different scale (e.g. the user has zoomed and we're waiting for the next
//! image to be completed) would simply be blurry; with distance interpolation,
//! it remains sharper (though not pixel-perfect).
//!
//! Output is stored in the [`MergeBuffers`] object, which wraps a single GPU
//! buffer with back-to-back distance and color images.  Note that if color has
//! not been computed, then the second image instead stores merged shape index,
//! (which is not particularly meaningful to users).
use crate::{
    Gpu, RegPipeline, ShapeColor, ShapeColorBuffers, ShapeColorError,
    buf::{
        ArrayBuffer, BufferSizeError, ImageBuffer, buffer_ro, buffer_rw,
        buffer_uniform,
    },
    pixel::{PixelBufferTag, RawDistancePixel},
    shaders, tag,
};
use fidget_core::{render::ImageSize, shape::ShapeVars, vm::VmShape};
use fidget_raster::RgbaImage;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub use crate::voxel::effects::ColorError;

const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

/// Returns a shader for merging images
fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + shaders::COMMON + super::DISTANCE_PIXEL_SHADER
}

fn color_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code
        + COLOR_SHADER
        + super::TRANSFORM_INPUT
        + super::DISTANCE_PIXEL_SHADER
        + shaders::COMMON
        + shaders::TAPE_INTERPRETER
        + shaders::DUMMY_STACK
        + shaders::FLOAT_OPS
}

////////////////////////////////////////////////////////////////////////////////

/// Type indicating an image size mismatch
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {}, got {} × {}",
    expected.width(), expected.height(),
    actual.width(), actual.height(),
)]
pub struct ImageSizeMismatch {
    expected: ImageSize,
    actual: ImageSize,
}

/// Error returned when submitting a merge operation
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// Image sizes in the slice are not consistent
    #[error(transparent)]
    ImageSizeMismatch(#[from] ImageSizeMismatch),

    /// An error occurred while resizing the output buffer
    #[error(transparent)]
    OutputSize(BufferSizeError),
}

////////////////////////////////////////////////////////////////////////////////

tag!(
    MergedPixelBufferTag,
    [u32; 2], // This is a hack; we store two images side by side
    STORAGE | COPY_SRC,
    "Buffer tag for on-GPU merged images"
);

/// Handle to a set of buffers used when merging images
pub struct MergeBuffers {
    config: wgpu::Buffer,

    /// Stores two images back-to-back
    ///
    /// The first image is [`RawDistancePixel`] data; the second is initially
    /// the shape index then is rewritten to be color.
    out: ImageBuffer<MergedPixelBufferTag>,

    /// Set to `true` if the second half of the output buffer represents color
    has_color: bool,

    /// Number of images merged together
    image_count: usize,
}

impl MergeBuffers {
    /// Resets the merge buffer
    ///
    /// The next call to [`Context::submit_merge`] will clear the buffer and
    /// begin accumulating from scratch.
    pub fn reset(&mut self) {
        self.has_color = false;
        self.image_count = 0;
    }
}

/// Mirror of the WGSL `MergeConfig` object
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
pub(crate) struct MergeConfig {
    /// Image size, in pixels
    pub image_size: [u32; 2],

    /// Whether or not convert NaN values to distance values
    pub remove_nans: u32,

    /// Offset applied to indices when merging
    ///
    /// When this is 0, we initialize the output image
    pub index_base: u32,

    /// Number of valid image buffers (0-7)
    pub image_count: u32,

    // padding to the nearest multiple of 8
    pub _pad: u32,
}

////////////////////////////////////////////////////////////////////////////////

/// WGPU context for applying various effects
pub struct Context {
    gpu: Gpu,

    merge_bind_group_layout: wgpu::BindGroupLayout,
    merge_pipeline: wgpu::ComputePipeline,

    color_ctx: ColorContext,
}

impl Context {
    /// Builds a new context for applying effects
    pub fn new(gpu: &Gpu) -> Self {
        let merge_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_uniform(0),
                    buffer_ro(1), // image0
                    buffer_ro(2), // image1
                    buffer_ro(3), // image2
                    buffer_ro(4), // image3
                    buffer_ro(5), // image4
                    buffer_ro(6), // image5
                    buffer_ro(7), // image6
                    buffer_rw(8), // out
                ],
            },
        );
        let shader_code = merge_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects merge pipeline"),
                bind_group_layouts: &[Some(&merge_bind_group_layout)],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects merge shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let merge_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects merge compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let color_ctx = ColorContext::new(&gpu.device);

        Self {
            gpu: gpu.clone(),
            merge_bind_group_layout,
            merge_pipeline,
            color_ctx,
        }
    }

    /// Submits a set of merge operations to combine all of the images
    ///
    /// The output buffer is resized to fit the images
    ///
    /// If the incoming slice is empty, then no work is submitted
    pub fn submit_merge(
        &self,
        images: &[&ImageBuffer<PixelBufferTag>],
        remove_nans: bool,
        buf: &mut MergeBuffers,
    ) -> Result<(), MergeError> {
        let Some(size) = images.first().map(|i| i.size()) else {
            return Ok(());
        };
        for i in &images[1..] {
            let actual = i.size();
            if actual != size {
                return Err(ImageSizeMismatch {
                    expected: size,
                    actual,
                }
                .into());
            }
        }
        if buf.image_count > 0 && size != buf.out.size() {
            return Err(ImageSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }
            .into());
        }

        buf.out
            .grow_to_fit(&self.gpu.device, size)
            .map_err(MergeError::OutputSize)?;
        buf.has_color = false;
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("merge compute encoder"),
            },
        );
        // Scope to bound the lifetime of compute_pass
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("merge compute pass"),
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.merge_pipeline);
            for chunk in images.chunks(7) {
                let cfg = MergeConfig {
                    image_size: [size.width(), size.height()],
                    remove_nans: remove_nans as u32,
                    index_base: buf.image_count as u32,
                    image_count: chunk.len() as u32,
                    _pad: 0,
                };
                {
                    let mut writer = self
                        .gpu
                        .queue
                        .write_buffer_with(
                            &buf.config,
                            0,
                            (std::mem::size_of::<MergeConfig>() as u64)
                                .try_into()
                                .unwrap(),
                        )
                        .unwrap();
                    writer.copy_from_slice(cfg.as_bytes());
                }
                let image_bind = |i| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: chunk
                        .get(i)
                        .unwrap_or_else(|| chunk.first().unwrap())
                        .bind_active(),
                };

                let bg = self.gpu.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("merge bind group"),
                        layout: &self.merge_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf.config.as_entire_binding(),
                            },
                            image_bind(0),
                            image_bind(1),
                            image_bind(2),
                            image_bind(3),
                            image_bind(4),
                            image_bind(5),
                            image_bind(6),
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: buf.out.bind_active(),
                            },
                        ],
                    },
                );
                compute_pass.set_bind_group(0, Some(&bg), &[]);
                compute_pass.dispatch_workgroups(
                    size.width().div_ceil(8),
                    size.height().div_ceil(8),
                    1,
                );
                buf.image_count += chunk.len();
            }
        }
        self.gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Builds a new set of [`MergeBuffers`] for the given image size
    pub fn merge_buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<MergeBuffers, BufferSizeError> {
        let config = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<MergeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = ImageBuffer::new(
            &self.gpu.device,
            "merge output".to_owned(),
            image_size,
        )?;
        Ok(MergeBuffers {
            config,
            out,
            image_count: 0,
            has_color: false,
        })
    }

    /// Submits a color evaluation pass
    ///
    /// Image size is set from the `MergeBuffers`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color(
        &self,
        merge: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &ShapeColorBuffers<ColorConfig>,
    ) -> Result<(), ColorError> {
        self.submit_color_with_vars(merge, settings, shape, &Default::default())
    }

    /// Submits a color evaluation pass with auxiliary variables
    ///
    /// Image size is set from the `MergeBuffers`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color_with_vars(
        &self,
        merge: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &ShapeColorBuffers<ColorConfig>,
        vars: &ShapeVars<f32>,
    ) -> Result<(), ColorError> {
        self.color_ctx
            .submit(merge, settings, shape, vars, &self.gpu)
    }

    /// Returns an [`ImageReadBuffer`] to read from a [`MergeBuffers`] object
    pub fn image_buffer(&self) -> ImageReadBuffer {
        ImageReadBuffer::new(&self.gpu.device, "image".to_owned())
    }

    /// Copies image data and maps a CPU-readable image buffer
    pub fn map_image<'a>(
        &self,
        image_in: &'a MergeBuffers,
        image_out: &'a mut ImageReadBuffer,
    ) -> MappedImage<'a> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        image_out
            .grow_to_fit(&self.gpu.device, image_in.out.size())
            .expect(
                "image_in.out.size() should always be \
                a valid size for ImageReadBuffer::grow_to_fit",
            );
        encoder.copy_buffer_to_buffer(
            image_in.out.data(),
            0,
            image_out.buffer.data(),
            0,
            image_in.out.size_bytes(),
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = image_out.buffer.map_async(|_| {});
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        MappedImage {
            image: image_out,
            has_color: image_in.has_color,
            slice,
        }
    }

    /// Build a set of buffers for doing shape color evaluation
    pub fn color_buffers(
        &self,
        colors: &[ShapeColor<VmShape>],
    ) -> Result<ShapeColorBuffers<ColorConfig>, ShapeColorError> {
        ShapeColorBuffers::new(colors, &self.gpu.device)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Buffer for reading data back from the GPU
///
/// This object is constructed by [`Context::image_buffer`] and may only be used
/// with that particular [`Context`].
///
/// Once mapped, this is wrapped by a [`MappedImage`]
pub struct ImageReadBuffer {
    /// Image render size
    image_size: ImageSize,

    /// Result buffer that can be read back from the CPU
    buffer: ImageReadArrayBuffer,
}

impl ImageReadBuffer {
    fn new(device: &wgpu::Device, name: String) -> Self {
        let image_size = 64.into();
        Self {
            image_size,
            buffer: ImageReadArrayBuffer::new(
                device,
                name,
                image_size.width() as usize * image_size.height() as usize * 2,
            )
            .expect("64 should always be a valid size"),
        }
    }

    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        image_size: ImageSize,
    ) -> Result<(), BufferSizeError> {
        self.image_size = image_size;
        self.buffer.grow_to_fit(
            device,
            image_size.width() as usize * image_size.height() as usize * 2,
        )
    }
}

tag!(ImageReadTag, u32, COPY_DST | MAP_READ);
type ImageReadArrayBuffer = ArrayBuffer<ImageReadTag>;

/// Handle to a mapped image, which unmaps the image when dropped
pub struct MappedImage<'a> {
    image: &'a ImageReadBuffer,
    slice: wgpu::BufferSlice<'a>,

    /// Set to `true` if the color portion of the buffer is valid
    has_color: bool,
}

impl Drop for MappedImage<'_> {
    fn drop(&mut self) {
        self.image.buffer.data().unmap();
    }
}

impl MappedImage<'_> {
    /// Returns the image's distance data
    pub fn distance(&self) -> super::Image {
        // Get the pixel-populated image
        let result = <[RawDistancePixel]>::ref_from_bytes(
            &self.slice.get_mapped_range()[..self.image_bytes()],
        )
        .unwrap()
        .to_owned();
        super::Image::build(result, self.image.image_size).unwrap()
    }

    /// Returns the image's color data
    pub fn color(&self) -> Option<RgbaImage> {
        if self.has_color {
            // Get the pixel-populated image
            let result = <[[u8; 4]]>::ref_from_bytes(
                &self.slice.get_mapped_range()[self.image_bytes()..],
            )
            .unwrap()
            .to_owned();
            Some(RgbaImage::build(result, self.image.image_size).unwrap())
        } else {
            None
        }
    }

    fn image_bytes(&self) -> usize {
        (self.image.image_size.width() as usize)
            * (self.image.image_size.height() as usize)
            * std::mem::size_of::<RawDistancePixel>()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Configuration for the color evaluation pass
///
/// This is public because it's used in a public function signature, but it's
/// unlikely to be used by library end-users.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct ColorConfig {
    /// Screen-to-model transform matrix (mat3x3)
    mat: [f32; 12],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    /// Z height at which to evaluate the colors
    z: f32,

    /// Image size (pixels)
    image_size: [u32; 2],

    ///When non-zero, only compute color for filled pixels
    only_filled: u32,

    // alignment
    _pad: u32,
    // this is followed by a tape_data flexible array member
}

/// Settings for per-pixel color evaluation
#[derive(Copy, Clone, Debug)]
pub struct ColorSettings {
    /// Z height at which to evaluate the colors
    pub z: f32,

    /// When non-zero, only compute color for filled pixels
    pub only_filled: bool,

    /// Transform matrix
    ///
    /// This should be the same transform matrix used to evaluate the original
    /// pixel image (unless you're doing something weird)
    pub world_to_model: nalgebra::Matrix3<f32>,
}

struct ColorContext {
    /// Configuration bind group layout (shape-specific, includes tape data)
    config_bind_group_layout: wgpu::BindGroupLayout,

    /// Image bind groups (input and output)
    image_bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for computing per-pixel color
    color_pipeline: RegPipeline,
}

impl ColorContext {
    pub fn new(device: &wgpu::Device) -> Self {
        let config_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("color config and shape"),
                entries: &[buffer_ro(0), buffer_ro(1), buffer_ro(2)],
            });
        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("color images"),
                entries: &[buffer_rw(0)],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("color pipeline"),
                bind_group_layouts: &[
                    Some(&config_bind_group_layout),
                    Some(&image_bind_group_layout),
                ],
                immediate_size: 0u32,
            });
        let color_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = color_shader(reg_count);
            let shader_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("color ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("color_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            config_bind_group_layout,
            image_bind_group_layout,
            color_pipeline,
        }
    }

    fn submit(
        &self,
        image: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &ShapeColorBuffers<ColorConfig>,
        vars: &ShapeVars<f32>,
        gpu: &Gpu,
    ) -> Result<(), ColorError> {
        if image.image_count != shape.shape_count {
            return Err(ColorError::BadShapeCount {
                merge_count: image.image_count,
                shape_count: shape.shape_count,
            });
        }
        let size = image.out.size();
        let mat = settings.world_to_model
            * ImageSize::new(size.width(), size.height()).screen_to_world();
        let mut mat4 = nalgebra::Matrix4x3::<f32>::identity();
        mat4.fixed_view_mut::<3, 3>(0, 0).copy_from(&mat);

        let config_bg = shape
            .config_bind_group(&gpu.device, &self.config_bind_group_layout);

        let config = ColorConfig {
            mat: mat4.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            image_size: [size.width(), size.height()],
            only_filled: settings.only_filled.into(),
            _pad: 0,
            z: settings.z,
        };
        shape.write_config(&config, &gpu.queue);
        shape.write_vars(vars, &gpu.queue)?;

        // Create a command encoder and dispatch the compute work
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_bind_group(0, config_bg, &[]);

            // TODO this creates a bind group for every evaluation, instead of
            // caching it somewhere.  However, *where* to cache it is not
            // obvious, because it combines fields from two different buffer
            // objects.
            let image_bg =
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("color image bind group"),
                    layout: &self.image_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: image.out.bind_active(),
                    }],
                });
            compute_pass.set_bind_group(1, &image_bg, &[]);
            compute_pass.set_pipeline(self.color_pipeline.get(shape.reg_count));
            compute_pass.dispatch_workgroups(
                size.width().div_ceil(8),
                size.height().div_ceil(8),
                1,
            );
        }
        gpu.queue.submit(Some(encoder.finish()));
        image.has_color = true;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_merge_shader() {
        crate::compile_shader(&merge_shader(), "merge");
    }

    #[test]
    fn compile_color_shader() {
        crate::compile_shader(&color_shader(16), "color");
    }

    #[test]
    fn merge_config_layout() {
        crate::test::compare_struct_layout::<MergeConfig>(
            &merge_shader(),
            "MergeConfig",
        );
    }

    #[test]
    fn color_config_layout() {
        crate::test::compare_struct_layout::<ColorConfig>(
            &color_shader(16),
            "Config",
        );
    }
}
