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
//! Output is stored in the [`MergeBuffers`] object, and may be accessed with
//! [`output_distance`](MergeBuffers::output_distance) and
//! [`output_color`](MergeBuffers::output_color).
//! Note that if color has not been computed, `output_color` will return `None`.
use crate::{
    CopyVarsError, Gpu, RegPipeline,
    buf::{BufferSizeError, FlexBuffer, buffer_ro, buffer_rw, buffer_uniform},
    color::{ColorWorkspace as GenericColorWorkspace, ShapeColorBuffers},
    pixel::{PixelBufferTag, RawDistancePixel},
    shaders, tag,
};
use fidget_core::{render::ImageSize, shape::ShapeVars};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub use crate::voxel::effects::ColorError;

const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

/// Workspace for evaluating color expressions
pub type ColorWorkspace = GenericColorWorkspace<ColorConfig>;

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
    pub PixelDistanceBufferTag,
    RawDistancePixel,
    ImageSize,
    STORAGE | COPY_SRC,
    "Buffer tag for on-GPU distance images"
);

tag!(
    pub PixelColorBufferTag,
    u32, // also doubles as shape index, but not in user-visible APIs
    ImageSize,
    STORAGE | COPY_SRC,
    "Buffer tag for on-GPU color images"
);

/// Handle to a set of buffers used when merging images
pub struct MergeBuffers {
    config: wgpu::Buffer,
    distance: FlexBuffer<PixelDistanceBufferTag>,
    color: FlexBuffer<PixelColorBufferTag>,

    /// Number of images merged together
    image_count: usize,

    /// Indicates whether the `color` buffer represents color
    ///
    /// When this is `false`, the `color` buffer represents shape index instead,
    /// and [`output_color`](Self::output_color) returns `None`.
    has_color: bool,
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

    /// Returns `true` if the output color buffer is valid
    pub fn has_color(&self) -> bool {
        self.has_color
    }

    /// Returns a handle to the distance output buffer
    pub fn output_distance(&self) -> &FlexBuffer<PixelDistanceBufferTag> {
        &self.distance
    }

    /// Returns a handle to the color output buffer, if populated
    pub fn output_color(&self) -> Option<&FlexBuffer<PixelColorBufferTag>> {
        if self.has_color {
            Some(&self.color)
        } else {
            None
        }
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
                    buffer_ro(1), // image
                    buffer_rw(2), // distance
                    buffer_rw(3), // color (used here as an index)
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

    /// Submits a set of merge operations to accumulate a single image
    ///
    /// [`MergeBuffers::reset`] should be called before the first call to
    /// `submit_merge`. For the first merge after a reset, the output buffer is
    /// resized to fit the images; subsequent merges must be of the same size.
    pub fn submit_merge(
        &self,
        image: &FlexBuffer<PixelBufferTag>,
        remove_nans: bool,
        buf: &mut MergeBuffers,
    ) -> Result<(), MergeError> {
        let size = image.size();
        if buf.image_count > 0 {
            let buf_size = buf.distance.size();
            if size != buf_size {
                return Err(ImageSizeMismatch {
                    expected: size,
                    actual: buf_size,
                }
                .into());
            }
        } else {
            buf.distance
                .grow_to_fit(&self.gpu.device, size)
                .map_err(MergeError::OutputSize)?;
            buf.color
                .grow_to_fit(&self.gpu.device, size)
                .map_err(MergeError::OutputSize)?;
        }
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
            let cfg = MergeConfig {
                image_size: [size.width(), size.height()],
                remove_nans: remove_nans as u32,
                index_base: buf.image_count as u32,
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

            let bg =
                self.gpu
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("merge bind group"),
                        layout: &self.merge_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf.config.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: image.bind_active(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buf.distance.bind_active(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buf.color.bind_active(),
                            },
                        ],
                    });
            compute_pass.set_bind_group(0, Some(&bg), &[]);
            compute_pass.dispatch_workgroups(
                size.width().div_ceil(8),
                size.height().div_ceil(8),
                1,
            );
            buf.image_count += 1;
        }
        self.gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Builds a new set of [`MergeBuffers`] for the given image size
    pub fn merge_buffers(&self) -> MergeBuffers {
        let config = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<MergeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let distance = FlexBuffer::new(
            &self.gpu.device,
            "pixel merge distance".to_owned(),
            64.into(),
        )
        .unwrap();
        let color = FlexBuffer::new(
            &self.gpu.device,
            "pixel merge color".to_owned(),
            64.into(),
        )
        .unwrap();
        MergeBuffers {
            config,
            distance,
            color,
            image_count: 0,
            has_color: false,
        }
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
        shape: &ShapeColorBuffers,
        bufs: &mut ColorWorkspace,
    ) -> Result<(), ColorError> {
        self.submit_color_with_vars(
            merge,
            settings,
            shape,
            bufs,
            &Default::default(),
        )
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
        shape: &ShapeColorBuffers,
        bufs: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
    ) -> Result<(), ColorError> {
        self.color_ctx
            .submit(merge, settings, shape, bufs, vars, &self.gpu)
    }

    /// Returns a new workspace for color evaluation
    pub fn color_workspace(&self) -> ColorWorkspace {
        ColorWorkspace::new(&self.gpu.device)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Configuration for the color evaluation pass
///
/// This is quietly public because it's part of a public API, but is unlikely to
/// be useful for end-users of the library.
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
    _pad: [u32; 3],
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
                entries: &[
                    buffer_ro(0), // distance
                    buffer_rw(1), // color
                ],
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
        shape: &ShapeColorBuffers,
        bufs: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
        gpu: &Gpu,
    ) -> Result<(), ColorError> {
        if image.image_count != shape.shape_count() {
            return Err(ColorError::BadShapeCount {
                merge_count: image.image_count,
                shape_count: shape.shape_count(),
            });
        } else if image.has_color {
            return Err(ColorError::AlreadyHasColor);
        }
        let size = image.distance.size();
        let mat = settings.world_to_model
            * ImageSize::new(size.width(), size.height()).screen_to_world();
        let mut mat4 = nalgebra::Matrix4x3::<f32>::identity();
        mat4.fixed_view_mut::<3, 3>(0, 0).copy_from(&mat);

        bufs.copy_vars(gpu, shape.var_map(), vars)
            .map_err(|e| match e {
                CopyVarsError::BufferSize(b) => ColorError::VarBufferSize(b),
                CopyVarsError::MissingVar(v) => ColorError::MissingVar(v),
            })?;

        bufs.copy_tape(gpu, shape.bytecode())
            .map_err(ColorError::ConfigBufferSize)?;
        bufs.copy_shape_starts(gpu, shape.shape_start())
            .expect("shape starts should always fit if shape bytecode fits");

        // We'll write the config last, because writing the tape could have
        // invalidated it.
        let config = ColorConfig {
            mat: mat4.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            image_size: [size.width(), size.height()],
            only_filled: settings.only_filled.into(),
            _pad: [0; 3],
            z: settings.z,
        };
        bufs.copy_config(gpu, &config);

        let config_bg =
            bufs.config_bind_group(&gpu.device, &self.config_bind_group_layout);

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
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: image.distance.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: image.color.bind_active(),
                        },
                    ],
                });
            compute_pass.set_bind_group(1, &image_bg, &[]);
            compute_pass
                .set_pipeline(self.color_pipeline.get(shape.reg_count()));
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
