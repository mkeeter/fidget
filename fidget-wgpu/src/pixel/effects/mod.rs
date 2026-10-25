//! Pixel post-processing pipelines
//!
//! There is a notable difference between voxel and pixel post-processing:
//! instead of producing a fully-populated RGBA image to be drawn to the screen,
//! we produce two images: one containing [`RawDistancePixel`] values, and one
//! containing RGBA values (as 4-byte words).
//!
//! This is because – when the final shader draws to the screen – we can improve
//! visual fidelity by interpolating between distance values (typically by
//! storing distance as a texture and getting interpolation for free). If we
//! baked the final RGBA image, then drawing it at a different scale (e.g. the
//! user has zoomed and we're waiting for the next image to be completed) would
//! simply be blurry; with distance interpolation, it remains sharper (though
//! not pixel-perfect).
//!
//! Output is stored in the [`MergeBuffers`] object, and may be accessed with
//! [`output_distance`](MergeBuffers::output_distance) and
//! [`output_color`](MergeBuffers::output_color). Note that if color has not
//! been computed, `output_color` will return `None`.
use crate::{
    ConfigWorkspace, Gpu, ImageSizeMismatch, RegPipeline,
    buf::BufferSizeError,
    buf::{FlexBuffer, buffer_ro, buffer_rw},
    color::RenderColor,
    pixel::RawDistancePixel,
    shaders, tag,
};
use fidget_core::{
    render::ImageSize,
    shape::{MissingVar, ShapeVars},
};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Error returned when submitting a color evaluation operation
#[derive(Debug, thiserror::Error)]
pub enum ColorError {
    /// A variable is missing from the map
    #[error(transparent)]
    MissingVar(#[from] MissingVar),

    /// Wrong number of shapes
    #[error(
        "merged image is built from {merge_count} shapes,
         but shape color buffer expected {shape_count} shapes"
    )]
    BadShapeCount {
        /// Number of shapes in the merged image
        merge_count: usize,
        /// Number of shapes in the color buffer
        shape_count: usize,
    },

    /// Shape color has already been populated
    #[error("shape color has already been populated")]
    AlreadyHasColor,
}

const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

/// Workspace for evaluating color expressions
pub type ColorWorkspace = ConfigWorkspace<ColorConfig>;

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

/// Error returned when submitting a merge operation
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// Image sizes in the slice are not consistent
    #[error(transparent)]
    ImageSizeMismatch(#[from] ImageSizeMismatch),
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

/// Handle to a workspace for image merge operations
pub type MergeWorkspace = ConfigWorkspace<MergeConfig>;

/// Handle to size-dependent buffers for merge operations
pub struct MergeBuffers {
    distance: FlexBuffer<PixelDistanceBufferTag>,
    color: FlexBuffer<PixelColorBufferTag>,

    /// Bind group to merge buffers
    merge_bind_group: wgpu::BindGroup,

    /// Bind group to do color evaluation
    color_bind_group: wgpu::BindGroup,

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
///
/// This is public because it's part of an exported generic object, but users
/// are not expected to use it directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct MergeConfig {
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

    /// Bind group layout for merge operations
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
                    buffer_rw(0), // distance
                    buffer_rw(1), // color
                ],
            },
        );

        let shader_code = merge_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects merge pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&merge_bind_group_layout),
                ],
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
                label: Some("pixel effects merge compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let color_ctx = ColorContext::new(gpu);

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
    /// `submit_merge`. The [`MergeBuffers`] object must match the image size;
    /// otherwise an error will be returned.
    pub fn submit_merge(
        &self,
        image: &super::Buffers,
        remove_nans: bool,
        workspace: &mut MergeWorkspace,
        buf: &mut MergeBuffers,
    ) -> Result<(), MergeError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("merge compute encoder"),
            },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode_merge(
            image,
            remove_nans,
            workspace,
            buf,
            &mut encoder,
            &mut staging,
        )?;
        staging.finish();
        self.gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    /// Low-level function to encode a merge operation into a command encoder
    pub fn encode_merge(
        &self,
        image: &super::Buffers,
        remove_nans: bool,
        workspace: &mut MergeWorkspace,
        buf: &mut MergeBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), MergeError> {
        let size = image.pixels.size();
        let buf_size = buf.distance.size();
        if size != buf_size {
            return Err(ImageSizeMismatch {
                expected: size,
                actual: buf_size,
            }
            .into());
        }
        buf.has_color = false;

        // Write the config using a staging buffer
        let cfg = MergeConfig {
            image_size: [size.width(), size.height()],
            remove_nans: remove_nans as u32,
            index_base: buf.image_count as u32,
        };
        workspace.encode_copy(encoder, staging, &cfg);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.merge_pipeline);

        compute_pass.set_bind_group(0, Some(&workspace.bind_group), &[]);
        compute_pass.set_bind_group(1, Some(&image.bind_groups.output), &[]);
        compute_pass.set_bind_group(2, Some(&buf.merge_bind_group), &[]);
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.image_count += 1;
        Ok(())
    }

    /// Builds a new set of [`MergeWorkspace`] for the given image size
    pub fn merge_workspace(&self) -> MergeWorkspace {
        self.gpu.config_workspace("merge")
    }

    /// Build a new set of buffers for merging pixel images
    pub fn merge_buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<MergeBuffers, BufferSizeError> {
        let distance =
            FlexBuffer::new(&self.gpu.device, "merge distance", image_size)?;
        let color =
            FlexBuffer::new(&self.gpu.device, "merge color", image_size)?;
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: distance.bind_active(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: color.bind_active(),
            },
        ];
        let merge_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("merge bind group"),
                    layout: &self.merge_bind_group_layout,
                    entries,
                });
        let color_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("color bind group"),
                    layout: &self.color_ctx.image_bind_group_layout,
                    entries,
                });
        Ok(MergeBuffers {
            distance,
            color,
            merge_bind_group,
            color_bind_group,
            image_count: 0,
            has_color: false,
        })
    }

    /// Submits a color evaluation pass
    ///
    /// Image size is set from the [`MergeBuffers`]; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color(
        &self,
        merge: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &RenderColor,
        workspace: &mut ColorWorkspace,
    ) -> Result<(), ColorError> {
        self.submit_color_with_vars(
            merge,
            settings,
            shape,
            workspace,
            &Default::default(),
        )
    }

    /// Submits a color evaluation pass with auxiliary variables
    ///
    /// Image size is set from the `MergeWorkspace`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color_with_vars(
        &self,
        merge: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &RenderColor,
        workspace: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
    ) -> Result<(), ColorError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode_color(
            merge,
            settings,
            shape,
            workspace,
            vars,
            &mut encoder,
            &mut staging,
        )?;
        staging.finish();
        self.gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    /// Low-level function to encode a color rendering pass
    #[allow(clippy::too_many_arguments)]
    pub fn encode_color(
        &self,
        merge: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &RenderColor,
        workspace: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), ColorError> {
        self.color_ctx
            .encode(merge, settings, shape, workspace, vars, encoder, staging)
    }

    /// Returns a new workspace for color evaluation
    pub fn color_workspace(&self) -> ColorWorkspace {
        self.gpu.config_workspace("color")
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

    _pad: u32,
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
    /// Image bind groups (input and output)
    image_bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for computing per-pixel color
    color_pipeline: RegPipeline,
}

impl ColorContext {
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
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
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.color_bind_group_layout),
                    Some(&image_bind_group_layout),
                ],
                immediate_size: 0u32,
            });
        let device_ = device.clone();
        let color_pipeline = RegPipeline::build(Box::new(move |reg_count| {
            let shader_code = color_shader(reg_count);
            let shader_module =
                device_.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device_.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("color ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("color_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }));

        Self {
            image_bind_group_layout,
            color_pipeline,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        image: &mut MergeBuffers,
        settings: ColorSettings,
        shape: &RenderColor,
        workspace: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
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
        shape.copy_vars(encoder, staging, vars)?;

        let config = ColorConfig {
            mat: mat4.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            image_size: [size.width(), size.height()],
            only_filled: settings.only_filled.into(),
            z: settings.z,
            _pad: 0u32,
        };
        workspace.encode_copy(encoder, staging, &config);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, shape.bind_group(), &[]);
        compute_pass.set_bind_group(2, &image.color_bind_group, &[]);
        compute_pass.set_pipeline(self.color_pipeline.get(shape.reg_count()));
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
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
