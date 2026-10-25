//! Voxel post-processing effects
//!
//! These effects let us set up a simple rendering pipeline:
//!
//! - Start with [`GeometryPixel`](fidget_raster::voxel::GeometryPixel) buffers
//!   (16 bytes per pixel, stored on the GPU).
//! - Merge and denoise a set of buffers into a single image containing
//!   [`PackedVoxel`] data (normals, depth, and source image index packed into 8
//!   bytes per pixel).
//! - Apply shading to a [`PackedVoxel`] buffer, producing an RGBA image buffer

use crate::{
    ConfigWorkspace, CopyVarsError, Gpu, RegPipeline,
    buf::{BufferSizeError, FlexBuffer, buffer_ro, buffer_rw},
    color::ShapeColorBuffers,
    shaders, tag,
};
use fidget_core::{
    render::{ImageSize, VoxelSize},
    shape::{MissingVar, ShapeVars},
};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// WGPU context for applying various effects
pub struct Context {
    gpu: Gpu,

    merge_pipeline: wgpu::ComputePipeline,
    shade_pipeline: wgpu::ComputePipeline,
    heightmap_pipeline: wgpu::ComputePipeline,

    ssao_ctx: SsaoContext,

    color_ctx: ColorContext,
}

/// Workspace for evaluating color expressions
pub type ColorWorkspace = ConfigWorkspace<ColorConfig>;

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const SHADE_SHADER: &str = include_str!("shaders/shade.wgsl");
const HEIGHTMAP_SHADER: &str = include_str!("shaders/heightmap.wgsl");
const SSAO_SHADER: &str = include_str!("shaders/ssao.wgsl");
const BLUR_SHADER: &str = include_str!("shaders/blur.wgsl");
const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
}

fn shade_shader() -> String {
    SHADE_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
}

fn heightmap_shader() -> String {
    HEIGHTMAP_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
}

fn ssao_shader() -> String {
    SSAO_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
}

fn blur_shader() -> String {
    BLUR_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
}

fn color_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code
        + COLOR_SHADER
        + COMMON_SHADER
        + super::TRANSFORM_INPUT
        + shaders::COMMON
        + shaders::TAPE_INTERPRETER
        + shaders::DUMMY_STACK
        + shaders::FLOAT_OPS
}

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

/// Configuration for the color evaluation pass
///
/// This is quietly public because it's part of a public API, but is unlikely to
/// be useful for end-users of the library.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct ColorConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    // padding for alignment
    _pad1: u32,

    /// Image size (pixels)
    image_size: [u32; 2],

    // padding for alignment
    _pad2: [u32; 2],
}

/// GPU-side configuration object for merging images
///
/// This is public because it's used in a public type, but is not expected to be
/// used directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct MergeConfig {
    /// Image size, in pixels
    pub image_size: [u32; 2],

    /// Whether or not to denoise when merging (non-zero is true)
    pub denoise: u32,

    /// Scale to compensate for z-flattening
    pub z_scale: f32,

    /// Offset applied to indices when merging
    ///
    /// When this is 0, we initialize the output image
    pub index_base: u32,

    // Padding to next multiple of 8 bytes
    _pad: u32,
}

/// GPU-side configuration object for shading
///
/// This is public because it's used in a public type, but is not expected to be
/// used directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct ShadeConfig {
    /// Image size, in pixels
    image_size: [u32; 3],

    /// Flags to set the presence of color and SSAO
    flags: u32,
}

/// GPU-side configuration object for heightmap evaluation
///
/// This is public because it's used in a public type, but is not expected to be
/// used directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct HeightmapConfig {
    /// Image size, in pixels
    image_size: [u32; 3],

    /// Flag to indicate whether the color is valid
    has_color: u32,
}

/// Settings for image merging pass
#[derive(Copy, Clone, Debug)]
pub struct MergeSettings {
    /// Denoise normals (replacing backwards-facing normals with a neighbor)
    pub denoise: bool,

    /// Non-uniform Z scaling applied to the model
    ///
    /// The Z normal is *divided* by this value when merging images, to cancel
    /// out the effect of the scaling.
    pub z_scale: f32,
}

impl Default for MergeSettings {
    fn default() -> Self {
        Self {
            denoise: true,
            z_scale: 1.0,
        }
    }
}

/// Must match constants in `shade.wgsl`
const SHADE_CONFIG_HAS_SSAO: u32 = 1u32;
const SHADE_CONFIG_HAS_COLOR: u32 = 2u32;

tag!(pub MergeVoxelBufferTag, PackedVoxel, VoxelSize, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU merged ([`PackedVoxel`]) images"
);

/// Handle to workspace buffers used when merging images
pub type MergeWorkspace = ConfigWorkspace<MergeConfig>;

/// Handle to size-dependent buffers used when merging images
pub struct MergeBuffers {
    out: FlexBuffer<MergeVoxelBufferTag>,
    /// Bind group holding `out` as a read-only value
    ro_bind_group: wgpu::BindGroup,
    /// Bind group holding `out` as a read-write value
    rw_bind_group: wgpu::BindGroup,
    /// Stateful tracker for number of images accumulated
    image_count: usize,
}

impl MergeBuffers {
    /// Returns a handle to the output buffer
    pub fn output(&self) -> &FlexBuffer<MergeVoxelBufferTag> {
        &self.out
    }

    /// Resets the merge buffer
    ///
    /// The next call to [`Context::submit_merge`] will clear the buffer and
    /// begin accumulating from scratch.
    pub fn reset(&mut self) {
        self.image_count = 0;
    }
}

/// Handle to workspace buffers used when rendering a heightmap
pub type HeightmapWorkspace = ConfigWorkspace<HeightmapConfig>;

tag!(
    pub ShadedImageTag, u32, VoxelSize, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU shaded (RGBA) images"
);

/// Handle to configuration buffers used when computing shading
pub type ShadeWorkspace = ConfigWorkspace<ShadeConfig>;

/// Handle to size-dependent buffers used when computing shading
pub struct ShadeBuffers {
    out: FlexBuffer<ShadedImageTag>,
    bind_group: wgpu::BindGroup,
    has_color: bool,
}

impl ShadeBuffers {
    /// Returns a reference to the output buffer
    pub fn output(&self) -> &FlexBuffer<ShadedImageTag> {
        &self.out
    }
}

/// Error returned when submitting a merge operation
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(#[from] VoxelSizeMismatch),
}

/// Error returned when submitting a shade operation
#[derive(Debug, thiserror::Error)]
pub enum ShadeError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(VoxelSizeMismatch),

    /// The image size is mismatched between input and SSAO buffers
    #[error(transparent)]
    SsaoSizeMismatch(ImageSizeMismatch),
}

/// Error returned when submitting a heightmap operation
#[derive(Debug, thiserror::Error)]
pub enum HeightmapError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(#[from] VoxelSizeMismatch),
}

/// Error returned when submitting an SSAO operation
#[derive(Debug, thiserror::Error)]
pub enum SsaoError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(#[from] ImageSizeMismatch),
}

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

    /// The input and output buffers have mismatched sizes
    #[error(transparent)]
    SizeMismatch(#[from] VoxelSizeMismatch),

    /// Shape color has already been populated
    #[error("shape color has already been populated")]
    AlreadyHasColor,
}

impl From<CopyVarsError> for ColorError {
    fn from(value: CopyVarsError) -> Self {
        match value {
            CopyVarsError::MissingVar(v) => Self::MissingVar(v),
        }
    }
}

/// Type indicating an image size mismatch in a 3D image
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {} × {}, got {} × {} × {}",
    expected.width(), expected.height(), expected.depth(),
    actual.width(),   actual.height(),   actual.depth()
)]
pub struct VoxelSizeMismatch {
    expected: VoxelSize,
    actual: VoxelSize,
}

/// Type indicating an image size mismatch in a 2D image
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {}, got {} × {}",
    expected.width(), expected.height(),
    actual.width(),   actual.height(),
)]
pub struct ImageSizeMismatch {
    expected: ImageSize,
    actual: ImageSize,
}

impl Context {
    /// Builds a new context for applying effects
    pub fn new(gpu: &Gpu) -> Self {
        let shader_code = merge_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects merge pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&gpu.output_bind_group_layout),
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
                label: Some("voxel effects merge compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let shader_code = shade_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects shade pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.input_bind_group_layout), // image
                    Some(&gpu.input_bind_group_layout), // occlusion
                    Some(&gpu.output_bind_group_layout), // out
                ],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects shade shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let shade_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects shade compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("shade_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let shader_code = heightmap_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects heightmap pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&gpu.output_bind_group_layout),
                ],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects heightmap shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let heightmap_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects heightmap compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("heightmap_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let ssao_ctx = SsaoContext::new(gpu);
        let color_ctx = ColorContext::new(gpu);

        Self {
            gpu: gpu.clone(),
            merge_pipeline,
            shade_pipeline,
            heightmap_pipeline,
            ssao_ctx,
            color_ctx,
        }
    }

    /// Builds a new [`MergeWorkspace`]
    pub fn merge_workspace(&self) -> MergeWorkspace {
        self.gpu.config_workspace("merge")
    }

    /// Builds a new [`HeightmapWorkspace`]
    pub fn heightmap_workspace(&self) -> HeightmapWorkspace {
        self.gpu.config_workspace("heightmap")
    }

    /// Builds a new set of [`MergeBuffers`] for a particular render size
    pub fn merge_buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<MergeBuffers, BufferSizeError> {
        let out =
            FlexBuffer::new(&self.gpu.device, "merge output", image_size)?;
        let rw_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("merge output bind group"),
                    layout: &self.gpu.output_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out.bind_active(),
                    }],
                });
        let ro_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("merge output bind group"),
                    layout: &self.gpu.input_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out.bind_active(),
                    }],
                });
        Ok(MergeBuffers {
            out,
            rw_bind_group,
            ro_bind_group,
            image_count: 0,
        })
    }

    /// Builds a new set of [`ShadeWorkspace`]
    ///
    /// These will be resized when first used (in
    /// either [`submit_color`](Self::submit_color) or
    /// [`submit_shade`](Self::submit_shade))
    pub fn shade_workspace(&self) -> ShadeWorkspace {
        self.gpu.config_workspace("shade")
    }

    /// Builds a new set of [`ShadeBuffers`] for a particular render size
    pub fn shade_buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<ShadeBuffers, BufferSizeError> {
        let out =
            FlexBuffer::new(&self.gpu.device, "shade output", image_size)?;
        let bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("merge output bind group"),
                    layout: &self.gpu.output_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out.bind_active(),
                    }],
                });
        Ok(ShadeBuffers {
            out,
            bind_group,
            has_color: false,
        })
    }

    /// Builds a new set of [`SsaoWorkspace`] buffers
    pub fn ssao_workspace(&self) -> SsaoWorkspace {
        let ssao = self.gpu.config_workspace("ssao");
        let blur = self.gpu.config_workspace("blur");
        SsaoWorkspace { ssao, blur }
    }

    /// Builds a new set of [`SsaoBuffers`], which are size-dependent
    pub fn ssao_buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<SsaoBuffers, BufferSizeError> {
        let raw_occlusion = FlexBuffer::new(
            &self.gpu.device,
            "ssao raw occlusion",
            image_size,
        )?;
        let blurred_occlusion = FlexBuffer::new(
            &self.gpu.device,
            "ssao blurred occlusion",
            image_size,
        )?;
        let ssao_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssao bind group"),
                    layout: &self.gpu.output_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: raw_occlusion.bind_active(),
                    }],
                });
        let blur_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("blur bind group"),
                    layout: &self.ssao_ctx.blur_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: raw_occlusion.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: blurred_occlusion.bind_active(),
                        },
                    ],
                });
        let output_bind_group =
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssao output bind group"),
                    layout: &self.gpu.input_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: blurred_occlusion.bind_active(),
                    }],
                });
        Ok(SsaoBuffers {
            raw_occlusion,
            blurred_occlusion,
            ssao_bind_group,
            blur_bind_group,
            output_bind_group,
        })
    }

    /// Accumulates an image into a merged image buffer
    ///
    /// [`MergeWorkspace::reset`] should be called before the first call to
    /// `submit_merge`. For the first merge after a reset, the output buffer is
    /// resized to fit the images; subsequent merges must be of the same size.
    pub fn submit_merge(
        &self,
        image: &super::Buffers,
        settings: MergeSettings,
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
            settings,
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

    /// Low-level function to encode a merge operation to a command encoder
    pub fn encode_merge(
        &self,
        image: &super::Buffers,
        settings: MergeSettings,
        workspace: &mut MergeWorkspace,
        buf: &mut MergeBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), MergeError> {
        let size = image.geom.size();
        if buf.out.size() != size {
            return Err(VoxelSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }
            .into());
        }
        let cfg = MergeConfig {
            image_size: [size.width(), size.height()],
            denoise: settings.denoise as u32,
            index_base: buf.image_count as u32,
            z_scale: settings.z_scale,
            _pad: 0,
        };
        let mut writer = staging.write_buffer(
            encoder,
            &workspace.config,
            0,
            (std::mem::size_of::<MergeConfig>() as u64)
                .try_into()
                .unwrap(),
        );
        writer.copy_from_slice(cfg.as_bytes());

        // Scope to bound the lifetime of compute_pass
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.merge_pipeline);
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, &image.bind_groups.output, &[]);
        compute_pass.set_bind_group(2, &buf.rw_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.image_count += 1;
        Ok(())
    }

    /// Submits an operation to shade an image
    ///
    /// The output buffer is resized to fit the incoming image
    pub fn submit_shade(
        &self,
        image: &MergeBuffers,
        ssao: Option<&SsaoBuffers>,
        workspace: &mut ShadeWorkspace,
        buf: &mut ShadeBuffers,
    ) -> Result<(), ShadeError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("shade compute encoder"),
            },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode_shade(
            image,
            ssao,
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

    /// Low-level function to encode a shading operation to a command encoder
    pub fn encode_shade(
        &self,
        image: &MergeBuffers,
        ssao: Option<&SsaoBuffers>,
        workspace: &mut ShadeWorkspace,
        buf: &mut ShadeBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), ShadeError> {
        let size = image.out.size();
        if size != buf.out.size() {
            return Err(ShadeError::ImageSizeMismatch(VoxelSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }));
        }
        if let Some(ssao) = ssao
            && ssao.blurred_occlusion.size() != size.into()
        {
            return Err(ShadeError::SsaoSizeMismatch(ImageSizeMismatch {
                expected: size.into(),
                actual: ssao.blurred_occlusion.size(),
            }));
        }

        // Load the config before beginning the compute pass
        let cfg = ShadeConfig {
            image_size: [size.width(), size.height(), size.depth()],
            flags: if ssao.is_some() {
                SHADE_CONFIG_HAS_SSAO
            } else {
                0
            } | if buf.has_color {
                SHADE_CONFIG_HAS_COLOR
            } else {
                0
            },
        };
        workspace.encode_copy(encoder, staging, &cfg);

        // Scope to bound the lifetime of compute_pass
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.shade_pipeline);
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, &image.ro_bind_group, &[]);
        compute_pass.set_bind_group(
            2,
            ssao.map(|s| &s.output_bind_group)
                .unwrap_or(&image.ro_bind_group), // dummy binding
            &[],
        );
        compute_pass.set_bind_group(3, &buf.bind_group, &[]);

        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.has_color = false;
        Ok(())
    }

    /// Submits an operation to generate RGB color from a heightmap
    ///
    /// The output buffer is resized to fit the incoming image
    pub fn submit_heightmap(
        &self,
        image: &MergeBuffers,
        buf: &mut ShadeBuffers,
        workspace: &mut HeightmapWorkspace,
    ) -> Result<(), HeightmapError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("heightmap compute encoder"),
            },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode_heightmap(
            image,
            buf,
            workspace,
            &mut encoder,
            &mut staging,
        )?;
        staging.finish();
        self.gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    /// Low-level function to encode heightmap rendering
    pub fn encode_heightmap(
        &self,
        image: &MergeBuffers,
        buf: &mut ShadeBuffers,
        workspace: &mut HeightmapWorkspace,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), HeightmapError> {
        let size = image.out.size();
        if size != buf.out.size() {
            return Err(HeightmapError::ImageSizeMismatch(VoxelSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }));
        }

        let cfg = HeightmapConfig {
            image_size: [size.width(), size.height(), size.depth()],
            has_color: buf.has_color.into(),
        };
        workspace.encode_copy(encoder, staging, &cfg);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("heightmap compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.heightmap_pipeline);
        compute_pass.set_bind_group(0, Some(&workspace.bind_group), &[]);
        compute_pass.set_bind_group(1, Some(&image.ro_bind_group), &[]);
        compute_pass.set_bind_group(2, Some(&buf.bind_group), &[]);
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.has_color = false;
        Ok(())
    }

    /// Submits a pass to compute an SSAO buffer
    pub fn submit_ssao(
        &self,
        image: &MergeBuffers,
        workspace: &mut SsaoWorkspace,
        buf: &mut SsaoBuffers,
    ) -> Result<(), SsaoError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("ssao command encoder"),
            },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode_ssao(image, workspace, buf, &mut encoder, &mut staging)?;
        staging.finish();
        self.gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    /// Low-level function to encode an SSAO pass
    pub fn encode_ssao(
        &self,
        image: &MergeBuffers,
        workspace: &mut SsaoWorkspace,
        buf: &mut SsaoBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), SsaoError> {
        self.ssao_ctx
            .encode(image, workspace, buf, encoder, staging)
    }

    /// Submits a color evaluation pass
    ///
    /// Image size is set from the `MergeWorkspace`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color(
        &self,
        merge: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        workspace: &mut ColorWorkspace,
        out: &mut ShadeBuffers,
    ) -> Result<(), ColorError> {
        self.submit_color_with_vars(
            merge,
            world_to_model,
            shape,
            workspace,
            &Default::default(),
            out,
        )
    }

    /// Submits a color evaluation pass with auxiliary variables
    ///
    /// Image size is set from the `MergeWorkspace`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color_with_vars(
        &self,
        merge: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        workspace: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
        out: &mut ShadeBuffers,
    ) -> Result<(), ColorError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode_color(
            merge,
            world_to_model,
            shape,
            workspace,
            vars,
            out,
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
        merge: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        workspace: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
        out: &mut ShadeBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), ColorError> {
        self.color_ctx.encode(
            merge,
            world_to_model,
            shape,
            workspace,
            vars,
            out,
            encoder,
            staging,
        )
    }

    /// Returns a new workspace for color evaluation
    pub fn color_workspace(&self) -> ColorWorkspace {
        self.gpu.config_workspace("color")
    }
}

////////////////////////////////////////////////////////////////////////////////

tag!(pub SsaoRawBufferTag, f32, ImageSize, STORAGE | COPY_SRC,
    "Tag for a raw SSAO occlusion buffer");
tag!(pub SsaoBlurredBufferTag, f32, ImageSize, STORAGE | COPY_SRC,
    "Tag for a blurred SSAO occlusion buffer");

/// Handle to a set of configuration buffers used when running an SSAO pass
pub struct SsaoWorkspace {
    ssao: ConfigWorkspace<SsaoConfig>,
    blur: ConfigWorkspace<BlurConfig>,
}

/// Handle to a set of image buffers used when running an SSAO pass
///
/// Unlike [`SsaoWorkspace`], these buffers are size-dependent
pub struct SsaoBuffers {
    /// Raw occlusion buffer
    raw_occlusion: FlexBuffer<SsaoRawBufferTag>,

    /// Bind group used in the SSAO pass (writing to `raw_occlusion`)
    ssao_bind_group: wgpu::BindGroup,

    /// Blurred occlusion buffer
    blurred_occlusion: FlexBuffer<SsaoBlurredBufferTag>,

    /// Bind group used in the blur pass
    ///
    /// This pass reads from `raw_occlusion` and writes to `blurred_occlusion`
    blur_bind_group: wgpu::BindGroup,

    /// Bind group which exposes the blurred occlusion buffer (read-only)
    output_bind_group: wgpu::BindGroup,
}

impl SsaoBuffers {
    /// Returns a shared handle to the raw SSAO occlusion buffer
    pub fn raw_occlusion(&self) -> &FlexBuffer<SsaoRawBufferTag> {
        &self.raw_occlusion
    }

    /// Returns a shared handle to the blurred SSAO occlusion buffer
    pub fn blurred_occlusion(&self) -> &FlexBuffer<SsaoBlurredBufferTag> {
        &self.blurred_occlusion
    }
}

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
struct SsaoConfig {
    /// Image size, in voxels
    image_size: [u32; 3],

    /// Radius of SSAO sampling
    radius: f32,
}

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
struct BlurConfig {
    /// Image size, in pixels
    image_size: [u32; 2],

    /// Pixel radius of blur
    radius: u32,

    /// Padding to 16 bytes
    _pad: u32,
}

struct SsaoContext {
    /// Fixed bind group for SSAO pass
    ///
    /// This contains the SSAO kernel and noise buffers, which are constants.
    ssao_bind_group: wgpu::BindGroup,

    /// Pipeline for computing per-pixel SSAO
    ssao_pipeline: wgpu::ComputePipeline,

    /// Layout for blur pipeline
    blur_bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for blurring an SSAO image
    blur_pipeline: wgpu::ComputePipeline,
}

impl SsaoContext {
    pub fn new(gpu: &Gpu) -> Self {
        let ssao_fixed_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[buffer_ro(0), buffer_ro(1)],
            },
        );

        const KERNEL_SIZE: usize = 64;
        const NOISE_SIZE: usize = 16;

        // Build constant buffers and their bind group
        let ssao_kernel_size_bytes =
            KERNEL_SIZE * std::mem::size_of::<[f32; 3]>();
        let ssao_kernel = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao kernel"),
            size: ssao_kernel_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        let ssao_kernel_values =
            fidget_raster::effects::ssao_kernel(KERNEL_SIZE);
        ssao_kernel
            .get_mapped_range_mut(0..ssao_kernel_size_bytes as u64)
            .unwrap()
            .copy_from_slice(ssao_kernel_values.as_slice().as_bytes());
        ssao_kernel.unmap();

        let ssao_noise_size_bytes =
            NOISE_SIZE * std::mem::size_of::<[f32; 2]>();
        let ssao_noise = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao noise"),
            size: ssao_noise_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        let ssao_noise_values = fidget_raster::effects::ssao_noise(NOISE_SIZE);
        ssao_noise
            .get_mapped_range_mut(0..ssao_noise_size_bytes as u64)
            .unwrap()
            .copy_from_slice(ssao_noise_values.as_slice().as_bytes());
        ssao_noise.unmap();

        let ssao_bind_group =
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao fixed bind group"),
                layout: &ssao_fixed_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ssao_kernel.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ssao_noise.as_entire_binding(),
                    },
                ],
            });

        let shader_code = ssao_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects ssao pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&ssao_fixed_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&gpu.output_bind_group_layout),
                ],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects ssao shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let ssao_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects ssao compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("ssao_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let blur_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[buffer_ro(0), buffer_rw(1)],
            },
        );
        let shader_code = blur_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects blur pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&blur_bind_group_layout),
                ],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects blur shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let blur_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects blur compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("blur_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        Self {
            ssao_bind_group,
            ssao_pipeline,
            blur_bind_group_layout,
            blur_pipeline,
        }
    }

    fn encode(
        &self,
        image: &MergeBuffers,
        workspace: &mut SsaoWorkspace,
        buf: &mut SsaoBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), SsaoError> {
        let image_size = image.out.size();
        if buf.raw_occlusion.size() != image_size.into() {
            return Err(SsaoError::ImageSizeMismatch(ImageSizeMismatch {
                expected: image_size.into(),
                actual: buf.raw_occlusion.size(),
            }));
        }

        // Write config data to GPU buffers
        let cfg = SsaoConfig {
            image_size: [
                image_size.width(),
                image_size.height(),
                image_size.depth(),
            ],
            radius: 0.1,
        };
        workspace.ssao.encode_copy(encoder, staging, &cfg);

        let cfg = BlurConfig {
            image_size: [image_size.width(), image_size.height()],
            radius: 2,
            _pad: 0,
        };
        workspace.blur.encode_copy(encoder, staging, &cfg);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.ssao_pipeline);

        compute_pass.set_bind_group(0, Some(&workspace.ssao.bind_group), &[]);
        compute_pass.set_bind_group(1, Some(&self.ssao_bind_group), &[]);
        compute_pass.set_bind_group(2, Some(&image.ro_bind_group), &[]);
        compute_pass.set_bind_group(3, Some(&buf.ssao_bind_group), &[]);
        compute_pass.dispatch_workgroups(
            image_size.width().div_ceil(8),
            image_size.height().div_ceil(8),
            1,
        );

        compute_pass.set_pipeline(&self.blur_pipeline);
        compute_pass.set_bind_group(0, Some(&workspace.blur.bind_group), &[]);
        compute_pass.set_bind_group(1, Some(&buf.blur_bind_group), &[]);
        compute_pass.dispatch_workgroups(
            image_size.width().div_ceil(8),
            image_size.height().div_ceil(8),
            1,
        );

        Ok(())
    }
}

struct ColorContext {
    /// Pipeline for computing per-pixel color
    color_pipeline: RegPipeline,
}

impl ColorContext {
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("color pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.color_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&gpu.output_bind_group_layout),
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

        Self { color_pipeline }
    }

    #[allow(clippy::too_many_arguments)] // what are ya gonna do?
    fn encode(
        &self,
        image: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        workspace: &mut ColorWorkspace,
        vars: &ShapeVars<f32>,
        out: &mut ShadeBuffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), ColorError> {
        if image.image_count != shape.shape_count() {
            return Err(ColorError::BadShapeCount {
                merge_count: image.image_count,
                shape_count: shape.shape_count(),
            });
        }
        let size = image.out.size();
        if size != out.out.size() {
            return Err(VoxelSizeMismatch {
                expected: size,
                actual: out.out.size(),
            }
            .into());
        }
        shape.copy_vars(encoder, staging, vars)?;

        out.has_color = true;
        let mat = world_to_model * size.screen_to_world();
        let config = ColorConfig {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            image_size: [size.width(), size.height()],
            _pad1: 0,
            _pad2: [0; 2],
        };
        workspace.encode_copy(encoder, staging, &config);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, shape.bind_group(), &[]);
        compute_pass.set_bind_group(2, &image.ro_bind_group, &[]);
        compute_pass.set_bind_group(3, &out.bind_group, &[]);

        compute_pass.set_pipeline(self.color_pipeline.get(shape.reg_count()));
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{context::Tree, vm::VmShape};
    use fidget_raster::voxel::RenderSize;

    #[test]
    fn packed_voxel_size() {
        assert_eq!(std::mem::size_of::<PackedVoxel>(), 8);
    }

    #[test]
    fn compile_merge_shader() {
        crate::compile_shader(&merge_shader(), "merge");
    }

    #[test]
    fn compile_shade_shader() {
        crate::compile_shader(&shade_shader(), "shade");
    }

    #[test]
    fn compile_heightmap_shader() {
        crate::compile_shader(&heightmap_shader(), "heightmap");
    }

    #[test]
    fn compile_ssao_shader() {
        crate::compile_shader(&ssao_shader(), "ssao");
    }

    #[test]
    fn compile_blur_shader() {
        crate::compile_shader(&blur_shader(), "blur");
    }

    #[test]
    fn compile_color_shader() {
        crate::compile_shader(&color_shader(16), "color");
    }

    /// Render a sphere-plane union and check for occlusion bias
    ///
    /// Because the image is perfectly symmetrical, we'd expect the average
    /// occlusion across each of the four corners to be very similar.  If it's
    /// not, then that's likely a sampling bias – which we have seen before!
    #[test]
    fn ssao_bias() {
        crate::require_gpu!();
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let voxel_ctx = crate::voxel::Context::new(&gpu);
        let mut voxel_workspace = voxel_ctx.workspace();
        let effects_ctx = crate::voxel::effects::Context::new(&gpu);
        let mut merge_workspace = effects_ctx.merge_workspace();

        let size = 128;
        let image_size = RenderSize::from(size);
        let mut voxel_buf = voxel_ctx.buffers(image_size).unwrap();
        let mut merge_buf = effects_ctx.merge_buffers(image_size).unwrap();

        let (x, y, z) = Tree::axes();
        let sphere =
            (x.square() + y.square() + z.square()).sqrt() - Tree::constant(0.5);
        let vm_shape = VmShape::from(sphere.min(z));
        let shape = gpu.shape_buffers(&vm_shape).unwrap();

        voxel_ctx
            .submit(
                &shape,
                &mut voxel_workspace,
                &mut voxel_buf,
                &crate::voxel::RenderConfig {
                    world_to_model: nalgebra::Matrix4::identity(),
                },
            )
            .unwrap();
        effects_ctx
            .submit_merge(
                &voxel_buf,
                Default::default(),
                &mut merge_workspace,
                &mut merge_buf,
            )
            .unwrap();
        let mut ssao_workspace = effects_ctx.ssao_workspace();
        let mut ssao_buf = effects_ctx.ssao_buffers(image_size.into()).unwrap();
        effects_ctx
            .submit_ssao(&merge_buf, &mut ssao_workspace, &mut ssao_buf)
            .unwrap();
        let ssao_out = gpu.read_vec(ssao_buf.raw_occlusion());

        let quadrants =
            [(0, 0), (size / 2, 0), (0, size / 2), (size / 2, size / 2)];
        let mut averages = Vec::with_capacity(quadrants.len());
        for (dx, dy) in quadrants {
            let mut sum = 0.0;
            let mut count = 0.0;
            for x in 0..size / 2 {
                for y in 0..size / 2 {
                    let x = (x + dx) as usize;
                    let y = (y + dy) as usize;
                    sum += ssao_out[x + y * size as usize];
                    count += 1.0;
                }
            }
            averages.push(sum / count);
        }
        for (i, qa) in quadrants.iter().enumerate() {
            for (j, qb) in quadrants.iter().enumerate() {
                let oa = averages[i];
                let ob = averages[j];
                let d = (oa - ob).abs();
                let epsilon = 0.01;
                if d > epsilon {
                    panic!(
                        "average occlusion between quadrants with offsets \
                        {qa:?} and {qb:?} differs by too much: \
                        {oa:.3} ≉ {ob:.3}"
                    );
                }
            }
        }
    }

    #[test]
    fn color_config_layout() {
        crate::test::compare_struct_layout::<ColorConfig>(
            &color_shader(16),
            "Config",
        );
    }

    #[test]
    fn merge_config_layout() {
        crate::test::compare_struct_layout::<MergeConfig>(
            &merge_shader(),
            "MergeConfig",
        );
    }

    #[test]
    fn shade_config_layout() {
        crate::test::compare_struct_layout::<ShadeConfig>(
            &shade_shader(),
            "ShadeConfig",
        );
    }

    #[test]
    fn heightmap_config_layout() {
        crate::test::compare_struct_layout::<HeightmapConfig>(
            &heightmap_shader(),
            "HeightmapConfig",
        );
    }

    #[test]
    fn ssao_config_layout() {
        crate::test::compare_struct_layout::<SsaoConfig>(
            &ssao_shader(),
            "SsaoConfig",
        );
    }

    #[test]
    fn blur_config_layout() {
        crate::test::compare_struct_layout::<BlurConfig>(
            &blur_shader(),
            "BlurConfig",
        );
    }
}
