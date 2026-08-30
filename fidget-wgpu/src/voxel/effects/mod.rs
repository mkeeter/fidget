//! On-GPU effects
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
    Gpu, RegPipeline, ShapeColor, ShapeColorBuffers, ShapeColorError,
    buf::{
        BufferSizeError, DepthImageBuffer, ImageBuffer, ImageReadBuffer,
        buffer_ro, buffer_rw, buffer_uniform,
    },
    shaders, tag,
    voxel::GeomBufferTag,
};
use fidget_core::{
    render::VoxelSize,
    shape::{MissingVar, ShapeVars},
    var::Var,
    vm::VmShape,
};
use std::num::NonZeroU64;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// WGPU context for applying various effects
pub struct Context {
    gpu: Gpu,

    merge_bind_group_layout: wgpu::BindGroupLayout,
    merge_pipeline: wgpu::ComputePipeline,

    shade_bind_group_layout: wgpu::BindGroupLayout,
    shade_pipeline: wgpu::ComputePipeline,

    ssao_ctx: SsaoContext,

    color_ctx: ColorContext,
}

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const SHADE_SHADER: &str = include_str!("shaders/shade.wgsl");
const SSAO_SHADER: &str = include_str!("shaders/ssao.wgsl");
const BLUR_SHADER: &str = include_str!("shaders/blur.wgsl");
const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
}

fn shade_shader() -> String {
    SHADE_SHADER.to_owned() + COMMON_SHADER + shaders::COMMON
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
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
struct ColorConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    // padding for alignment
    _pad: u32,

    /// Image size (pixels)
    image_size: [u32; 2],
    // this is followed by a tape_data flexible array member
}

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
struct MergeConfig {
    /// Image size, in pixels
    image_size: [u32; 2],

    /// Whether or not to denoise when merging (non-zero is true)
    denoise: u32,

    /// Offset applied to indices when merging
    ///
    /// When this is 0, we initialize the output image
    index_base: u32,
}

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
struct ShadeConfig {
    /// Image size, in pixels
    image_size: [u32; 3],

    /// Flags to set the presence of color and SSAO
    flags: u32,
}

/// Must match constants in `shade.wgsl`
const SHADE_CONFIG_HAS_SSAO: u32 = 1u32;
const SHADE_CONFIG_HAS_COLOR: u32 = 2u32;

tag!(pub MergeVoxelBufferTag, PackedVoxel, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU merged ([`PackedVoxel`]) images"
);

/// Handle to a set of buffers used when merging images
pub struct MergeBuffers {
    config: wgpu::Buffer,
    out: DepthImageBuffer<MergeVoxelBufferTag>,
    image_count: usize,
}

impl MergeBuffers {
    /// Returns a handle to the output buffer
    pub fn output(&self) -> &DepthImageBuffer<MergeVoxelBufferTag> {
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

tag!(
    pub ShadedImageTag, u32, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU shaded (RGBA) images"
);

/// Handle to a set of buffers used when shading images
pub struct ShadeBuffers {
    config: wgpu::Buffer,
    out: DepthImageBuffer<ShadedImageTag>,
}

impl ShadeBuffers {
    /// Returns a reference to the output buffer
    pub fn output(&self) -> &DepthImageBuffer<ShadedImageTag> {
        &self.out
    }
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

/// Error returned when submitting a shade operation
#[derive(Debug, thiserror::Error)]
pub enum ShadeError {
    /// An error occurred while resizing the output buffer
    #[error(transparent)]
    OutputSize(BufferSizeError),

    /// Input and output buffers are different sizes when `has_color` is true
    ///
    /// This is not allowed because `has_color = true` means that the output
    /// buffer's pixel values should be used as diffuse color
    #[error(
        "input and output buffers must be the same size when
        `has_color` is true, but they do not match"
    )]
    InvalidColorSize,
}

/// Error returned when submitting an SSAO operation
#[derive(Debug, thiserror::Error)]
pub enum SsaoError {
    /// An error occurred while resizing the output buffer
    #[error(transparent)]
    OutputSize(BufferSizeError),
}

/// Error returned when submitting a color evaluation operation
#[derive(Debug, thiserror::Error)]
pub enum ColorError {
    /// An error occurred while resizing the output buffer
    #[error(transparent)]
    OutputSize(BufferSizeError),

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
}

/// Type indicating an image size mismatch
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {} × {}, got {} × {} × {}",
    expected.width(), expected.height(), expected.depth(),
    actual.width(),   actual.height(),   actual.depth()
)]
pub struct ImageSizeMismatch {
    expected: VoxelSize,
    actual: VoxelSize,
}

impl Context {
    /// Builds a new context for applying effects
    pub fn new(gpu: &Gpu) -> Self {
        let merge_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_uniform(0), // config
                    buffer_ro(1),      // image
                    buffer_rw(2),      // out
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

        let shade_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_uniform(0),
                    buffer_ro(1), // image
                    buffer_ro(2), // ssao occlusion
                    buffer_rw(3), // out
                ],
            },
        );
        let shader_code = shade_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects shade pipeline"),
                bind_group_layouts: &[Some(&shade_bind_group_layout)],
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

        let ssao_ctx = SsaoContext::new(&gpu.device);
        let color_ctx = ColorContext::new(&gpu.device);

        Self {
            gpu: gpu.clone(),
            merge_bind_group_layout,
            merge_pipeline,
            shade_bind_group_layout,
            shade_pipeline,
            ssao_ctx,
            color_ctx,
        }
    }

    /// Builds a new set of [`MergeBuffers`]
    ///
    /// These will be resized when first used (in
    /// [`submit_merge`](Self::submit_merge))
    pub fn merge_buffers(&self) -> MergeBuffers {
        let config = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<MergeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = DepthImageBuffer::new(
            &self.gpu.device,
            "merge output".to_owned(),
            64.into(),
        )
        .expect("64 is always a valid size");
        MergeBuffers {
            config,
            out,
            image_count: 0,
        }
    }

    /// Builds a new set of [`ShadeBuffers`]
    ///
    /// These will be resized when first used (in
    /// either [`submit_color`](Self::submit_color) or
    /// [`submit_shade`](Self::submit_shade))
    pub fn shade_buffers(&self) -> ShadeBuffers {
        let config = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade config"),
            size: std::mem::size_of::<ShadeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = DepthImageBuffer::new(
            &self.gpu.device,
            "shade output".to_owned(),
            64.into(),
        )
        .expect("64 is always a valid size");
        ShadeBuffers { config, out }
    }

    /// Builds a new set of [`SsaoBuffers`]
    ///
    /// These will be resized when first used (in
    /// [`submit_ssao`](Self::submit_ssao))
    pub fn ssao_buffers(&self) -> SsaoBuffers {
        let ssao_config =
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ssao config"),
                size: std::mem::size_of::<SsaoConfig>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let image_size = 64.into();
        let raw_occlusion = ImageBuffer::new(
            &self.gpu.device,
            "ssao raw occlusion".to_owned(),
            image_size,
        )
        .expect("64 is always a valid size");
        let blur_config =
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("blur config"),
                size: std::mem::size_of::<BlurConfig>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let blurred_occlusion = ImageBuffer::new(
            &self.gpu.device,
            "ssao blurred occlusion".to_owned(),
            image_size,
        )
        .expect("64 is always a valid size");
        SsaoBuffers {
            ssao_config,
            blur_config,
            raw_occlusion,
            blurred_occlusion,
        }
    }

    /// Accumulates an image into a merged image buffer
    ///
    /// If the merged buffer has been reset (with [`MergeBuffers::reset`]), then
    /// it is resized to fit the incoming image; otherwise, an error is returned
    /// if the image size does not match.
    pub fn submit_merge(
        &self,
        image: &DepthImageBuffer<GeomBufferTag>,
        denoise: bool,
        buf: &mut MergeBuffers,
    ) -> Result<(), MergeError> {
        let size = image.size();
        if buf.image_count == 0 {
            buf.out
                .grow_to_fit(&self.gpu.device, size)
                .map_err(MergeError::OutputSize)?;
        } else if buf.out.size() != size {
            return Err(ImageSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }
            .into());
        }
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
                denoise: denoise as u32,
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
                                resource: buf.out.bind_active(),
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

    /// Submits an operation to shade an image
    ///
    /// The output buffer is resized to fit the images
    pub fn submit_shade(
        &self,
        image: &MergeBuffers,
        ssao: Option<&SsaoBuffers>,
        buf: &mut ShadeBuffers,
        has_color: bool,
        out: Option<&mut ImageReadBuffer<ShadedImageTag>>,
    ) -> Result<(), ShadeError> {
        let size = image.out.size();
        if has_color && size != buf.out.size() {
            return Err(ShadeError::InvalidColorSize);
        }
        buf.out
            .grow_to_fit(&self.gpu.device, size)
            .map_err(ShadeError::OutputSize)?;
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("shade compute encoder"),
            },
        );

        // Scope to bound the lifetime of compute_pass
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("shade compute pass"),
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.shade_pipeline);
            let cfg = ShadeConfig {
                image_size: [size.width(), size.height(), size.depth()],
                flags: if ssao.is_some() {
                    SHADE_CONFIG_HAS_SSAO
                } else {
                    0
                } | if has_color { SHADE_CONFIG_HAS_COLOR } else { 0 },
            };
            {
                let mut writer = self
                    .gpu
                    .queue
                    .write_buffer_with(
                        &buf.config,
                        0,
                        buf.config.size().try_into().unwrap(),
                    )
                    .unwrap();
                writer.copy_from_slice(cfg.as_bytes());
            }
            // TODO This is created on every pass
            let bg =
                self.gpu
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("shade bind group"),
                        layout: &self.shade_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf.config.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: image.out.bind_active(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: ssao
                                    .map(|s| {
                                        s.blurred_occlusion().bind_active()
                                    })
                                    .unwrap_or_else(|| image.out.bind_active()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buf.out.bind_active(),
                            },
                        ],
                    });
            compute_pass.set_bind_group(0, Some(&bg), &[]);
            compute_pass.dispatch_workgroups(
                size.width().div_ceil(8),
                size.height().div_ceil(8),
                1,
            );
        }
        if let Some(image) = out {
            image
                .grow_to_fit(&self.gpu.device, buf.out.size().into())
                .expect(
                    "buf.out.size should always be \
                     a valid size for grow_to_fit",
                );
            encoder.copy_buffer_to_buffer(
                buf.out.data(),
                0,
                image.data(),
                0,
                buf.out.size_bytes(),
            );
        }
        self.gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Submits a pass to compute an SSAO buffer
    pub fn submit_ssao(
        &self,
        image: &MergeBuffers,
        buf: &mut SsaoBuffers,
    ) -> Result<(), SsaoError> {
        self.ssao_ctx.submit(image, buf, &self.gpu)
    }

    /// Build a set of buffers for doing shape color evaluation
    pub fn color_buffers(
        &self,
        colors: &[ShapeColor<VmShape>],
    ) -> Result<ShapeColorBuffers, ShapeColorError> {
        ShapeColorBuffers::new(
            colors,
            &self.gpu.device,
            std::mem::size_of::<ColorConfig>(),
        )
    }

    /// Submits a color evaluation pass
    ///
    /// Image size is set from the `MergeBuffers`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color(
        &self,
        merge: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        out: &mut ShadeBuffers,
    ) -> Result<(), ColorError> {
        self.submit_color_with_vars(
            merge,
            world_to_model,
            shape,
            &Default::default(),
            out,
        )
    }

    /// Submits a color evaluation pass with auxiliary variables
    ///
    /// Image size is set from the `MergeBuffers`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_color_with_vars(
        &self,
        merge: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        vars: &ShapeVars<f32>,
        out: &mut ShadeBuffers,
    ) -> Result<(), ColorError> {
        self.color_ctx.submit(
            merge,
            world_to_model,
            shape,
            vars,
            out,
            &self.gpu,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

tag!(pub SsaoRawBufferTag, f32, STORAGE | COPY_SRC,
    "Tag for a raw SSAO occlusion buffer");
tag!(pub SsaoBlurredBufferTag, f32, STORAGE | COPY_SRC,
    "Tag for a blurred SSAO occlusion buffer");

/// Handle to a set of buffers used when running an SSAO pass
pub struct SsaoBuffers {
    ssao_config: wgpu::Buffer, // TODO add a `ConfigBuffer` type?
    raw_occlusion: ImageBuffer<SsaoRawBufferTag>,

    blur_config: wgpu::Buffer,
    blurred_occlusion: ImageBuffer<SsaoBlurredBufferTag>,
}

impl SsaoBuffers {
    /// Returns a shared handle to the raw SSAO occlusion buffer
    pub fn raw_occlusion(&self) -> &ImageBuffer<SsaoRawBufferTag> {
        &self.raw_occlusion
    }

    /// Returns a shared handle to the blurred SSAO occlusion buffer
    pub fn blurred_occlusion(&self) -> &ImageBuffer<SsaoBlurredBufferTag> {
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

    /// Layout for bind group that accepts buffers from the user
    ssao_bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for computing per-pixel SSAO
    ssao_pipeline: wgpu::ComputePipeline,

    /// Layout for blur pipeline
    blur_bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for blurring an SSAO image
    blur_pipeline: wgpu::ComputePipeline,
}

impl SsaoContext {
    pub fn new(device: &wgpu::Device) -> Self {
        let ssao_fixed_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[buffer_ro(0), buffer_ro(1)],
            });
        let ssao_user_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[buffer_uniform(0), buffer_ro(1), buffer_rw(2)],
            });

        const KERNEL_SIZE: usize = 64;
        const NOISE_SIZE: usize = 16;

        // Build constant buffers and their bind group
        let ssao_kernel_size_bytes =
            KERNEL_SIZE * std::mem::size_of::<[f32; 3]>();
        let ssao_kernel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao kernel"),
            size: ssao_kernel_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        let ssao_kernel_values =
            fidget_raster::effects::ssao_kernel(KERNEL_SIZE);
        ssao_kernel
            .get_mapped_range_mut(0..ssao_kernel_size_bytes as u64)
            .copy_from_slice(ssao_kernel_values.as_slice().as_bytes());
        ssao_kernel.unmap();

        let ssao_noise_size_bytes =
            NOISE_SIZE * std::mem::size_of::<[f32; 2]>();
        let ssao_noise = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao noise"),
            size: ssao_noise_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        let ssao_noise_values = fidget_raster::effects::ssao_noise(NOISE_SIZE);
        ssao_noise
            .get_mapped_range_mut(0..ssao_noise_size_bytes as u64)
            .copy_from_slice(ssao_noise_values.as_slice().as_bytes());
        ssao_noise.unmap();

        let ssao_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("effects ssao pipeline"),
                bind_group_layouts: &[
                    Some(&ssao_user_bind_group_layout),
                    Some(&ssao_fixed_bind_group_layout),
                ],
                immediate_size: 0u32,
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("effects ssao shader module"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let ssao_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("effects ssao compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("ssao_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[buffer_uniform(0), buffer_ro(1), buffer_rw(2)],
            });
        let shader_code = blur_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("effects blur pipeline"),
                bind_group_layouts: &[Some(&blur_bind_group_layout)],
                immediate_size: 0u32,
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("effects blur shader module"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let blur_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("effects blur compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("blur_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            ssao_bind_group,
            ssao_bind_group_layout: ssao_user_bind_group_layout,
            ssao_pipeline,
            blur_bind_group_layout,
            blur_pipeline,
        }
    }

    fn submit(
        &self,
        image: &MergeBuffers,
        buf: &mut SsaoBuffers,
        gpu: &Gpu,
    ) -> Result<(), SsaoError> {
        let image_size = image.out.size();
        buf.raw_occlusion
            .grow_to_fit(&gpu.device, image_size.into())
            .map_err(SsaoError::OutputSize)?;
        buf.blurred_occlusion
            .grow_to_fit(&gpu.device, image_size.into())
            .map_err(SsaoError::OutputSize)?;

        // TODO make this passed in?
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("ssao command encoder"),
            },
        );

        // Scope to bound the lifetime of compute_pass
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ssao compute pass"),
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.ssao_pipeline);
            let cfg = SsaoConfig {
                image_size: [
                    image_size.width(),
                    image_size.height(),
                    image_size.depth(),
                ],
                radius: 0.1,
            };
            {
                let mut writer = gpu
                    .queue
                    .write_buffer_with(
                        &buf.ssao_config,
                        0,
                        (std::mem::size_of::<SsaoConfig>() as u64)
                            .try_into()
                            .unwrap(),
                    )
                    .unwrap();
                writer.copy_from_slice(cfg.as_bytes());
            }

            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao bind group"),
                layout: &self.ssao_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.ssao_config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image.out.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf.raw_occlusion.bind_active(),
                    },
                ],
            });
            compute_pass.set_bind_group(0, Some(&bg), &[]);
            compute_pass.set_bind_group(1, Some(&self.ssao_bind_group), &[]);
            compute_pass.dispatch_workgroups(
                image_size.width().div_ceil(8),
                image_size.height().div_ceil(8),
                1,
            );
        }

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ssao blur compute pass"),
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.blur_pipeline);
            let cfg = BlurConfig {
                image_size: [image_size.width(), image_size.height()],
                radius: 2,
                _pad: 0,
            };
            {
                let mut writer = gpu
                    .queue
                    .write_buffer_with(
                        &buf.blur_config,
                        0,
                        (std::mem::size_of::<BlurConfig>() as u64)
                            .try_into()
                            .unwrap(),
                    )
                    .unwrap();
                writer.copy_from_slice(cfg.as_bytes());
            }

            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur bind group"),
                layout: &self.blur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.blur_config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf.raw_occlusion.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf.blurred_occlusion.bind_active(),
                    },
                ],
            });
            compute_pass.set_bind_group(0, Some(&bg), &[]);
            compute_pass.dispatch_workgroups(
                image_size.width().div_ceil(8),
                image_size.height().div_ceil(8),
                1,
            );
        }

        gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }
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
                entries: &[buffer_ro(0), buffer_rw(1)],
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

    /// The output buffer is resized to fit `image`
    fn submit(
        &self,
        image: &MergeBuffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &ShapeColorBuffers,
        vars: &ShapeVars<f32>,
        out: &mut ShadeBuffers,
        gpu: &Gpu,
    ) -> Result<(), ColorError> {
        if image.image_count != shape.shape_count {
            return Err(ColorError::BadShapeCount {
                merge_count: image.image_count,
                shape_count: shape.shape_count,
            });
        }
        let size = image.out.size();
        out.out
            .grow_to_fit(&gpu.device, size)
            .map_err(ColorError::OutputSize)?;

        let mat = world_to_model * size.screen_to_world();

        let config_bg = shape
            .config_bind_group(&gpu.device, &self.config_bind_group_layout);

        let config = ColorConfig {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            image_size: [size.width(), size.height()],
            _pad: 0,
        };

        {
            // We load the `ColorConfig`; tape data is already in the buffer
            let config_len = std::mem::size_of_val(&config);
            let mut writer = gpu
                .queue
                .write_buffer_with(
                    &shape.config,
                    0,
                    (config_len as u64).try_into().unwrap(),
                )
                .unwrap();
            writer.copy_from_slice(config.as_bytes());
        }

        // Copy vars (if present)
        if let Some(var_size) = NonZeroU64::new(shape.vars.size()) {
            let mut writer = gpu
                .queue
                .write_buffer_with(&shape.vars, 0, var_size)
                .unwrap();
            for (v, i) in shape.var_map.iter() {
                match v {
                    Var::X | Var::Y | Var::Z => (),
                    Var::V(vi) => {
                        let Some(value) = vars.get(vi) else {
                            return Err(MissingVar { var: vi }.into());
                        };
                        let offset = i * std::mem::size_of::<f32>();
                        writer
                            .slice(offset..offset + 4)
                            .copy_from_slice(value.as_bytes());
                    }
                }
            }
        }

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
                            resource: image.out.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out.out.bind_active(),
                        },
                    ],
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
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let voxel_ctx = crate::voxel::Context::new(&gpu);
        let effects_ctx = crate::voxel::effects::Context::new(&gpu);

        let size = 128;
        let image_size = RenderSize::from(size);
        let mut buf = voxel_ctx.buffers();
        let mut merge_buf = effects_ctx.merge_buffers();

        let (x, y, z) = Tree::axes();
        let sphere =
            (x.square() + y.square() + z.square()).sqrt() - Tree::constant(0.5);
        let vm_shape = VmShape::from(sphere.min(z));
        let shape = gpu.shape(&vm_shape).unwrap();

        voxel_ctx
            .submit(
                &shape,
                &mut buf,
                None,
                &crate::voxel::RenderConfig {
                    image_size,
                    world_to_model: nalgebra::Matrix4::identity(),
                },
            )
            .unwrap();
        effects_ctx
            .submit_merge(buf.image_storage_buffer(), true, &mut merge_buf)
            .unwrap();
        let mut ssao_buf = effects_ctx.ssao_buffers();
        effects_ctx.submit_ssao(&merge_buf, &mut ssao_buf).unwrap();
        let ssao_out = gpu.read_vec::<f32>(ssao_buf.raw_occlusion().data());

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
