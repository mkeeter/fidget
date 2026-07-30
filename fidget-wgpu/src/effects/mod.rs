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
    Gpu,
    buf::{
        BufferSizeError, BufferTag, ImageBuffer, ImageReadBuffer, MappedImage,
        buffer_ro, buffer_rw, buffer_uniform,
    },
    tag,
    voxel::GeomBufferTag,
};
use fidget_core::render::{ImageSize, VoxelSize};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// WGPU context for applying various effects
pub struct Context {
    gpu: Gpu,

    merge_bind_group_layout: wgpu::BindGroupLayout,
    merge_pipeline: wgpu::ComputePipeline,

    shade_bind_group_layout: wgpu::BindGroupLayout,
    shade_pipeline: wgpu::ComputePipeline,

    ssao_ctx: SsaoContext,
}

impl Gpu {
    /// Returns a readable buffer for the given image buffer
    pub fn read_buffer_for<T: BufferTag>(
        &self,
        buf: &ImageBuffer<T>,
    ) -> ImageReadBuffer<T> {
        ImageBuffer::new(
            &self.device,
            format!("{} (read)", buf.name()),
            buf.size(),
        )
        .expect("buf.size should always be a valid size for ImageBuffer::new")
    }

    /// Maps a readable image buffer, returning a mapped image
    pub fn map<'a, T: BufferTag>(
        &self,
        buf: &'a mut ImageReadBuffer<T>,
    ) -> MappedImage<'a, T> {
        MappedImage::map(&self.device, buf)
    }

    /// Debug function to read from a buffer to a `Vec<T>`
    pub fn read_vec<T: FromBytes + Immutable + Clone + Copy>(
        &self,
        buf: &wgpu::Buffer,
    ) -> Vec<T> {
        let scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buf.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &scratch, 0, buf.size());
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = scratch.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let result = <[T]>::ref_from_bytes(&buffer_slice.get_mapped_range())
            .unwrap()
            .to_vec();
        scratch.unmap();
        result
    }
}

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const SHADE_SHADER: &str = include_str!("shaders/shade.wgsl");
const SSAO_SHADER: &str = include_str!("shaders/ssao.wgsl");

fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + COMMON_SHADER + crate::COMMON_SHADER
}

fn shade_shader() -> String {
    SHADE_SHADER.to_owned() + COMMON_SHADER + crate::COMMON_SHADER
}

fn ssao_shader() -> String {
    SSAO_SHADER.to_owned() + COMMON_SHADER + crate::COMMON_SHADER
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

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
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

    /// Number of valid image buffers (0-7)
    image_count: u32,

    // padding to the nearest multiple of 8
    _pad: u32,
}

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
struct ShadeConfig {
    /// Image size, in pixels
    image_size: [u32; 3],
    _pad: u32,
}

tag!(MergeVoxelBufferTag, PackedVoxel, STORAGE | COPY_SRC);

/// Handle to a set of buffers used when merging images
pub struct MergeBuffers {
    config: wgpu::Buffer,
    out: ImageBuffer<MergeVoxelBufferTag>,
    depth: u32,
}

tag!(
    pub ShadedImageTag, u32, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU shaded (RGBA) images"
);

/// Handle to a set of buffers used when shading images
pub struct ShadeBuffers {
    config: wgpu::Buffer,
    out: ImageBuffer<ShadedImageTag>,
}

impl ShadeBuffers {
    /// Returns a reference to the output buffer
    pub fn output(&self) -> &ImageBuffer<ShadedImageTag> {
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
}

/// Type indicating an image size mismatch
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {}, got {} × {}",
    expected.width(), expected.height(),
    actual.width(), actual.height()
)]
pub struct ImageSizeMismatch {
    expected: ImageSize,
    actual: ImageSize,
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

        let shade_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_uniform(0),
                    buffer_ro(1), // image
                    buffer_rw(2), // out
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

        Self {
            gpu: gpu.clone(),
            merge_bind_group_layout,
            merge_pipeline,
            shade_bind_group_layout,
            shade_pipeline,
            ssao_ctx,
        }
    }

    /// Builds a new set of [`MergeBuffers`] for the given image size
    pub fn merge_buffers(
        &self,
        image_size: VoxelSize,
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
            ImageSize::new(image_size.width(), image_size.height()),
        )?;
        Ok(MergeBuffers {
            config,
            out,
            depth: image_size.depth(),
        })
    }

    /// Builds a new set of [`ShadeBuffers`] for the given image size
    pub fn shade_buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<ShadeBuffers, BufferSizeError> {
        let config = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade config"),
            size: std::mem::size_of::<ShadeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = ImageBuffer::new(
            &self.gpu.device,
            "shade output".to_owned(),
            ImageSize::new(image_size.width(), image_size.height()),
        )?;
        Ok(ShadeBuffers { config, out })
    }

    /// Builds a new set of [`MergeBuffers`] for the given image size
    pub fn ssao_buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<SsaoBuffers, BufferSizeError> {
        let config = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<SsaoConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = ImageBuffer::new(
            &self.gpu.device,
            "ssao output".to_owned(),
            ImageSize::new(image_size.width(), image_size.height()),
        )?;
        Ok(SsaoBuffers { config, out })
    }

    /// Submits a set of merge operations to combine all of the images
    ///
    /// The output buffer is resized to fit the images
    ///
    /// If the incoming slice is empty, then no work is submitted
    pub fn submit_merge(
        &self,
        images: &[&ImageBuffer<GeomBufferTag>],
        denoise: bool,
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
        buf.out
            .grow_to_fit(&self.gpu.device, size)
            .map_err(MergeError::OutputSize)?;
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        // Scope to bound the lifetime of compute_pass
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.merge_pipeline);
            for (i, chunk) in images.chunks(7).enumerate() {
                let cfg = MergeConfig {
                    image_size: [size.width(), size.height()],
                    denoise: denoise as u32,
                    index_base: i as u32 * 7,
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
            }
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
        buf: &mut ShadeBuffers,
        out: Option<&mut ImageReadBuffer<ShadedImageTag>>,
    ) -> Result<(), ShadeError> {
        let size = image.out.size();
        buf.out
            .grow_to_fit(&self.gpu.device, size)
            .map_err(ShadeError::OutputSize)?;
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        // Scope to bound the lifetime of compute_pass
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.shade_pipeline);
            let cfg = ShadeConfig {
                image_size: [size.width(), size.height(), image.depth],
                _pad: 0u32,
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
            image.grow_to_fit(&self.gpu.device, buf.out.size()).expect(
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
    ) -> Result<(), BufferSizeError> {
        self.ssao_ctx.submit(image, buf, &self.gpu)
    }
}

////////////////////////////////////////////////////////////////////////////////

tag!(pub SsaoBufferTag, f32, STORAGE | COPY_SRC,
    "tag for a raw SSAO occlusion buffer");

/// Handle to a set of buffers used when running an SSAO pass
pub struct SsaoBuffers {
    config: wgpu::Buffer,
    pub(crate) out: ImageBuffer<SsaoBufferTag>,
}

impl SsaoBuffers {
    /// Returns a handle to the raw SSAO occlusion buffer
    pub fn occlusion(&self) -> &ImageBuffer<SsaoBufferTag> {
        &self.out
    }
}

#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
struct SsaoConfig {
    /// Image size, in voxels
    image_size: [u32; 3],

    /// Whether or not to denoise when merging (non-zero is true)
    radius: f32,
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
    /*
    /// Pipeline for blurring an SSAO image
    blur_pipeline: wgpu::ComputePipeline,
    */
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
            .copy_from_slice(
                ssao_kernel_values.transpose().as_slice().as_bytes(),
            );

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
            .copy_from_slice(
                ssao_noise_values.transpose().as_slice().as_bytes(),
            );

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
        Self {
            ssao_bind_group,
            ssao_bind_group_layout: ssao_user_bind_group_layout,
            ssao_pipeline,
        }
    }

    fn submit(
        &self,
        image: &MergeBuffers,
        buf: &mut SsaoBuffers,
        gpu: &Gpu,
    ) -> Result<(), BufferSizeError> {
        let image_size = image.out.size();
        buf.out.grow_to_fit(&gpu.device, image_size)?;

        // TODO make this passed in?
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        // Scope to bound the lifetime of compute_pass
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None, // TODO add timestamps?
                });
            compute_pass.set_pipeline(&self.ssao_pipeline);
            let cfg = SsaoConfig {
                image_size: [
                    image_size.width(),
                    image_size.height(),
                    image.depth,
                ],
                radius: 0.1,
            };
            {
                let mut writer = gpu
                    .queue
                    .write_buffer_with(
                        &buf.config,
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
                        resource: buf.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image.out.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf.out.bind_active(),
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
        gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn packed_voxel_size() {
        assert_eq!(std::mem::size_of::<PackedVoxel>(), 8);
    }

    #[test]
    fn compile_shaders() {
        #[allow(clippy::single_element_loop)] // there will be more
        for (src, desc) in [
            (merge_shader(), "merge"),
            (shade_shader(), "shade"),
            (ssao_shader(), "ssao"),
        ] {
            crate::compile_shader(&src, desc);
        }
    }
}
