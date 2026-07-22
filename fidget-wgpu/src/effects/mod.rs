//! On-GPU effects
//!
//! These effects let us set up a simple rendering pipeline:
//!
//! - Start with [`GeometryPixel`](fidget_raster::voxel::GeometryPixel) buffers
//!   (16 bytes per pixel, stored on the GPU).
//! - Merge and denoise a set of buffers into a single image containing
//!   [`PackedVoxel`] data (normals, depth, and source image index packed into 8
//!   bytes per pixel).
//! - Compute an SSAO buffer from normal and depths in the combined image.
//! - If images have associated color functions, then render per-pixel diffuse
//!   colors based on position (including depth) and source image index;
//!   otherwise, compute diffuse color based on a single per-image color.
//! - Copy packed voxels, diffuse color, and SSAO data (previously stored in
//!   buffers) into textures.  These three textures can be used as inputs for a
//!   standard deferred rendering pipeline.

use crate::{
    BufferSizeError, ImageBuffer, buffer_ro, buffer_rw, buffer_uniform,
    usage::{COPY_DST_MAP_READ, STORAGE_COPY_SRC},
    voxel::ImageStorageBuffer,
};
use fidget_core::render::{ImageSize, VoxelSize};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// WGPU context for applying various effects
pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,

    merge_bind_group_layout: wgpu::BindGroupLayout,
    merge_pipeline: wgpu::ComputePipeline,

    shade_bind_group_layout: wgpu::BindGroupLayout,
    shade_pipeline: wgpu::ComputePipeline,
}

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const SHADE_SHADER: &str = include_str!("shaders/shade.wgsl");

fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + COMMON_SHADER
}

fn shade_shader() -> String {
    SHADE_SHADER.to_owned() + COMMON_SHADER
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
    normal: [i8; 2],

    /// Shape index
    index: u16,

    /// Depth of the voxel
    ///
    /// If this is 0, then the voxel is not populated
    z: u32,
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

/// Handle to a set of buffers used when merging images
pub struct MergeBuffers {
    config: wgpu::Buffer,
    out: ImageBuffer<PackedVoxel, STORAGE_COPY_SRC>,
    depth: u32,
}

/// Handle to a set of buffers used when shading images
pub struct ShadeBuffers {
    config: wgpu::Buffer,
    out: ImageBuffer<u32, STORAGE_COPY_SRC>,
}

/// Handle to a mapped image, which unmaps the image when dropped
pub struct MappedShadedImage<'a> {
    buf: &'a ImageBuffer<u32, COPY_DST_MAP_READ>,
    slice: wgpu::BufferSlice<'a>,
}

impl Drop for MappedShadedImage<'_> {
    fn drop(&mut self) {
        self.buf.data.unmap();
    }
}

impl MappedShadedImage<'_> {
    /// Returns the image's data
    pub fn image(&self) -> fidget_raster::Image<u32, ImageSize> {
        let result = <[u32]>::ref_from_bytes(&self.slice.get_mapped_range())
            .unwrap()
            .to_owned();
        fidget_raster::Image::build(result, self.buf.size).unwrap()
    }
}

/// Buffer for reading shaded data back from the GPU
///
/// This object is constructed by [`Context::shaded_read_buffer`] and may only
/// be used with that particular [`Context`].
///
/// Once mapped, this is wrapped by a [`MappedShadedImage`]
pub struct ShadedReadBuffer {
    buffer: ImageBuffer<u32, COPY_DST_MAP_READ>,
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
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let merge_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            });
        let shader_code = merge_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("effects merge pipeline"),
                bind_group_layouts: &[Some(&merge_bind_group_layout)],
                immediate_size: 0u32,
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("effects merge shader module"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let merge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("effects merge compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let shade_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_uniform(0),
                    buffer_ro(1), // image
                    buffer_rw(2), // out
                ],
            });
        let shader_code = shade_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("effects shade pipeline"),
                bind_group_layouts: &[Some(&shade_bind_group_layout)],
                immediate_size: 0u32,
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("effects shade shader module"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let shade_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("effects shade compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("shade_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            merge_bind_group_layout,
            merge_pipeline,
            shade_bind_group_layout,
            shade_pipeline,
        }
    }

    /// Builds a new set of [`MergeBuffers`] for the given image size
    pub fn merge_buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<MergeBuffers, BufferSizeError> {
        let config = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<MergeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = ImageBuffer::new(
            &self.device,
            "merge output".to_owned(),
            ImageSize::new(image_size.width(), image_size.height()),
        )?;
        Ok(MergeBuffers {
            config,
            out,
            depth: image_size.depth(),
        })
    }

    /// Builds a new set of [`MergeBuffers`] for the given image size
    pub fn shade_buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<ShadeBuffers, BufferSizeError> {
        let config = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade config"),
            size: std::mem::size_of::<ShadeConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = ImageBuffer::new(
            &self.device,
            "shade output".to_owned(),
            ImageSize::new(image_size.width(), image_size.height()),
        )?;
        Ok(ShadeBuffers { config, out })
    }

    /// Submits a set of merge operations to combine all of the images
    ///
    /// The output buffer is resized to fit the images
    ///
    /// If the incoming slice is empty, then no work is submitted
    pub fn submit_merge(
        &self,
        images: &[ImageStorageBuffer],
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
            .grow_to_fit(&self.device, size)
            .map_err(MergeError::OutputSize)?;
        let mut encoder = self.device.create_command_encoder(
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
                    denoise: 1,
                    index_base: i as u32 * 7,
                    image_count: chunk.len() as u32,
                    _pad: 0,
                };
                {
                    let mut writer = self
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
                        .bind(),
                };

                let bg =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    });
                compute_pass.set_bind_group(0, Some(&bg), &[]);
                compute_pass.dispatch_workgroups(
                    size.width().div_ceil(8),
                    size.height().div_ceil(8),
                    1,
                );
            }
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Submits an operation to shade an image
    ///
    /// The output buffer is resized to fit the images
    pub fn submit_shade(
        &self,
        image: &MergeBuffers,
        buf: &mut ShadeBuffers,
        out: Option<&mut ShadedReadBuffer>,
    ) -> Result<(), ShadeError> {
        let size = image.out.size;
        buf.out
            .grow_to_fit(&self.device, size)
            .map_err(ShadeError::OutputSize)?;
        println!("resizing to {size:?}");
        let mut encoder = self.device.create_command_encoder(
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
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            println!(
                "growing to fit {:?}, copying {}",
                buf.out.size,
                buf.out.size_bytes()
            );
            image.buffer.grow_to_fit(&self.device, buf.out.size).expect(
                "buf.out.size should always be \
                 a valid size for grow_to_fit",
            );
            encoder.copy_buffer_to_buffer(
                &buf.out.data,
                0,
                &image.buffer.data,
                0,
                buf.out.size_bytes(),
            );
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Returns an [`ShadedReadBuffer`], sized to read from a [`Buffers`] object
    pub fn shaded_read_buffer(
        &self,
        buffers: &ShadeBuffers,
    ) -> ShadedReadBuffer {
        let buffer = ImageBuffer::new(
            &self.device,
            "shaded image".to_owned(),
            buffers.out.size,
        )
        .expect(
            "buffers.out.size should always be \
             a valid size for ImageBuffer::new",
        );
        ShadedReadBuffer { buffer }
    }

    /// Maps a [`ShadedReadBuffer`] so that it can be read on the CPU
    pub fn map_shaded_image<'a>(
        &self,
        image: &'a mut ShadedReadBuffer,
    ) -> MappedShadedImage<'a> {
        let slice = image.buffer.map_async(|_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        MappedShadedImage {
            buf: &image.buffer,
            slice,
        }
    }
}

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
        for (src, desc) in
            [(merge_shader(), "merge"), (shade_shader(), "shade")]
        {
            crate::compile_shader(&src, desc);
        }
    }
}
