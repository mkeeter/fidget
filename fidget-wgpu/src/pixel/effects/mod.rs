use crate::{
    Gpu,
    RegPipeline,
    RenderShape,
    TAPE_DATA_CAPACITY,
    TapeWord,
    buf::{
        ArrayBuffer, BufferItemCount, BufferSizeError, BufferType, ImageBuffer,
        buffer_ro, buffer_ro_dyn, buffer_rw, buffer_uniform,
    },
    pixel::{PixelBufferTag, RawDistancePixel},
    shaders,
    tag,
    voxel::effects::MergeConfig, // same layout, unit-tested to confirm
};
use fidget_core::render::ImageSize;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub use crate::voxel::effects::{ImageSizeMismatch, MergeError};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

/// Returns a shader for merging images
fn merge_shader() -> String {
    MERGE_SHADER.to_owned()
        + COMMON_SHADER
        + shaders::COMMON
        + super::DISTANCE_PIXEL_SHADER
}

////////////////////////////////////////////////////////////////////////////////

/// Tagged pixel structure used on the GPU
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
pub struct TaggedRawDistancePixel {
    /// Distance value
    pub distance: RawDistancePixel,
    /// Shape index
    pub index: u32,
}

tag!(pub MergeVoxelBufferTag, TaggedRawDistancePixel, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU merged ([`TaggedRawDistancePixel`]) images"
);

/// Handle to a set of buffers used when merging images
pub struct MergeBuffers {
    config: wgpu::Buffer,
    out: ImageBuffer<MergeVoxelBufferTag>,
    image_count: usize,
}

////////////////////////////////////////////////////////////////////////////////

/// WGPU context for applying various effects
pub struct Context {
    gpu: Gpu,

    merge_bind_group_layout: wgpu::BindGroupLayout,
    merge_pipeline: wgpu::ComputePipeline,
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

        Self {
            gpu: gpu.clone(),
            merge_bind_group_layout,
            merge_pipeline,
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
        buf.image_count = images.len();
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
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_merge_shader() {
        crate::compile_shader(&merge_shader(), "merge");
    }

    #[test]
    fn merge_config_layout() {
        crate::test::compare_struct_layout::<MergeConfig>(
            &merge_shader(),
            "MergeConfig",
        );
    }
}
