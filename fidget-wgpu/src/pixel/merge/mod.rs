//! Merging multiple 2D (pixel) images
use crate::{
    ConfigWorkspace, Gpu, ImageSizeMismatch,
    buf::BufferSizeError,
    buf::{FlexBuffer, buffer_ro, buffer_rw},
    shaders, tag,
};
use fidget_core::render::ImageSize;
use fidget_raster::pixel::RawDistancePixel;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[expect(unused_imports)] // used by rustdoc
use crate::Context as _;

const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

/// Returns a shader for merging images
fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + shaders::COMMON + super::DISTANCE_PIXEL_SHADER
}

/// Error returned when submitting a merge operation
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// Image sizes in the slice are not consistent
    #[error(transparent)]
    ImageSizeMismatch(#[from] ImageSizeMismatch),
}

/// Handle to a workspace for image merge operations
pub type Workspace = ConfigWorkspace<MergeConfig>;

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

/// Handle to size-dependent buffers for merge operations
pub struct Buffers {
    pub(crate) distance: FlexBuffer<PixelDistanceBufferTag>,
    color: FlexBuffer<PixelColorBufferTag>,

    /// Bind group to merge buffers
    merge_bind_group: wgpu::BindGroup,

    /// Bind group to do color evaluation
    pub(crate) color_bind_group: wgpu::BindGroup,

    /// Number of images merged together
    pub(crate) image_count: usize,

    /// Indicates whether the `color` buffer represents color
    ///
    /// When this is `false`, the `color` buffer represents shape index instead,
    /// and [`output_color`](Self::output_color) returns `None`.
    pub(crate) has_color: bool,
}

impl Buffers {
    pub(crate) fn new(
        device: &wgpu::Device,
        image_size: ImageSize,
        merge_bind_group_layout: &wgpu::BindGroupLayout,
        color_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, BufferSizeError> {
        let distance = FlexBuffer::new(device, "merge distance", image_size)?;
        let color = FlexBuffer::new(device, "merge color", image_size)?;
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
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge bind group"),
                layout: merge_bind_group_layout,
                entries,
            });
        let color_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("color bind group"),
                layout: color_bind_group_layout,
                entries,
            });
        Ok(Buffers {
            distance,
            color,
            merge_bind_group,
            color_bind_group,
            image_count: 0,
            has_color: false,
        })
    }

    /// Resets the merge buffer
    ///
    /// The next call to [`Context::submit`] will clear the buffer and
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

/// Settings for merge operations
pub struct Settings {
    remove_nans: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { remove_nans: true }
    }
}

/// WGPU context for merging images
///
/// Note that [`Buffers::reset`] should be called before the first call to
/// `submit` or `encode`. The [`Buffers`] object must match the image size;
/// otherwise an error will be returned.
pub struct Context {
    gpu: Gpu,

    /// Bind group layout for merge operations
    merge_bind_group_layout: wgpu::BindGroupLayout,

    /// Bind group layout for color operations
    ///
    /// This isn't used in our pipelines, but is part of [`Buffers`] construction
    color_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl crate::Context for Context {
    type Workspace = Workspace;
    type Buffers = Buffers;
    type BuffersError = BufferSizeError;
    type EncodeError = MergeError;
    type Settings<'a>
        = Settings
    where
        Self: 'a;
    type Args<'a> = &'a super::render::Buffers;
    type Size = ImageSize;

    fn new(gpu: &Gpu) -> Self {
        let merge_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // distance
                    buffer_rw(1), // color
                ],
            },
        );
        let color_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // distance
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
        let pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("pixel effects merge compute pipeline"),
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
            color_bind_group_layout,
            pipeline,
        }
    }

    fn encode(
        &self,
        image: &super::render::Buffers,
        settings: Settings,
        workspace: &mut Workspace,
        buf: &mut Buffers,
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
            remove_nans: settings.remove_nans as u32,
            index_base: buf.image_count as u32,
        };
        workspace.encode_copy(encoder, staging, &cfg);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.pipeline);

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

    fn workspace(&self) -> Workspace {
        self.gpu.config_workspace("merge")
    }

    fn buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<Buffers, BufferSizeError> {
        Buffers::new(
            &self.gpu.device,
            image_size,
            &self.merge_bind_group_layout,
            &self.color_bind_group_layout,
        )
    }

    fn gpu(&self) -> &Gpu {
        &self.gpu
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
