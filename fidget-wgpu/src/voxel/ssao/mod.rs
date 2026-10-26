//! Evaluation of screen-space ambient occlusion
use crate::{
    ConfigWorkspace, Gpu, ImageSizeMismatch,
    buf::BufferSizeError,
    buf::{FlexBuffer, buffer_ro, buffer_rw},
    tag,
};
use fidget_core::render::ImageSize;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Context for computing screen-space ambient occlusion
pub struct Context {
    gpu: Gpu,

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

const SSAO_SHADER: &str = include_str!("shaders/ssao.wgsl");
const BLUR_SHADER: &str = include_str!("shaders/blur.wgsl");

fn ssao_shader() -> String {
    SSAO_SHADER.to_owned() + super::PACKED_VOXEL + crate::shaders::COMMON
}

fn blur_shader() -> String {
    BLUR_SHADER.to_owned() + super::PACKED_VOXEL + crate::shaders::COMMON
}

impl crate::Context for Context {
    type Workspace = Workspace;
    type BuffersError = BufferSizeError;
    type Buffers = Buffers;
    type EncodeError = SsaoError;
    type Args<'a> = &'a super::merge::Buffers;
    type Settings<'a> = ();
    type Size = ImageSize;

    fn new(gpu: &Gpu) -> Self {
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
            gpu: gpu.clone(),
            ssao_bind_group,
            ssao_pipeline,
            blur_bind_group_layout,
            blur_pipeline,
        }
    }

    fn encode(
        &self,
        image: &super::merge::Buffers,
        _settings: (),
        workspace: &mut Workspace,
        buf: &mut Buffers,
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

    fn workspace(&self) -> Workspace {
        let ssao = self.gpu.config_workspace("ssao");
        let blur = self.gpu.config_workspace("blur");
        Workspace { ssao, blur }
    }

    fn buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<Buffers, BufferSizeError> {
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
                    layout: &self.blur_bind_group_layout,
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
        Ok(Buffers {
            raw_occlusion,
            blurred_occlusion,
            ssao_bind_group,
            blur_bind_group,
            output_bind_group,
        })
    }
}

tag!(pub SsaoRawBufferTag, f32, ImageSize, STORAGE | COPY_SRC,
    "Tag for a raw SSAO occlusion buffer");
tag!(pub SsaoBlurredBufferTag, f32, ImageSize, STORAGE | COPY_SRC,
    "Tag for a blurred SSAO occlusion buffer");

/// Handle to a set of configuration buffers used when running an SSAO pass
pub struct Workspace {
    ssao: ConfigWorkspace<SsaoConfig>,
    blur: ConfigWorkspace<BlurConfig>,
}

/// Handle to a set of image buffers used when running an SSAO pass
///
/// Unlike [`Workspace`], these buffers are size-dependent
pub struct Buffers {
    /// Raw occlusion buffer
    pub(crate) raw_occlusion: FlexBuffer<SsaoRawBufferTag>,

    /// Bind group used in the SSAO pass (writing to `raw_occlusion`)
    pub(crate) ssao_bind_group: wgpu::BindGroup,

    /// Blurred occlusion buffer
    pub(crate) blurred_occlusion: FlexBuffer<SsaoBlurredBufferTag>,

    /// Bind group used in the blur pass
    ///
    /// This pass reads from `raw_occlusion` and writes to `blurred_occlusion`
    pub(crate) blur_bind_group: wgpu::BindGroup,

    /// Bind group which exposes the blurred occlusion buffer (read-only)
    pub(crate) output_bind_group: wgpu::BindGroup,
}

impl Buffers {
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

/// Error returned when submitting an SSAO operation
#[derive(Debug, thiserror::Error)]
pub enum SsaoError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(#[from] ImageSizeMismatch),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Context as _, voxel::render::RenderSize};
    use fidget_core::{context::Tree, vm::VmShape};

    /// Render a sphere-plane union and check for occlusion bias
    ///
    /// Because the image is perfectly symmetrical, we'd expect the average
    /// occlusion across each of the four corners to be very similar.  If it's
    /// not, then that's likely a sampling bias – which we have seen before!
    #[test]
    fn ssao_bias() {
        crate::require_gpu!();
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let voxel_ctx = crate::voxel::render::Context::new(&gpu);
        let mut voxel_workspace = voxel_ctx.workspace();
        let merge_ctx = crate::voxel::merge::Context::new(&gpu);
        let mut merge_workspace = merge_ctx.workspace();
        let ssao_ctx = crate::voxel::ssao::Context::new(&gpu);

        let size = 128;
        let image_size = RenderSize::from(size);
        let mut voxel_buf = voxel_ctx.buffers(image_size).unwrap();
        let mut merge_buf = merge_ctx.buffers(image_size).unwrap();

        let (x, y, z) = Tree::axes();
        let sphere =
            (x.square() + y.square() + z.square()).sqrt() - Tree::constant(0.5);
        let vm_shape = VmShape::from(sphere.min(z));
        let shape = gpu.build_render_shape(&vm_shape).unwrap();

        voxel_ctx
            .submit(
                &gpu,
                &shape,
                Default::default(),
                &mut voxel_workspace,
                &mut voxel_buf,
            )
            .unwrap();
        merge_ctx
            .submit(
                &gpu,
                &voxel_buf,
                Default::default(),
                &mut merge_workspace,
                &mut merge_buf,
            )
            .unwrap();
        let mut ssao_workspace = ssao_ctx.workspace();
        let mut ssao_buf = ssao_ctx.buffers(image_size.into()).unwrap();
        ssao_ctx
            .submit(&gpu, &merge_buf, (), &mut ssao_workspace, &mut ssao_buf)
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
    fn compile_ssao_shader() {
        crate::compile_shader(&ssao_shader(), "ssao");
    }

    #[test]
    fn compile_blur_shader() {
        crate::compile_shader(&blur_shader(), "blur");
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
