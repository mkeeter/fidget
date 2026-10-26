//! Color evaluation for 2D (pixel) images
use crate::{
    ConfigWorkspace, Gpu, RegPipeline,
    buf::{buffer_ro, buffer_rw},
    color::RenderColor,
    shaders,
};
use fidget_core::{
    render::ImageSize,
    shape::{MissingVar, ShapeVars},
};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

/// Workspace for evaluating color expressions
pub type Workspace = ConfigWorkspace<ColorConfig>;

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
pub struct Settings<'a> {
    /// Free variables used in evaluation
    pub vars: &'a ShapeVars<f32>,

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

static EMPTY_VARS: ShapeVars<f32> = ShapeVars::new();

impl Default for Settings<'_> {
    fn default() -> Self {
        Self {
            vars: &EMPTY_VARS,
            z: 0.0,
            only_filled: false,
            world_to_model: nalgebra::Matrix3::identity(),
        }
    }
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

    /// Shape color has already been populated
    #[error("shape color has already been populated")]
    AlreadyHasColor,
}

/// Context for color evaluation
pub struct Context {
    gpu: Gpu,

    /// Merge bind groups (read-write distance and color)
    ///
    /// This isn't used directly, but is part of constructing buffers.
    merge_bind_group_layout: wgpu::BindGroupLayout,

    /// Image bind groups (read-only distance, read-write color)
    color_bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for computing per-pixel color
    pipeline: RegPipeline,
}

impl crate::Context for Context {
    type Workspace = Workspace;
    type Buffers = super::merge::Buffers;
    type Settings<'a> = Settings<'a>;
    type Args<'a> = &'a RenderColor;
    type EncodeError = ColorError;
    type BuffersError = crate::buf::BufferSizeError;
    type Size = ImageSize;

    fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("color images"),
                entries: &[
                    buffer_ro(0), // distance
                    buffer_rw(1), // color
                ],
            });
        let merge_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("merge images"),
                entries: &[
                    buffer_rw(0), // distance
                    buffer_rw(1), // color
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("color pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.color_bind_group_layout),
                    Some(&color_bind_group_layout),
                ],
                immediate_size: 0u32,
            });
        let device_ = device.clone();
        let pipeline = RegPipeline::build(Box::new(move |reg_count| {
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
            gpu: gpu.clone(),
            color_bind_group_layout,
            merge_bind_group_layout,
            pipeline,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        shape: &RenderColor,
        settings: Settings,
        workspace: &mut Workspace,
        image: &mut super::merge::Buffers,
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
        shape.copy_vars(encoder, staging, settings.vars)?;

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
        compute_pass.set_pipeline(self.pipeline.get(shape.reg_count()));
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        image.has_color = true;
        Ok(())
    }

    /// Returns a new workspace for color evaluation
    fn workspace(&self) -> Workspace {
        self.gpu.config_workspace("color")
    }

    fn buffers(
        &self,
        image_size: ImageSize,
    ) -> Result<Self::Buffers, Self::BuffersError> {
        super::merge::Buffers::new(
            &self.gpu.device,
            image_size,
            &self.merge_bind_group_layout,
            &self.color_bind_group_layout,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_color_shader() {
        crate::compile_shader(&color_shader(16), "color");
    }

    #[test]
    fn color_config_layout() {
        crate::test::compare_struct_layout::<ColorConfig>(
            &color_shader(16),
            "Config",
        );
    }
}
