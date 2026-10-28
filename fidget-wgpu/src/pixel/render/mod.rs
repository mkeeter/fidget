//! GPU-accelerated 2D rendering
//!
//! See the [`voxel`](crate::voxel) module for details docs; this module is
//! analogous (down to the naming of types).
use crate::{
    BadCopySize, Context as _, Gpu, RegPipeline, RenderShape,
    TAPE_DATA_CAPACITY, TapeWord,
    buf::{
        BufferSizeError, BufferType, FlexBuffer, ReadBuffer, buffer_ro,
        buffer_rw,
    },
    shaders, tag,
};
use fidget_core::{
    render::ImageSize,
    shape::{MissingVar, ShapeVars},
};
use fidget_raster::pixel::{Image, RawDistancePixel};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub use fidget_raster::pixel::RenderSize;

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const INTERVAL_INPUT: &str = include_str!("shaders/interval_input.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const PIXEL_TILES_SHADER: &str = include_str!("shaders/pixel_tiles.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

/// Returns a shader for interval root tiles
fn interval_root_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += COMMON_SHADER;
    shader_code += INTERVAL_ROOT_SHADER;
    shader_code += INTERVAL_INPUT;
    shader_code += super::DISTANCE_PIXEL_SHADER;
    shader_code += super::TRANSFORM_INPUT;
    shader_code += shaders::INTERVAL_OPS;
    shader_code += shaders::COMMON;
    shader_code += shaders::TAPE_INTERPRETER;
    shader_code += shaders::STACK;
    shader_code += shaders::TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for interval tile -> subtile reduction
fn interval_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += COMMON_SHADER;
    shader_code += INTERVAL_TILES_SHADER;
    shader_code += INTERVAL_INPUT;
    shader_code += super::DISTANCE_PIXEL_SHADER;
    shader_code += super::TRANSFORM_INPUT;
    shader_code += shaders::INTERVAL_OPS;
    shader_code += shaders::COMMON;
    shader_code += shaders::TAPE_INTERPRETER;
    shader_code += shaders::STACK;
    shader_code += shaders::TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for pixel tile evaluation
fn pixel_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += PIXEL_TILES_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += super::TRANSFORM_INPUT;
    shader_code += super::DISTANCE_PIXEL_SHADER;
    shader_code += shaders::FLOAT_OPS;
    shader_code += shaders::COMMON;
    shader_code += shaders::TAPE_INTERPRETER;
    shader_code += shaders::DUMMY_STACK;
    shader_code
}

/// Returns a shader for merging images
fn merge_shader() -> String {
    MERGE_SHADER.to_owned()
        + COMMON_SHADER
        + super::DISTANCE_PIXEL_SHADER
        + shaders::COMMON
}

////////////////////////////////////////////////////////////////////////////////

/// Optional settings for 2D rendering
#[derive(Copy, Clone)]
pub struct Settings<'a> {
    /// Values for free variables
    pub vars: &'a ShapeVars<f32>,

    /// World-to-model transform
    pub world_to_model: nalgebra::Matrix3<f32>,

    /// Render the distance values of individual pixels
    pub pixel_perfect: bool,

    /// Z position at which to render the slice
    pub z: f32,
}

impl Default for Settings<'_> {
    fn default() -> Self {
        Self {
            vars: &crate::EMPTY_VARS,
            world_to_model: nalgebra::Matrix3::identity(),
            pixel_perfect: false,
            z: 0.0,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// A render size is rounded up to the next multiple of 64 on every axis
///
/// The internal `ImageSize` stores divided-by-64 values, so that the render
/// size cannot be constructed with an invalid state.
#[derive(Copy, Clone, Debug)]
struct TileRenderSize(ImageSize);

impl From<ImageSize> for TileRenderSize {
    fn from(image_size: ImageSize) -> Self {
        let nx = image_size.width().div_ceil(64);
        let ny = image_size.height().div_ceil(64);
        Self(ImageSize::new(nx, ny))
    }
}

impl TileRenderSize {
    /// Number of tiles in the X axis
    fn nx(&self) -> u32 {
        self.0.width()
    }

    /// Number of tiles in the Y axis
    fn ny(&self) -> u32 {
        self.0.height()
    }

    /// Number of voxels in the X axis (always a multiple of 64)
    fn width(&self) -> u32 {
        self.0.width() * 64
    }

    /// Number of voxels in the Y axis (always a multiple of 64)
    fn height(&self) -> u32 {
        self.0.height() * 64
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Root context, which produces a list of 64² tiles
struct RootContext {
    /// Pipelines for 64² tile evaluation
    root_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl RootContext {
    fn new(gpu: &Gpu) -> Self {
        // Create bind group layout and bind group
        let device = &gpu.device;
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tile_tapes
                    buffer_rw(1), // tiles_out
                    buffer_rw(2), // tile_values
                ],
            });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(&gpu.output_bind_group_layout), // config + tape
                    Some(&gpu.input_bind_group_layout),  // vars
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        let device_ = device.clone();
        let root_pipeline = RegPipeline::build(Box::new(move |reg_count| {
            let shader_code = interval_root_shader(reg_count);
            let shader_module =
                device_.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device_.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval root ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_root_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }));

        Self {
            bind_group_layout,
            root_pipeline,
        }
    }

    fn run(
        &self,
        buffers: &Buffers,
        reg_count: u8,
        render_size: TileRenderSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        compute_pass.set_pipeline(self.root_pipeline.get(reg_count));
        compute_pass.set_bind_group(2, &buffers.bind_groups.root_tiles, &[]);

        // Workgroup is 8x8x8, so we divide by 8 here on each axis
        let nx = render_size.nx().div_ceil(8);
        let ny = render_size.ny().div_ceil(8);
        compute_pass.dispatch_workgroups(nx, ny, 1);
    }
}
////////////////////////////////////////////////////////////////////////////////

/// Intermediate tiles context, which produces a list of 8² tiles
struct IntervalTilesContext {
    /// Pipelines for 8² tile evaluation
    tiles_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IntervalTilesContext {
    fn new(gpu: &Gpu) -> Self {
        // Create bind group layout and bind group
        let device = &gpu.device;
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tile_tapes
                    buffer_ro(1), // tiles_in
                    buffer_rw(2), // subtiles_out
                    buffer_rw(3), // subtile_values
                ],
            });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(&gpu.output_bind_group_layout), // config + tape
                    Some(&gpu.input_bind_group_layout),  // vars
                    Some(&bind_group_layout),            // vars
                ],
                immediate_size: 0u32,
            });

        let device_ = device.clone();
        let tiles_pipeline = RegPipeline::build(Box::new(move |reg_count| {
            let shader_code = interval_tiles_shader(reg_count);
            let shader_module =
                device_.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device_.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval tiles ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tiles_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }));

        Self {
            bind_group_layout,
            tiles_pipeline,
        }
    }

    fn run(
        &self,
        buffers: &Buffers,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        compute_pass.set_pipeline(self.tiles_pipeline.get(reg_count));
        compute_pass.set_bind_group(
            2,
            &buffers.bind_groups.interval_tiles,
            &[],
        );

        // Indirect dispatch based on previous tile output
        compute_pass
            .dispatch_workgroups_indirect(buffers.tile64.tiles.data(), 0);
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Pixel tiles context, which outputs individual pixels
struct PixelTilesContext {
    /// Pipelines for 8² pixel tile evaluation
    tiles_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl PixelTilesContext {
    fn new(gpu: &Gpu) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tile_tapes
                    buffer_ro(1), // tiles_in
                    buffer_rw(2), // result
                ],
            },
        );
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(&gpu.output_bind_group_layout), // config + tape
                    Some(&gpu.input_bind_group_layout),  // vars
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            },
        );

        let device_ = gpu.device.clone();
        let tiles_pipeline = RegPipeline::build(Box::new(move |reg_count| {
            let shader_code = pixel_tiles_shader(reg_count);
            let shader_module =
                device_.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device_.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("pixel tiles ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("pixel_tiles_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }));

        Self {
            bind_group_layout,
            tiles_pipeline,
        }
    }

    fn run(
        &self,
        buffers: &Buffers,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        compute_pass.set_pipeline(self.tiles_pipeline.get(reg_count));
        compute_pass.set_bind_group(2, &buffers.bind_groups.pixel_tiles, &[]);

        // Indirect dispatch based on previous tile output
        compute_pass
            .dispatch_workgroups_indirect(buffers.tile8.tiles.data(), 0);
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Doppelganger of the WGSL `struct Config`
///
/// Fields are carefully ordered to require no internal padding (enforced by
/// `zerocopy` derives)
#[derive(Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
struct Config {
    /// Screen-to-model transform matrix
    ///
    /// This is a 3x3 matrix in WGSL, but each row is padded to 16 bytes, so
    /// it's a total of 12 floats.
    mat: [f32; 12],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    /// Initial offset in `tape_data`
    tape_data_offset: u32,

    /// Render size, rounded up to the nearest multiple of 64
    render_size: [u32; 2],

    /// Image size (not rounded)
    image_size: [u32; 2],

    /// Z position at which to render
    z: f32,

    /// Flag indicating whether to recurse down to individual pixels
    pixel_perfect: u32,

    /// Number of words in the trailing tape buffer
    tape_data_capacity: u32,

    /// Padding
    _pad: u32,
    // This is followed by a flexible array member containing tape data
}

tag!(TileTapesBufferTag, u32, usize, STORAGE | COPY_DST);
tag!(pub PixelBufferTag, RawDistancePixel, ImageSize, STORAGE | COPY_SRC | COPY_DST,
    "Tag for a on-GPU buffer storing [`RawDistancePixel`] values");

/// Size-dependent buffers used in rendering
///
/// This object is constructed by [`Context::buffers`] and may only be used with
/// that particular [`Context`].
pub struct Buffers {
    /// Image render size
    ///
    /// Note that the tile buffers below round up to the nearest root tile
    /// (64² voxels).
    image_size: ImageSize,

    /// Map from tile to the relevant tape (as a start index)
    tile_tapes: FlexBuffer<TileTapesBufferTag>,

    /// Root tile buffers (64²)
    tile64: TileBuffers<64>,

    /// Second-stage tile buffers (8²)
    tile8: TileBuffers<8>,

    /// Pixel data
    pub(crate) pixels: FlexBuffer<PixelBufferTag>,

    /// Bind groups
    pub(crate) bind_groups: BindGroups,
}

impl Buffers {
    /// Builds a new set of buffers for a particular render size
    fn new(ctx: &Context, image_size: ImageSize) -> Result<Self, BuffersError> {
        let device = &ctx.gpu.device;
        let render_size = TileRenderSize::from(image_size);
        let tile_tapes = FlexBuffer::new(
            device,
            "tile tape",
            Self::tile_tapes_buf_size(render_size),
        )
        .map_err(|err| BuffersError {
            buf: BufferName::TileTapes,
            err,
        })?;
        let pixels =
            FlexBuffer::new(device, "pixels", image_size).map_err(|err| {
                BuffersError {
                    buf: BufferName::Pixel,
                    err,
                }
            })?;
        let tile64 = TileBuffers::new(device, render_size).map_err(|err| {
            BuffersError {
                buf: BufferName::Tile64(err.buf),
                err: err.err,
            }
        })?;
        let tile8 = TileBuffers::new(device, render_size).map_err(|err| {
            BuffersError {
                buf: BufferName::Tile8(err.buf),
                err: err.err,
            }
        })?;

        let bind_groups = {
            let root_tiles =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("interval root bind group"),
                    layout: &ctx.root_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: tile_tapes.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: tile64.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: tile64.values.bind_active(),
                        },
                    ],
                });
            let interval_tiles =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("interval tiles bind group"),
                    layout: &ctx.tiles_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: tile_tapes.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: tile64.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: tile8.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: tile8.values.bind_active(),
                        },
                    ],
                });

            let merge = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge bind group"),
                layout: &ctx.merge_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tile_tapes.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile64.values.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile8.values.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pixels.bind_active(),
                    },
                ],
            });
            let pixel_tiles =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pixel tiles bind group"),
                    layout: &ctx.pixels_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: tile_tapes.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: tile8.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pixels.bind_active(),
                        },
                    ],
                });

            let output = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("output bind group"),
                layout: &ctx.gpu.input_bind_group_layout, // read-only
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixels.bind_active(),
                }],
            });
            BindGroups {
                root_tiles,
                interval_tiles,
                merge,
                pixel_tiles,
                output,
            }
        };

        Ok(Self {
            image_size,
            tile_tapes,
            tile64,
            tile8,
            pixels,
            bind_groups,
        })
    }

    /// Returns the number of bytes in the `tile_tapes` buffer
    ///
    /// This is two levels of densely-allocated tiles; the first set are
    /// 64² and the second are 8².
    ///
    /// In other words, it looks something like this:
    ///
    /// ```text
    /// | index | index | index | ... |     densely packed 64² tape indices
    /// | index | index | index | ... |     densely packed 8² tape indices
    /// ```
    fn tile_tapes_buf_size(render_size: TileRenderSize) -> usize {
        let nx = usize::try_from(render_size.nx()).unwrap();
        let ny = usize::try_from(render_size.ny()).unwrap();

        // Total size computation:
        //    (nx * ny) + (nx * ny * 64)
        // => nx * ny * 65
        nx.checked_mul(ny).unwrap().checked_mul(65).unwrap()
    }

    /// Returns a handle to the image storage buffer
    ///
    /// This is intended for subsequent shaders which want to use the
    /// [`RawDistancePixel`] image data without copying to the CPU.  It requires
    /// an exclusive borrow of the `Workspace` object (and then extends that
    /// lifetime) so that other callers can't simultaneously touch the buffer.
    pub fn output(&mut self) -> &FlexBuffer<PixelBufferTag> {
        &self.pixels
    }
}

/// Configuration workspace buffers used in pixel rendering
pub struct Workspace {
    /// Config and tape data buffer (constant size)
    config_buf: wgpu::Buffer,
    /// Bind group which binds the config data buffer
    bind_group: wgpu::BindGroup,
}

impl Workspace {
    fn new(ctx: &Context) -> Self {
        // The config buffer is statically sized, so we can check it here
        static_assertions::const_assert!(
            (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>())
                as u64
                <= BufferType::Storage.max_size()
        );

        let device = &ctx.gpu.device;
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("config"),
            layout: &ctx.gpu.output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            }],
        });
        Workspace {
            config_buf,
            bind_group,
        }
    }
}

/// Error returned when creating buffers in a [`Workspace`] object
#[derive(Debug, thiserror::Error)]
#[error("failed to build {buf} buffer")]
pub struct BuffersError {
    /// Buffer which failed to be created
    pub buf: BufferName,
    /// Error returned by buffer creation
    #[source]
    pub err: BufferSizeError,
}

/// Names of all buffers, used for error reporting
#[derive(Debug)]
pub enum BufferName {
    /// Tiles from the 64² root tile pass
    Tile64(TileBufferName),
    /// Tiles from the 8² intermediate tile pass
    Tile8(TileBufferName),
    /// Buffer for tile tapes
    TileTapes,
    /// GPU-written image pixels (as [`RawDistancePixel`] values)
    Pixel,
    /// CPU-mappable image pixels (as [`RawDistancePixel`] values)
    Image,
    /// Variables
    Vars,
}

impl std::fmt::Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferName::Tile64(buf) => write!(f, "`{buf}` tile64"),
            BufferName::Tile8(buf) => write!(f, "`{buf}` tile8"),
            BufferName::TileTapes => write!(f, "`tile tapes`"),
            BufferName::Pixel => write!(f, "`pixel`"),
            BufferName::Image => write!(f, "`image`"),
            BufferName::Vars => write!(f, "`vars`"),
        }
    }
}

/// Error returned when submitting a voxel rasterization job to the GPU
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// Missing variable when evaluating
    #[error(transparent)]
    MissingVar(#[from] MissingVar),
    /// Invalid buffer size
    #[error(transparent)]
    BadBufferSize(#[from] BadCopySize),
}

/// Cached bind groups (constructed on-demand)
pub(crate) struct BindGroups {
    root_tiles: wgpu::BindGroup,
    interval_tiles: wgpu::BindGroup,
    pixel_tiles: wgpu::BindGroup,
    merge: wgpu::BindGroup,

    /// Bind group which exposes a single read-only image (storage) buffer
    ///
    /// This bind group is not used directly in pixel rendering, but is stored
    /// here because it's logically associated with the [`Buffers`] object.
    pub(crate) output: wgpu::BindGroup,
}

////////////////////////////////////////////////////////////////////////////////

tag!(TilesBufferTag, u32, usize, STORAGE | COPY_DST | INDIRECT);
tag!(ValuesBufferTag, u32, ImageSize, STORAGE | COPY_DST);

/// Root tile buffers store strata-packed tile lists
struct TileBuffers<const N: usize> {
    /// Output tiles
    tiles: FlexBuffer<TilesBufferTag>,

    /// Tile values (empty / full)
    values: FlexBuffer<ValuesBufferTag>,
}

/// Error type when building root tile buffers
#[derive(Debug, thiserror::Error)]
#[error("failed to create `{buf}` root tile buffer")]
pub struct TileBuffersError {
    /// Buffer which failed to be created
    pub buf: TileBufferName,
    /// Error returned by buffer creation
    #[source]
    pub err: BufferSizeError,
}

/// Names of buffers used by the root tile rendering pass (for error reporting)
#[derive(Debug)]
#[expect(missing_docs)]
pub enum TileBufferName {
    Tiles,
    Values,
}

impl std::fmt::Display for TileBufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TileBufferName::Tiles => "tiles",
            TileBufferName::Values => "values",
        };
        s.fmt(f)
    }
}

impl<const N: usize> TileBuffers<N> {
    /// Build a new root tiles buffer, which stores strata-packed tile lists
    fn new(
        device: &wgpu::Device,
        render_size: TileRenderSize,
    ) -> Result<Self, TileBuffersError> {
        // Allocate enough words to write all of the output tiles
        let tiles = FlexBuffer::new(
            device,
            format!("tiles_out{N}"),
            Self::tiles_buf_size(render_size),
        )
        .map_err(|err| TileBuffersError {
            buf: TileBufferName::Tiles,
            err,
        })?;

        let values_buf_size = Self::values_buf_size(render_size);
        let values =
            FlexBuffer::new(device, format!("tile{N}_values"), values_buf_size)
                .map_err(|err| TileBuffersError {
                    buf: TileBufferName::Values,
                    err,
                })?;
        Ok(Self { tiles, values })
    }

    fn tiles_buf_size(render_size: TileRenderSize) -> usize {
        let nx = usize::try_from(render_size.nx()).unwrap();
        let ny = usize::try_from(render_size.ny()).unwrap();
        // wg_dispatch: [u32; 3] (unused)
        // count: u32,
        4 + nx
            .checked_mul(ny)
            .unwrap()
            .checked_mul((64 / N) * (64 / N))
            .unwrap()
    }

    fn values_buf_size(render_size: TileRenderSize) -> ImageSize {
        ImageSize::new(
            render_size.nx().checked_mul(64 / N as u32).unwrap(),
            render_size.ny().checked_mul(64 / N as u32).unwrap(),
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 2D (distance field) rendering
pub struct Context {
    gpu: Gpu,

    /// Context for root tile evaluation (64²)
    root_ctx: RootContext,

    /// Context for second-stage tile evaluation (generating 8² tiles)
    tiles_ctx: IntervalTilesContext,

    /// Context for per-pixel evaluation (taking 8² tiles)
    pixels_ctx: PixelTilesContext,

    /// Context which resets buffers before evaluation
    reset_ctx: ResetContext,

    /// Context to merge tile and pixel images
    merge_ctx: MergeContext,
}

impl crate::Context for Context {
    type Workspace = Workspace;
    type Buffers = Buffers;
    type BuffersError = BuffersError;
    type EncodeError = MissingVar;
    type Size = ImageSize;
    type Args<'a> = &'a RenderShape;
    type Settings<'a> = Settings<'a>;

    fn new(gpu: &Gpu) -> Self {
        let root_ctx = RootContext::new(gpu);
        let tiles_ctx = IntervalTilesContext::new(gpu);
        let pixels_ctx = PixelTilesContext::new(gpu);
        let merge_ctx = MergeContext::new(gpu);

        Self {
            gpu: gpu.clone(),
            root_ctx,
            tiles_ctx,
            pixels_ctx,
            merge_ctx,
            reset_ctx: ResetContext,
        }
    }

    fn workspace(&self) -> Workspace {
        Workspace::new(self)
    }

    fn buffers(&self, size: ImageSize) -> Result<Buffers, BuffersError> {
        Buffers::new(self, size)
    }

    fn encode(
        &self,
        shape: &RenderShape,
        settings: Settings<'_>,
        workspace: &mut Workspace,
        buffers: &mut Buffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), MissingVar> {
        let render_size = TileRenderSize::from(buffers.image_size);

        // The WebGPU config type has a mat3x3f, but that type pads each row to
        // 16 bytes, so we'll just use a mat4x3 for simplicity
        let mat =
            settings.world_to_model * buffers.image_size.screen_to_world();
        let mut mat4 = nalgebra::Matrix4x3::<f32>::identity();
        mat4.fixed_view_mut::<3, 3>(0, 0).copy_from(&mat);

        // Divide by 2 to go from `u32` -> `TapeWord`
        let start_offset = shape.bytecode_len.try_into().unwrap();
        let config = Config {
            mat: mat4.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            render_size: [render_size.width(), render_size.height()],
            tape_data_capacity: TAPE_DATA_CAPACITY.try_into().unwrap(),
            image_size: [
                buffers.image_size.width(),
                buffers.image_size.height(),
            ],
            tape_data_offset: start_offset,
            z: settings.z,
            pixel_perfect: settings.pixel_perfect as u32,
            _pad: 0,
        };

        // We load the `Config` and shape tape data.
        let config_len = u64::try_from(std::mem::size_of_val(&config)).unwrap();
        let mut writer = staging.write_buffer(
            encoder,
            &workspace.config_buf,
            0,
            (config_len as u64).try_into().unwrap(),
        );
        writer
            .slice(..usize::try_from(config_len).unwrap())
            .copy_from_slice(config.as_bytes());

        // Encode a buffer-to-buffer copy to load the tape
        encoder.copy_buffer_to_buffer(
            &shape.tape,
            0u64,
            &workspace.config_buf,
            config_len,
            shape.tape.size(),
        );

        // Copy vars (if present)
        shape.copy_vars(encoder, staging, settings.vars)?;

        // Initial buffer reset pass
        self.reset_ctx.run(encoder, buffers);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        // These bind groups are used by every compute shader
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, &shape.vars_bind_group, &[]);

        // Populate root tiles (64x64x64, densely packed)
        self.root_ctx.run(
            buffers,
            shape.reg_count,
            render_size,
            &mut compute_pass,
        );
        self.tiles_ctx
            .run(buffers, shape.reg_count, &mut compute_pass);
        self.pixels_ctx
            .run(buffers, shape.reg_count, &mut compute_pass);

        // Merge filled tiles from large -> small
        self.merge_ctx.run(buffers, &mut compute_pass);
        drop(compute_pass);
        Ok(())
    }

    fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

impl Context {
    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is not present when built for the `wasm32` target
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run<'a>(
        &self,
        shape: &'a RenderShape,
        settings: Settings<'a>,
        workspace: &mut Workspace,
        buf: &mut Buffers,
        out: &mut ReadBuffer<PixelBufferTag>,
    ) -> Result<Image, RunError> {
        self.submit(shape, settings, workspace, buf)?;
        self.gpu.copy(buf.output(), out)?;
        let image = self.gpu.map_image(out);
        Ok(image.image())
    }

    /// Renders the image, with an async wait to read pixel data from the GPU
    pub async fn run_async<'a>(
        &self,
        shape: &'a RenderShape,
        settings: Settings<'a>,
        workspace: &mut Workspace,
        buf: &mut Buffers,
        out: &mut ReadBuffer<PixelBufferTag>,
    ) -> Result<Image, RunError> {
        self.submit(shape, settings, workspace, buf)?;
        self.gpu.copy(buf.output(), out)?;
        let image = self.gpu.map_image_async(out).await;
        Ok(image.image())
    }
}

////////////////////////////////////////////////////////////////////////////////

struct MergeContext {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl MergeContext {
    fn new(gpu: &Gpu) -> Self {
        let shader_code = merge_shader();

        // Create bind group layout and bind group
        let bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("merge bind group layout"),
                entries: &[
                    buffer_rw(0), // tile_tapes
                    buffer_ro(1), // tile64_values
                    buffer_ro(2), // tile8_values
                    buffer_rw(3), // pixels
                ],
            },
        );

        // Create the compute pipeline
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("merge pipeline layout"),
                bind_group_layouts: &[
                    Some(&gpu.output_bind_group_layout), // config + tape
                    Some(&gpu.input_bind_group_layout),  // vars (unused)
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            },
        );

        // Compile the shader
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("merge shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });

        let pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("merge"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    fn run(&self, buffers: &Buffers, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(2, &buffers.bind_groups.merge, &[]);
        compute_pass.dispatch_workgroups(
            buffers.image_size.width().div_ceil(8),
            buffers.image_size.height().div_ceil(8),
            1,
        );
    }
}
////////////////////////////////////////////////////////////////////////////////

struct ResetContext;

impl ResetContext {
    fn run(&self, encoder: &mut wgpu::CommandEncoder, buffers: &Buffers) {
        // Clear `count` and `wg_size` members of the tile output buffers
        encoder.clear_buffer(buffers.tile64.tiles.data(), 0, Some(16));
        encoder.clear_buffer(buffers.tile8.tiles.data(), 0, Some(16));
        buffers.tile64.values.clear(encoder);
        buffers.tile8.values.clear(encoder);
        buffers.pixels.clear(encoder);

        // Clear the whole tile tape map (TODO is this needed?)
        buffers.tile_tapes.clear(encoder);
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_interval_root_shader() {
        crate::compile_shader(&interval_root_shader(16), "interval root");
    }

    #[test]
    fn compile_interval_tiles_shader() {
        crate::compile_shader(&interval_tiles_shader(16), "interval tiles");
    }

    #[test]
    fn compile_pixel_tiles_shader() {
        crate::compile_shader(&pixel_tiles_shader(16), "pixel tiles");
    }

    #[test]
    fn compile_merge_shader() {
        crate::compile_shader(&merge_shader(), "merge");
    }

    #[test]
    fn pixel_config_layout() {
        // Pick any shader, since `struct Config` is in the common text
        crate::test::compare_struct_layout::<Config>(&merge_shader(), "Config");
    }
}
