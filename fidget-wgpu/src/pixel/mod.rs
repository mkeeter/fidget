//! GPU-accelerated 2D rendering
//!
//! See the [`voxel`](crate::voxel) module for details docs; this module is
//! analogous (down to the naming of types).
use crate::{
    Gpu, RegPipeline, RenderShape, TAPE_DATA_CAPACITY, TapeWord,
    buf::{
        ArrayBuffer, BufferItemCount, BufferSizeError, BufferType, ImageBuffer,
        buffer_ro, buffer_rw,
    },
    shaders, tag,
};
use fidget_core::{
    eval::Function,
    render::ImageSize,
    shape::{MissingVar, ShapeVars},
    var::Var,
};
use fidget_raster::pixel::{Image, RawDistancePixel};
use std::num::NonZeroU64;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub use fidget_raster::pixel::{RenderConfig, RenderSize};
pub mod effects;

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const DISTANCE_PIXEL_SHADER: &str = include_str!("shaders/distance_pixel.wgsl");
const INTERVAL_INPUT: &str = include_str!("shaders/interval_input.wgsl");
const TRANSFORM_INPUT: &str = include_str!("shaders/transform_input.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const PIXEL_TILES_SHADER: &str = include_str!("shaders/pixel_tiles.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

/// Returns a shader for interval root tiles
fn interval_root_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += COMMON_SHADER;
    shader_code += DISTANCE_PIXEL_SHADER;
    shader_code += INTERVAL_ROOT_SHADER;
    shader_code += INTERVAL_INPUT;
    shader_code += TRANSFORM_INPUT;
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
    shader_code += DISTANCE_PIXEL_SHADER;
    shader_code += INTERVAL_TILES_SHADER;
    shader_code += INTERVAL_INPUT;
    shader_code += TRANSFORM_INPUT;
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
    shader_code += TRANSFORM_INPUT;
    shader_code += COMMON_SHADER;
    shader_code += DISTANCE_PIXEL_SHADER;
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
        + DISTANCE_PIXEL_SHADER
        + shaders::COMMON
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
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
        vars_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tiles_out
                    buffer_rw(1), // tile_values
                ],
            });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(vars_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        let root_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_root_shader(reg_count);
            let shader_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval root ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_root_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            root_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        reg_count: u8,
        render_size: TileRenderSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.root_tiles(ctx, buffers);
        compute_pass.set_pipeline(self.root_pipeline.get(reg_count));
        compute_pass.set_bind_group(2, bind_group, &[]);

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
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
        vars_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tiles_in
                    buffer_rw(1), // subtiles_out
                    buffer_rw(2), // subtile_values
                ],
            });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(vars_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        let tiles_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_tiles_shader(reg_count);
            let shader_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval tiles ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tiles_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            tiles_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.interval_tiles(ctx, buffers);
        compute_pass.set_pipeline(self.tiles_pipeline.get(reg_count));
        compute_pass.set_bind_group(2, bind_group, &[]);

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
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
        vars_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tiles_in
                    buffer_rw(1), // result
                ],
            });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(vars_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        let tiles_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = pixel_tiles_shader(reg_count);
            let shader_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("pixel tiles ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("pixel_tiles_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            tiles_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.pixel_tiles(ctx, buffers);
        compute_pass.set_pipeline(self.tiles_pipeline.get(reg_count));
        compute_pass.set_bind_group(2, bind_group, &[]);

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

tag!(TileTapesBufferTag, u32, STORAGE | COPY_DST);
tag!(pub PixelBufferTag, RawDistancePixel, STORAGE | COPY_SRC | COPY_DST,
    "Tag for a on-GPU buffer storing [`RawDistancePixel`] values");

/// Buffers for rendering
///
/// This object is constructed by [`Context::buffers`] and may only be used with
/// that particular [`Context`].
///
/// A successfully constructed (or resized) `Buffers` object also guarantees
/// infallible construction of an [`ImageReadBuffer`] object of the same size.
pub struct Buffers {
    /// Image render size
    ///
    /// Note that the tile buffers below round up to the nearest root tile
    /// (64² voxels).
    image_size: ImageSize,

    /// Config and tape data buffer (constant size)
    config_buf: wgpu::Buffer,

    /// Map from tile to the relevant tape (as a start index)
    tile_tapes: ArrayBuffer<TileTapesBufferTag>,

    /// Root tile buffers (64²)
    tile64: TileBuffers<64>,

    /// Second-stage tile buffers (8²)
    tile8: TileBuffers<8>,

    /// Pixel data
    pixels: ImageBuffer<PixelBufferTag>,

    /// Query set for timestamps
    ///
    /// This must be present if and only if the parent context has timestamps
    /// enabled (per [`Context::has_timestamps`])
    timestamps: Option<wgpu::QuerySet>,

    /// Buffer into which we resolve the timestamp query
    ts_buf: wgpu::Buffer,

    /// Cached bind groups
    bind_groups: BindGroups,
}

impl Buffers {
    /// Builds a new set of buffers with a default size
    ///
    /// It is expected that these will be resized before being used
    fn new(device: &wgpu::Device, has_timestamps: bool) -> Self {
        // The config buffer is statically sized, so we can check it here
        static_assertions::const_assert!(
            (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>())
                as u64
                <= BufferType::Storage.max_size()
        );

        // Dummy size for infallible construction
        let image_size = 64.into();

        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_size = TileRenderSize::from(image_size);
        let tile_tapes = ArrayBuffer::new(
            device,
            "tile tape".to_string(),
            Self::tile_tapes_buf_size(render_size),
        )
        .unwrap();

        let pixels =
            ImageBuffer::new(device, "pixels".to_string(), image_size).unwrap();

        let ts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts"),
            size: 2 * std::mem::size_of::<u64>() as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tile64 = TileBuffers::new(device, render_size).unwrap();
        let tile8 = TileBuffers::new(device, render_size).unwrap();

        let timestamps = if has_timestamps {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamp query set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }))
        } else {
            None
        };

        Self {
            config_buf,
            image_size,
            tile_tapes,
            tile64,
            tile8,
            pixels,
            timestamps,
            ts_buf,
            bind_groups: Default::default(),
        }
    }

    /// Returns image buffer size (in bytes)
    fn image_buf_size(image_size: ImageSize) -> usize {
        image_size
            .item_count()
            // Convert from GeometryPixel item count to bytes
            .checked_mul(std::mem::size_of::<RawDistancePixel>())
            .unwrap()
            // Allocate an extra 16 bytes for timestamp queries
            .checked_add(16)
            .unwrap()
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

    /// Resizes to render the target image size
    ///
    /// Internal buffers are resized to fit (only getting larger)
    ///
    /// This function also checks that the size is appropriate for an
    /// [`ImageReadBuffer`] (though we do not store such an object), so that
    /// later functions can resize it infallibly.
    fn set_image_size(
        &mut self,
        device: &wgpu::Device,
        image_size: ImageSize,
    ) -> Result<(), BuffersError> {
        let render_size = TileRenderSize::from(image_size);
        let Buffers {
            image_size: image_size_ref,
            tile_tapes,
            tile64,
            tile8,
            config_buf: _,
            pixels,
            timestamps: _,
            ts_buf: _,
            bind_groups,
        } = self;
        // Clear our cached bind groups if the image sizes is changing
        if *image_size_ref != image_size {
            *bind_groups = Default::default();
        }
        *image_size_ref = image_size;
        tile_tapes
            .grow_to_fit(device, Self::tile_tapes_buf_size(render_size))
            .map_err(|err| BuffersError {
                buf: BufferName::TileTapes,
                err,
            })?;
        tile64
            .grow_to_fit(device, render_size)
            .map_err(|e| BuffersError {
                buf: BufferName::Tile64(e.buf),
                err: e.err,
            })?;
        tile8
            .grow_to_fit(device, render_size)
            .map_err(|e| BuffersError {
                buf: BufferName::Tile8(e.buf),
                err: e.err,
            })?;
        pixels
            .grow_to_fit(device, image_size)
            .map_err(|err| BuffersError {
                buf: BufferName::Pixel,
                err,
            })?;

        // Check that we can build an `ImageReadBuffer` of the appropriate
        // size (even though they are stored separately)
        ImageReadArrayBuffer::check_size(Self::image_buf_size(image_size))
            .map_err(|err| BuffersError {
                buf: BufferName::Image,
                err,
            })?;

        Ok(())
    }

    /// Returns total allocated size (in bytes)
    pub fn capacity(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let Buffers {
            image_size: _,
            config_buf,
            tile_tapes,
            tile64,
            tile8,
            pixels,
            timestamps: _,
            ts_buf,
            bind_groups: _,
        } = self;
        config_buf.size()
            + tile_tapes.capacity()
            + tile64.capacity()
            + tile8.capacity()
            + pixels.capacity()
            + ts_buf.size()
    }

    /// Returns total active size (in bytes)
    pub fn size(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let Buffers {
            image_size: _,
            config_buf,
            tile_tapes,
            tile64,
            tile8,
            pixels,
            timestamps: _,
            ts_buf,
            bind_groups: _,
        } = self;
        config_buf.size()
            + tile_tapes.size_bytes()
            + tile64.size()
            + tile8.size()
            + pixels.size_bytes()
            + ts_buf.size()
    }

    /// Returns a handle to the image storage buffer
    ///
    /// This is intended for subsequent shaders which want to use the
    /// [`RawDistancePixel`] image data without copying to the CPU.  It requires
    /// a exclusive borrow of the `Buffers` object (and then extends that
    /// lifetime) so that other callers can't simultaneously touch the buffer.
    pub fn image_storage_buffer(&mut self) -> &ImageBuffer<PixelBufferTag> {
        &self.pixels
    }
}

/// Error returned when resizing a [`Buffers`] object
#[derive(Debug, thiserror::Error)]
#[error("failed to build {buf} buffer")]
pub struct BuffersError {
    /// Buffer which failed to resize
    pub buf: BufferName,
    /// Error returned by buffer resizing
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
}

impl std::fmt::Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferName::Tile64(buf) => write!(f, "`{buf}` tile64"),
            BufferName::Tile8(buf) => write!(f, "`{buf}` tile8"),
            BufferName::TileTapes => write!(f, "`tile tapes`"),
            BufferName::Pixel => write!(f, "`pixel`"),
            BufferName::Image => write!(f, "`image`"),
        }
    }
}

/// Error returned when submitting a voxel rasterization job to the GPU
#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    /// Missing variable when evaluating
    #[error(transparent)]
    MissingVar(#[from] MissingVar),
    /// Error while resizing buffers
    #[error(transparent)]
    Buffers(#[from] BuffersError),
}

/// Cached bind groups (constructed on-demand)
#[derive(Default)]
struct BindGroups {
    common: std::cell::OnceCell<wgpu::BindGroup>,
    root_tiles: std::cell::OnceCell<wgpu::BindGroup>,
    interval_tiles: std::cell::OnceCell<wgpu::BindGroup>,
    pixel_tiles: std::cell::OnceCell<wgpu::BindGroup>,
    merge: std::cell::OnceCell<wgpu::BindGroup>,
}

impl BindGroups {
    fn common(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.common.get_or_init(|| {
            ctx.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("common bind group"),
                    layout: &ctx.common_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffers.config_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.tile_tapes.bind_active(),
                        },
                    ],
                })
        })
    }

    fn root_tiles(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.root_tiles.get_or_init(|| {
            ctx.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("interval root bind group"),
                    layout: &ctx.root_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffers.tile64.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.tile64.values.bind_active(),
                        },
                    ],
                })
        })
    }

    fn interval_tiles(
        &self,
        ctx: &Context,
        buffers: &Buffers,
    ) -> &wgpu::BindGroup {
        self.interval_tiles.get_or_init(|| {
            ctx.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("interval tiles bind group"),
                    layout: &ctx.tiles_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffers.tile64.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.tile8.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffers.tile8.values.bind_active(),
                        },
                    ],
                })
        })
    }

    fn pixel_tiles(
        &self,
        ctx: &Context,
        buffers: &Buffers,
    ) -> &wgpu::BindGroup {
        self.pixel_tiles.get_or_init(|| {
            ctx.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pixel tiles bind group"),
                    layout: &ctx.pixels_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffers.tile8.tiles.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.pixels.bind_active(),
                        },
                    ],
                })
        })
    }

    fn merge(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.merge.get_or_init(|| {
            ctx.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("merge bind group"),
                    layout: &ctx.merge_ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffers.tile64.values.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.tile8.values.bind_active(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffers.pixels.bind_active(),
                        },
                    ],
                })
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

tag!(ImageReadTag, u8, COPY_DST | MAP_READ);
type ImageReadArrayBuffer = ArrayBuffer<ImageReadTag>;

/// Buffer for reading data back from the GPU
///
/// This object is constructed by [`Context::image_buffer`] and may only be used
/// with that particular [`Context`].
///
/// Once mapped, this is wrapped by a [`MappedImage`]
pub struct ImageReadBuffer {
    /// Image render size
    image_size: ImageSize,

    /// Result buffer that can be read back from the CPU
    ///
    /// This is mostly image pixels (as [`RawDistancePixel`] values), but also
    /// contains two trailing `u64` values for timestamps.
    buffer: ImageReadArrayBuffer,
}

impl ImageReadBuffer {
    fn new(device: &wgpu::Device, name: String) -> Self {
        let image_size = 64.into();
        Self {
            image_size,
            buffer: ImageReadArrayBuffer::new(
                device,
                name,
                Buffers::image_buf_size(image_size),
            )
            .expect("64 should always be a valid size"),
        }
    }

    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        image_size: ImageSize,
    ) -> Result<(), BufferSizeError> {
        self.image_size = image_size;
        self.buffer
            .grow_to_fit(device, Buffers::image_buf_size(image_size))
    }
}

/// Handle to a mapped image, which unmaps the image when dropped
pub struct MappedImage<'a> {
    image: &'a ImageReadBuffer,
    slice: wgpu::BufferSlice<'a>,

    /// Nanoseconds per tick, for resolving timestamps
    ns_per_tick: Option<f32>,
}

impl Drop for MappedImage<'_> {
    fn drop(&mut self) {
        self.image.buffer.data().unmap();
    }
}

impl MappedImage<'_> {
    /// Returns the image's data
    pub fn image(&self) -> Image {
        // Get the pixel-populated image
        let result = <[RawDistancePixel]>::ref_from_bytes(
            &self.slice.get_mapped_range()[..self.image_bytes()],
        )
        .unwrap()
        .to_owned();
        Image::build(result, self.image.image_size).unwrap()
    }

    /// Returns the time spent in the compute pass
    ///
    /// This may be 0 on platforms which advertise `TIMESTAMP_QUERY` but do not
    /// actually populate timestamps, and will be `None` if the context does not
    /// have `TIMESTAMP_QUERY` enabled.
    pub fn time(&self) -> Option<std::time::Duration> {
        self.ns_per_tick.map(|ns_per_tick| {
            let slice = self.slice.get_mapped_range();
            let ts =
                <[u64]>::ref_from_bytes(&slice[self.image_bytes()..]).unwrap();
            std::time::Duration::from_nanos(
                (ts[1].saturating_sub(ts[0]) as f64 * ns_per_tick as f64)
                    as u64,
            )
        })
    }

    fn image_bytes(&self) -> usize {
        (self.image.image_size.width() as usize)
            * (self.image.image_size.height() as usize)
            * std::mem::size_of::<RawDistancePixel>()
    }
}

////////////////////////////////////////////////////////////////////////////////

tag!(TilesBufferTag, u32, STORAGE | COPY_DST | INDIRECT);
tag!(ValuesBufferTag, u32, STORAGE | COPY_DST);

/// Root tile buffers store strata-packed tile lists
struct TileBuffers<const N: usize> {
    /// Output tiles
    tiles: ArrayBuffer<TilesBufferTag>,

    /// Tile values (empty / full)
    values: ImageBuffer<ValuesBufferTag>,
}

/// Error type when resizing root tile buffers
#[derive(Debug, thiserror::Error)]
#[error("failed to resize `{buf}` root tile buffer")]
pub struct TileBuffersError {
    /// Buffer which failed to resize
    pub buf: TileBufferName,
    /// Error returned by buffer resizing
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
        let tiles = ArrayBuffer::new(
            device,
            format!("tiles_out{N}"),
            Self::tiles_buf_size(render_size),
        )
        .map_err(|err| TileBuffersError {
            buf: TileBufferName::Tiles,
            err,
        })?;

        let values_buf_size = Self::values_buf_size(render_size);
        let values = ImageBuffer::new(
            device,
            format!("tile{N}_values"),
            values_buf_size,
        )
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

    /// Grows all of the buffers to fit a particular render size
    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        render_size: TileRenderSize,
    ) -> Result<(), TileBuffersError> {
        // Destructure to make sure we take all members into account
        let TileBuffers { tiles, values } = self;
        tiles
            .grow_to_fit(device, Self::tiles_buf_size(render_size))
            .map_err(|err| TileBuffersError {
                buf: TileBufferName::Tiles,
                err,
            })?;
        values
            .grow_to_fit(device, Self::values_buf_size(render_size))
            .map_err(|err| TileBuffersError {
                buf: TileBufferName::Values,
                err,
            })?;

        Ok(())
    }

    /// Returns the number of bytes in use by buffers
    pub fn size(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let TileBuffers { tiles, values } = self;
        tiles.size_bytes() + values.size_bytes()
    }

    /// Returns the number of bytes allocated to buffers
    pub fn capacity(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let TileBuffers { tiles, values } = self;
        tiles.capacity() + values.capacity()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 2D (distance field) rendering
pub struct Context {
    gpu: Gpu,
    has_timestamps: bool,

    /// Bind group layout for the common bind group (used by all stages)
    common_bind_group_layout: wgpu::BindGroupLayout,

    /// Bind group layout for the vars bind group (also by all stages)
    vars_bind_group_layout: wgpu::BindGroupLayout,

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

impl Context {
    /// Build a new 2D rendering context, given a device and queue
    ///
    /// If render timestamps are desirable, then the device should be
    /// initialized with [`wgpu::Features::TIMESTAMP_QUERY`].
    pub fn new(gpu: &Gpu) -> Self {
        let has_timestamps = gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY);
        if !has_timestamps {
            log::warn!(
                "WGPU device is missing `TIMESTAMP_QUERY`; \
                 timestamps are disabled"
            );
        }

        // Create bind group layout and bind group
        let common_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("common bind group layout"),
                entries: &[
                    buffer_rw(0), // config (including tape buffer)
                    buffer_rw(1), // tile_tape (hierarchical)
                ],
            },
        );
        let vars_bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("vars bind group layout"),
                entries: &[
                    buffer_ro(0), // vars
                ],
            },
        );

        let root_ctx = RootContext::new(
            &gpu.device,
            &common_bind_group_layout,
            &vars_bind_group_layout,
        );
        let tiles_ctx = IntervalTilesContext::new(
            &gpu.device,
            &common_bind_group_layout,
            &vars_bind_group_layout,
        );
        let pixels_ctx = PixelTilesContext::new(
            &gpu.device,
            &common_bind_group_layout,
            &vars_bind_group_layout,
        );
        let merge_ctx = MergeContext::new(
            &gpu.device,
            &common_bind_group_layout,
            &vars_bind_group_layout,
        );

        Self {
            gpu: gpu.clone(),
            has_timestamps,
            common_bind_group_layout,
            vars_bind_group_layout,
            root_ctx,
            tiles_ctx,
            pixels_ctx,
            merge_ctx,
            reset_ctx: ResetContext,
        }
    }

    /// Builds a new [`Buffers`] object for use in rendering
    ///
    /// The buffers are initialized with a dummy size and resized automatically
    /// when passed into any of the runner functions (e.g. [`run`](Self::run) or
    /// [`submit`](Self::submit)).
    pub fn buffers(&self) -> Buffers {
        Buffers::new(&self.gpu.device, self.has_timestamps)
    }

    /// Returns an [`ImageReadBuffer`] to read from a [`Buffers`] object
    pub fn image_buffer(&self) -> ImageReadBuffer {
        ImageReadBuffer::new(&self.gpu.device, "image".to_owned())
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is not present when built for the `wasm32` target
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &self,
        shape: &RenderShape,
        buffers: &mut Buffers,
        out: &mut ImageReadBuffer,
        settings: RenderConfig,
    ) -> Result<Image, SubmitError> {
        self.run_with_vars(shape, &Default::default(), buffers, out, settings)
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is not present when built for the `wasm32` target
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_with_vars(
        &self,
        shape: &RenderShape,
        vars: &ShapeVars<f32>,
        buffers: &mut Buffers,
        out: &mut ImageReadBuffer,
        settings: RenderConfig,
    ) -> Result<Image, SubmitError> {
        self.submit_with_vars(shape, vars, buffers, &settings)?;
        let image = self.map_image(buffers, out);
        Ok(image.image())
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is only relevant for the web target
    #[cfg(any(target_arch = "wasm32", doc))]
    pub async fn run_async(
        &self,
        shape: &RenderShape,
        buffers: &mut Buffers,
        out: &mut ImageReadBuffer,
        settings: RenderConfig,
    ) -> Result<Image, SubmitError> {
        self.run_with_vars_async(
            shape,
            &Default::default(),
            buffers,
            out,
            settings,
        )
        .await
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is only relevant for the web target
    #[cfg(any(target_arch = "wasm32", doc))]
    pub async fn run_with_vars_async(
        &self,
        shape: &RenderShape,
        vars: &ShapeVars<f32>,
        buffers: &mut Buffers,
        out: &mut ImageReadBuffer,
        settings: RenderConfig,
    ) -> Result<Image, SubmitError> {
        self.submit_with_vars(shape, vars, buffers, Some(out), &settings)?;
        let image = self.map_image_async(out).await;
        Ok(image.image())
    }

    /// Submits a single image to be rendered on the GPU
    ///
    /// The resulting image (as a buffer of [`RawDistancePixel`] data) is
    /// available on the GPU in
    /// [`buffers.image_storage_buffer()`](Buffers::image_storage_buffer).
    pub fn submit(
        &self,
        shape: &RenderShape,
        buffers: &mut Buffers,
        settings: &RenderConfig,
    ) -> Result<(), SubmitError> {
        self.submit_with_vars(shape, &Default::default(), buffers, settings)
    }

    /// Submits a single image to be rendered on the GPU, with extra variables
    ///
    /// See [`submit`](Self::submit) for additional details.
    pub fn submit_with_vars(
        &self,
        shape: &RenderShape,
        vars: &ShapeVars<f32>,
        buffers: &mut Buffers,
        settings: &RenderConfig,
    ) -> Result<(), SubmitError> {
        buffers.set_image_size(&self.gpu.device, settings.image_size)?;
        let render_size = TileRenderSize::from(buffers.image_size);

        // The WebGPU config type has a mat3x3f, but that type pads each row to
        // 16 bytes, so we'll just use a mat4x4 for simplicity
        let mat =
            settings.world_to_model * buffers.image_size.screen_to_world();
        let mut mat4 = nalgebra::Matrix4x3::<f32>::identity();
        mat4.fixed_view_mut::<3, 3>(0, 0).copy_from(&mat);

        // Divide by 2 to go from `u32` -> `TapeWord`
        let start_offset = u32::try_from(shape.bytecode.len()).unwrap() / 2;
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

        {
            // We load the `Config` and shape tape data.
            let config_len = std::mem::size_of_val(&config);
            let mut writer = self
                .gpu
                .queue
                .write_buffer_with(
                    &buffers.config_buf,
                    0,
                    ((config_len + shape.bytecode.as_bytes().len()) as u64)
                        .try_into()
                        .unwrap(),
                )
                .unwrap();
            writer
                .slice(..config_len)
                .copy_from_slice(config.as_bytes());
            writer
                .slice(config_len..)
                .copy_from_slice(shape.bytecode.as_bytes());
        }

        // Copy vars (if present)
        if let Some(var_size) = NonZeroU64::new(shape.vars.size()) {
            let mut writer = self
                .gpu
                .queue
                .write_buffer_with(&shape.vars, 0, var_size)
                .unwrap();
            for (v, i) in shape.shape.inner().vars().iter() {
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
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        // Initial buffer reset pass
        self.reset_ctx.run(&mut encoder, buffers);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: buffers.timestamps.as_ref().map(
                    |query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    },
                ),
            });

        // Build the common config buffer
        let common_bind_group = buffers.bind_groups.common(self, buffers);
        compute_pass.set_bind_group(0, common_bind_group, &[]);
        let vars_bind_group = shape
            .vars_bind_group(&self.gpu.device, &self.vars_bind_group_layout);
        compute_pass.set_bind_group(1, vars_bind_group, &[]);

        // Populate root tiles (64x64x64, densely packed)
        self.root_ctx.run(
            self,
            buffers,
            shape.bytecode.reg_count(),
            render_size,
            &mut compute_pass,
        );
        self.tiles_ctx.run(
            self,
            buffers,
            shape.bytecode.reg_count(),
            &mut compute_pass,
        );
        self.pixels_ctx.run(
            self,
            buffers,
            shape.bytecode.reg_count(),
            &mut compute_pass,
        );

        // Merge filled tiles from large -> small
        self.merge_ctx.run(
            self,
            buffers,
            settings.image_size,
            &mut compute_pass,
        );
        drop(compute_pass);

        // Submit the commands and wait for the GPU to complete
        self.gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn copy_image(&self, buffers: &Buffers, image_out: &mut ImageReadBuffer) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("map_image"),
            },
        );
        image_out
            .grow_to_fit(&self.gpu.device, buffers.image_size)
            .expect(
                "buffers.image_size should always be \
                 a valid size for ImageReadBuffer::grow_to_fit",
            );
        // Resolve the raw GPU ticks into the resolve buffer, then copy them
        // into the last 16 bytes of the image buffer
        if let Some(timestamps) = &buffers.timestamps {
            encoder.resolve_query_set(timestamps, 0..2, &buffers.ts_buf, 0);
            encoder.copy_buffer_to_buffer(
                &buffers.ts_buf,
                0,
                image_out.buffer.data(),
                buffers.pixels.size_bytes(), // offset past the image data
                buffers.ts_buf.size(),
            );
        }

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            buffers.pixels.data(),
            0,
            image_out.buffer.data(),
            0,
            buffers.pixels.size_bytes(),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
    }

    /// Synchronously populates and maps an CPU-readable image buffer
    ///
    /// The image is borrowed exclusively to avoid double-mapping
    ///
    /// This is a blocking function suitable for use on the desktop
    #[cfg(not(target_arch = "wasm32"))]
    pub fn map_image<'a>(
        &self,
        buffers: &Buffers,
        image_out: &'a mut ImageReadBuffer,
    ) -> MappedImage<'a> {
        self.copy_image(buffers, image_out);
        let slice = image_out.buffer.map_async(|_| {});
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        MappedImage {
            image: image_out,
            slice,
            ns_per_tick: if self.has_timestamps {
                Some(self.gpu.queue.get_timestamp_period())
            } else {
                None
            },
        }
    }

    /// Asynchronously populates and maps an CPU-readable image buffer
    ///
    /// The image is borrowed exclusively to avoid double-mapping
    ///
    /// This is an `async` function suitable for use in WebAssembly.
    #[cfg(any(target_arch = "wasm32", doc))]
    pub async fn map_image_async<'a>(
        &self,
        buffers: &Buffers,
        image_out: &'a mut ImageReadBuffer,
    ) -> MappedImage<'a> {
        self.copy_image(buffers, image_out);
        let (tx, rx) = flume::bounded(0);
        let slice = image.buffer.map_async(move |_| tx.send(()).unwrap());
        rx.recv_async().await.unwrap();
        MappedImage {
            image,
            slice,
            ns_per_tick: if self.has_timestamps {
                Some(self.gpu.queue.get_timestamp_period())
            } else {
                None
            },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct MergeContext {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl MergeContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
        vars_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader_code = merge_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("merge bind group layout"),
                entries: &[
                    buffer_ro(0), // tile64_values
                    buffer_ro(1), // tile8_values
                    buffer_rw(2), // pixels
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("merge pipeline layout"),
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(vars_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        // Compile the shader
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("merge shader module"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("merge"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        render_size: ImageSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.merge(ctx, buffers);
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(2, bind_group, &[]);
        compute_pass.dispatch_workgroups(
            render_size.width().div_ceil(8),
            render_size.height().div_ceil(8),
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
    use super::effects::ColorSettings;
    use super::*;
    use crate::ShapeColor;
    use fidget_raster::RgbaImage;

    use fidget_core::{context::Tree, vm::VmShape};

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

    struct RenderOutput {
        distance: Image,
        color: RgbaImage,
    }

    fn render(
        shapes: &[(Tree, ShapeColor<Tree>)],
        render_config: RenderConfig,
    ) -> RenderOutput {
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let pixel_ctx = Context::new(&gpu);
        let effects_ctx = effects::Context::new(&gpu);

        let mut buf = pixel_ctx.buffers();
        let mut merge_buf =
            effects_ctx.merge_buffers(render_config.image_size).unwrap();

        // Render and accumulate each shape
        for (shape, _) in shapes {
            let shape = gpu.shape(&VmShape::from(shape.clone())).unwrap();
            pixel_ctx.submit(&shape, &mut buf, &render_config).unwrap();
            effects_ctx
                .submit_merge(buf.image_storage_buffer(), true, &mut merge_buf)
                .unwrap();
        }
        let shape_colors = shapes
            .iter()
            .map(|(_, c)| {
                let ShapeColor::Rgb { r, g, b } = c;
                ShapeColor::Rgb {
                    r: VmShape::from(r.clone()),
                    g: VmShape::from(g.clone()),
                    b: VmShape::from(b.clone()),
                }
            })
            .collect::<Vec<_>>();
        let shape_colors = effects_ctx.color_buffers(&shape_colors).unwrap();

        // Compute per-pixel colors
        effects_ctx
            .submit_color(
                &mut merge_buf,
                ColorSettings {
                    z: 0.0,
                    only_filled: false,
                    world_to_model: render_config.world_to_model,
                },
                &shape_colors,
            )
            .unwrap();

        let mut out = effects_ctx.image_buffer();
        let img = effects_ctx.map_image(&merge_buf, &mut out);

        RenderOutput {
            color: img.color().unwrap(),
            distance: img.distance(),
        }
    }

    #[test]
    fn pixel_pipeline() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        let (x, y, _z) = Tree::axes();
        let circle = (x.square() + y.square()).sqrt() - Tree::constant(0.5);

        // Test a variety of image sizes for correctness
        for image_size in [
            RenderSize::new(64, 64),
            RenderSize::new(128, 64),
            RenderSize::new(64, 128),
            RenderSize::new(27, 51),
        ] {
            let out = render(
                &[(
                    circle.clone(),
                    ShapeColor::Rgb {
                        r: Tree::x(),
                        g: Tree::y(),
                        b: Tree::constant(0.5),
                    },
                )],
                RenderConfig {
                    image_size,
                    world_to_model: nalgebra::Matrix3::identity(),
                    pixel_perfect: false,
                    z: 0.0,
                },
            );
            assert_eq!(out.color.size(), image_size);
            assert_eq!(out.distance.size(), image_size);

            // Basic circle inside/outside check
            let mat = image_size.screen_to_world();
            for j in 0..image_size.height() {
                for i in 0..image_size.width() {
                    let pos = mat.transform_point(&nalgebra::Point2::new(
                        i as f32, j as f32,
                    ));
                    let p = out.distance[(j as usize, i as usize)];
                    let r = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
                    if r < 0.5 {
                        assert!(
                            p.inside(),
                            "pixel should be inside at pixel ({i}, {j}) \
                            (pos {pos}) with radius {r}"
                        );
                    } else {
                        assert!(
                            !p.inside(),
                            "pixel should be outside at pixel ({i}, {j}) \
                            (pos {pos}) with radius {r}"
                        );
                    }
                }
            }

            for j in 0..image_size.height() {
                for i in 0..image_size.width() {
                    let pos = mat.transform_point(&nalgebra::Point2::new(
                        i as f32, j as f32,
                    ));
                    let p = out.color[(j as usize, i as usize)];
                    let r = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
                    let alpha = if r < 0.5 { 255 } else { 0 };
                    let expected_color = [
                        (pos.x.clamp(0.0, 1.0) * 255.0) as u8,
                        (pos.y.clamp(0.0, 1.0) * 255.0) as u8,
                        127,
                        alpha,
                    ];
                    assert_eq!(
                        p, expected_color,
                        "color mismatch at {i}, {j} ({pos})"
                    );
                }
            }
        }
    }

    #[test]
    fn pixel_multiple_images() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        let circle_a = ((Tree::x() - 0.5).square() + Tree::y().square()).sqrt()
            - Tree::constant(0.25);
        let circle_b = ((Tree::x() + 0.5).square() + Tree::y().square()).sqrt()
            - Tree::constant(0.25);

        // Test a variety of image sizes for correctness
        let image_size = RenderSize::new(64, 64);
        let out = render(
            &[
                (
                    circle_a,
                    ShapeColor::Rgb {
                        r: Tree::constant(0.0),
                        g: Tree::constant(0.0),
                        b: Tree::constant(1.0),
                    },
                ),
                (
                    circle_b,
                    ShapeColor::Rgb {
                        r: Tree::constant(0.0),
                        g: Tree::constant(1.0),
                        b: Tree::constant(0.0),
                    },
                ),
            ],
            RenderConfig {
                image_size,
                world_to_model: nalgebra::Matrix3::identity(),
                pixel_perfect: false,
                z: 0.0,
            },
        );
        assert_eq!(out.color.size(), image_size);
        assert_eq!(out.distance.size(), image_size);

        let mut pixels = String::new();
        for j in 0..image_size.height() {
            for i in 0..image_size.width() {
                let p = out.color[(j as usize, i as usize)];
                let c = match p {
                    [0, 0, 255, 0] => "b",
                    [0, 0, 255, 255] => "B",
                    [0, 255, 0, 0] => "g",
                    [0, 255, 0, 255] => "G",
                    _ => panic!("invalid color {p:?}"),
                };
                pixels += c;
            }
            pixels += "\n";
        }

        assert_eq!(
            pixels,
            "\
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            gggggggggggggggggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            gggggggggggggGGGGGGGgggggggggggggbbbbbbbbbbbbBBBBBBBbbbbbbbbbbbb
            gggggggggggGGGGGGGGGGGgggggggggggbbbbbbbbbbBBBBBBBBBBBbbbbbbbbbb
            ggggggggggGGGGGGGGGGGGGggggggggggbbbbbbbbbBBBBBBBBBBBBBbbbbbbbbb
            ggggggggggGGGGGGGGGGGGGggggggggggbbbbbbbbbBBBBBBBBBBBBBbbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            gggggggggGGGGGGGGGGGGGGGgggggggggbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            bbbbbbbggGGGGGGGGGGGGGGGbbbbbbbbbbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            bbbbbbbggGGGGGGGGGGGGGGGgbbbbbbbbbbbbbbbbBBBBBBBBBBBBBBBbbbbbbbb
            bbbbbbbgggGGGGGGGGGGGGGggbbbbbbbbbbbbbbbbbBBBBBBBBBBBBBbbbbbbbbb
            bbbbbbbgggGGGGGGGGGGGGGggbbbbbbbbbbbbbbbbbBBBBBBBBBBBBBbbbbbbbbb
            bbbbbbbggggGGGGGGGGGGGgggbbbbbbbbbbbbbbbbbbBBBBBBBBBBBbbbbbbbbbb
            bbbbbbbggggggGGGGGGGgggggbbbbbbbbbbbbbbbbbbbbBBBBBBBbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbggggggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            "
            .replace(" ", "")
        );
    }

    #[test]
    fn pixel_config_layout() {
        // Pick any shader, since `struct Config` is in the common text
        crate::test::compare_struct_layout::<Config>(&merge_shader(), "Config");
    }
}
