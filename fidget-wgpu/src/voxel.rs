//! GPU-accelerated 3D rendering
//!
//! # Theory
//!
//! This module implements an algorithm similar to the one described in
//! [Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces (Keeter '20)](https://www.mattkeeter.com/research/mpr/).
//! The rest of this section is intended for people who have read that paper
//! ("MPR" for short).
//!
//! We use interval arithmetic on a high-fanout hierarchy of tiles (64³, 16³,
//! 4³), followed by voxel and normal evaluation.  At each stage of interval
//! arithmetic, we compute a **simplified tape** for each tile containing only
//! portions of the expression which are active.
//!
//! After the root tile evaluation, tiles are sparse.  Tiles and tapes use an
//! atomic bump allocator to claim portions of a fixed buffer.  The tile buffer
//! is always sized to fit all possible tiles; the tape buffer can run out of
//! space, in which case we fall back to the previous (unsimplified) tape.
//!
//! ## Changes versus MPR
//!
//! There are a few notable changes compared to the MPR paper and reference
//! implementation.
//!
//! First, modern GPU APIs support indirect dispatch based on buffers on the GPU
//! itself.  This saves a round-trip: the 64³ shader can compute a dispatch size
//! for the 16³ shader and store it in a buffer (and so on for subsequent
//! stages).
//!
//! In a more significant change, evaluation is broken into **strata**:
//!
//! - The initial pass of 64³ tiles renders all of those tiles, any which are
//!   active are accumulated into a set of `depth / 64` strata
//! - Strata are evaluated one at a time in z-sorted order; this is where 16³,
//!   4³, voxel, and normal evaluation happens.  You can think of this as doing
//!   raymarching on 64³ voxels at a time.
//!
//! Strata-sorted evaluation has a few advantages:
//!
//! - We can statically allocate enough space for all tiles: all 64³ in the
//!   image, then all 16³ and 4³ tiles in a single strata.  It would be
//!   prohibitive to allocate storage for all 4³ tiles in the entire volume, but
//!   doing per-strata evaluation reduces the memory scaling from N³ to N².
//! - We get some amount of Z culling, because each pass can bail out if the
//!   result in the heightmap fully covers the tile
//!
//! # Practice
//! There are four core objects, each with different lifetimes
//!
//! - [`Context`] contains all of the pipelines used for 3D rendering.  It
//!   is very expensive to build and should be constructed once per thread /
//!   worker.
//! - [`RenderShape`] contains serialized bytecode to render a particular shape.
//!   Best practice is to rebuild it only when a shape changes (i.e. not once
//!   per frame), although in practice it's pretty fast to construct.
//! - [`Buffers`] contains GPU buffers needed for rendering at a particular
//!   image size.  It is primarily expensive in GPU memory, as it contains
//!   several full-frame buffers.  Best practice is to construct one [`Buffers`]
//!   object per worker context (or per simultaneous render); if image sizes
//!   change, it can be resized with [`Context::set_buffers_image_size`] (which
//!   will grow buffers, but does not shrink them).  Systems with high
//!   variability in image size may want to periodically compare
//!   [`size`](Buffers::size) versus [`capacity`](Buffers::capacity) and fully
//!   reallocate buffers (by constructing a new `Buffers` object) if they get
//!   too out of whack.
//! - [`RenderConfig`] sets the transform matrix for rendering.  This is cheap
//!   to construct and could be built once per frame
//!
//! With all that out of the way, usage is pretty simple:
//! - Build a [`Context`]
//! - Use [`Context::shape`] to convert from a [`VmShape`] to a [`RenderShape`]
//! - Use [`Context::buffers`] to get [`Buffers`] at a particular image size
//! - Call [`Context::run`] or [`Context::run_async`] to get an image
//!
//! ## Sync and async operation
//!
//! GPU operations are asynchronous; operations are submitted to a queue, and
//! are completed at some point in the future.  [`Context::run`] blocks until
//! operations are complete, but is only valid on the desktop; it uses
//! [`wgpu::Device::poll`], which is a no-op on the web.
//! [`Context::run_async`] is the async equivalent, and is only valid in WebGPU.
//! These functions are feature-flagged and available depending on compile
//! target (native versus WebAssembly).
//!
//! Lower-level building blocks are also available: [`Context::submit`] submits
//! the render operations to the GPU, and [`Context::map_image`] /
//! [`Context::map_image_async`] map the image buffer back to the GPU.
//!
//! To reuse the image buffer within a more complex GPU pipeline (without
//! copying to the host), see [`Buffers::image_storage_buffer`].

use crate::opcode_constants;
use fidget_bytecode::{Bytecode, ReservedRegister};
use fidget_core::{eval::Function, render::VoxelSize, var::Var, vm::VmShape};
use fidget_raster::voxel::{GeometryPixel, Image};
use std::collections::BTreeMap;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const VOXEL_TILES_SHADER: &str = include_str!("shaders/voxel_tiles.wgsl");
const STACK_SHADER: &str = include_str!("shaders/stack.wgsl");
const DUMMY_STACK_SHADER: &str = include_str!("shaders/dummy_stack.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const REPACK_SHADER: &str = include_str!("shaders/repack.wgsl");
const SORT_SHADER: &str = include_str!("shaders/sort.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_OPS_SHADER: &str = include_str!("shaders/interval_ops.wgsl");
const CLEAR_SHADER: &str = include_str!("shaders/clear.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const NORMALS_SHADER: &str = include_str!("shaders/normals.wgsl");
const TAPE_INTERPRETER: &str = include_str!("shaders/tape_interpreter.wgsl");
const TAPE_SIMPLIFY: &str = include_str!("shaders/tape_simplify.wgsl");

////////////////////////////////////////////////////////////////////////////////
// Error handling zone!  This is perhaps a bit overengineered, but it meets the
// desired behavior of function error types only containing errors that they can
// actually return.

/// Buffer type for error reporting
#[derive(Copy, Clone, Debug)]
pub enum BufferType {
    /// Uniform buffer ([`wgpu::BufferUsages::UNIFORM`])
    Uniform,
    /// Storage buffer ([`wgpu::BufferUsages::STORAGE`])
    Storage,
    /// Other buffer type (e.g. [`wgpu::BufferUsages::MAP_READ`])
    Generic,
}

impl std::fmt::Display for BufferType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BufferType::Uniform => "uniform",
            BufferType::Storage => "storage",
            BufferType::Generic => "generic",
        };
        s.fmt(f)
    }
}

impl BufferType {
    /// Maximum size of this buffer type, per the WebGPU spec
    pub const fn max_size(&self) -> u64 {
        // These are copied from the spec, since we don't ask for anything extra
        match self {
            // maxUniformBufferBindingSize
            BufferType::Uniform => 64 * 1024,
            // maxStorageBufferBindingSize
            BufferType::Storage => 128 * 1024 * 1024,
            // maxBufferSize
            BufferType::Generic => 256 * 1024 * 1024,
        }
    }

    fn check(&self, requested_size: u64) -> Result<(), BufferSizeError> {
        if !requested_size.is_multiple_of(4) {
            Err(BufferSizeError::NotAligned(requested_size))
        } else if requested_size > self.max_size() {
            Err(BufferSizeError::TooLarge {
                requested_size,
                buffer_type: *self,
            })
        } else {
            Ok(())
        }
    }
}

/// Error type when resizing a buffer beyond its limit
///
/// We check against maximum buffer sizes (from the WebGPU spec) and return an
/// error immediately, instead of deferring the error to the point where the
/// buffer is used.
#[derive(Debug, thiserror::Error)]
pub enum BufferSizeError {
    /// Buffer size is not aligned to 4 bytes
    #[error("requested size {0} must be a multiple of 4 bytes")]
    NotAligned(u64),

    /// Buffer size is too large for the requested buffer usage
    #[error(
        "requested size {requested_size} exceeds maximum {} for \
        {buffer_type} buffer",
        buffer_type.max_size()
    )]
    TooLarge {
        /// Size requested (in bytes)
        requested_size: u64,
        /// Buffer type (which determines the [max size](BufferType::max_size))
        buffer_type: BufferType,
    },
}

/// Error type when resizing intermediate tile buffers
#[derive(Debug, thiserror::Error)]
#[error("failed to resize `{buf}` tile buffer")]
pub struct TileBuffersError {
    /// Buffer which failed to resize
    pub buf: TileBufferName,
    /// Error returned by buffer resizing
    #[source]
    pub err: BufferSizeError,
}

/// Names of buffers used by the intermediate tile rendering pass
///
/// This is only used for error reporting
#[derive(Debug)]
#[expect(missing_docs)]
pub enum TileBufferName {
    Tiles,
    Sorted,
    Zmin,
}

impl std::fmt::Display for TileBufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TileBufferName::Tiles => "tiles",
            TileBufferName::Sorted => "sorted",
            TileBufferName::Zmin => "zmin",
        };
        s.fmt(f)
    }
}

/// Error type when resizing root tile buffers
#[derive(Debug, thiserror::Error)]
#[error("failed to resize `{buf}` root tile buffer")]
pub struct RootTileBuffersError {
    /// Buffer which failed to resize
    pub buf: RootTileBufferName,
    /// Error returned by buffer resizing
    #[source]
    pub err: BufferSizeError,
}

/// Names of buffers used by the root tile rendering pass (for error reporting)
#[derive(Debug)]
#[expect(missing_docs)]
pub enum RootTileBufferName {
    Tiles,
    Strata,
    Zmin,
    Zmax,
}

impl std::fmt::Display for RootTileBufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RootTileBufferName::Tiles => "tiles",
            RootTileBufferName::Strata => "strata",
            RootTileBufferName::Zmin => "zmin",
            RootTileBufferName::Zmax => "zmax",
        };
        s.fmt(f)
    }
}

/// Names of all buffers, used for error reporting
#[derive(Debug)]
#[expect(missing_docs)]
pub enum BufferName {
    /// Tiles from the 64³ root tile pass
    Tile64(RootTileBufferName),
    /// Tiles from the 16³ intermediate tile pass
    Tile16(TileBufferName),
    /// Tiles from the 4³ intermediate tile pass
    Tile4(TileBufferName),
    TileTapes,
    Voxels,
    Heightmap,
    Geom,
    Image,
}

impl std::fmt::Display for BufferName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferName::Tile64(buf) => write!(f, "`{buf}` tile64"),
            BufferName::Tile16(buf) => write!(f, "`{buf}` tile16"),
            BufferName::Tile4(buf) => write!(f, "`{buf}` tile4"),
            BufferName::TileTapes => write!(f, "`tile tapes`"),
            BufferName::Voxels => write!(f, "`voxels`"),
            BufferName::Heightmap => write!(f, "`heightmap`"),
            BufferName::Geom => write!(f, "`geom`"),
            BufferName::Image => write!(f, "`image`"),
        }
    }
}

/// Error returned when resizing a [`Buffers`] object
#[derive(Debug, thiserror::Error)]
#[error("failed to build {buf} buffer when requesting size {requested:?}")]
pub struct BuffersError {
    /// Requested size
    pub requested: VoxelSize,
    /// Buffer which failed to resize
    pub buf: BufferName,
    /// Error returned by buffer resizing
    pub err: BufferSizeError,
}

////////////////////////////////////////////////////////////////////////////////

/// Settings for 3D rendering
///
/// Note that this object only contains the world-to-model transform; the image
/// size is set by the [`Buffers`] object passed into [`run`](Context::run) or
/// [`run_async`](Context::run_async).
#[derive(Copy, Clone)]
pub struct RenderConfig {
    /// World-to-model transform
    pub world_to_model: nalgebra::Matrix4<f32>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            world_to_model: nalgebra::Matrix4::identity(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Doppelganger of the WGSL `struct Config`
///
/// Fields are carefully ordered to require no internal padding (enforced by
/// `zerocopy` derives)
#[derive(Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
#[repr(C)]
struct Config {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    /// Initial offset in `tape_data`
    tape_data_offset: u32,

    /// Render size, rounded up to the nearest multiple of 64
    render_size: [u32; 3],

    /// Number of words in the trailing tape buffer
    tape_data_capacity: u32,

    /// Image size (not rounded)
    image_size: [u32; 3],

    /// Length of the root tape
    root_tape_len: u32,
    // This is followed by a flexible array member containing tape data
}

/// A render size is rounded up to the next multiple of 64 on every axis
///
/// The internal `VoxelSize` stores divided-by-64 values, so that the render
/// size cannot be constructed with an invalid state.
#[derive(Copy, Clone, Debug)]
struct TileRenderSize(VoxelSize);

impl From<VoxelSize> for TileRenderSize {
    fn from(image_size: VoxelSize) -> Self {
        let nx = image_size.width().div_ceil(64);
        let ny = image_size.height().div_ceil(64);
        let nz = image_size.depth().div_ceil(64);
        Self(VoxelSize::new(nx, ny, nz))
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

    /// Number of tiles in the Z axis
    fn nz(&self) -> u32 {
        self.0.depth()
    }

    /// Number of voxels in the X axis (always a multiple of 64)
    fn width(&self) -> u32 {
        self.0.width() * 64
    }

    /// Number of voxels in the Y axis (always a multiple of 64)
    fn height(&self) -> u32 {
        self.0.height() * 64
    }

    /// Number of voxels in the Z axis (always a multiple of 64)
    fn depth(&self) -> u32 {
        self.0.depth() * 64
    }

    /// Number of pixels in total
    fn pixels(&self) -> usize {
        self.width() as usize * self.height() as usize
    }
}

/// Number of [`TapeWord`] words in the tape data flexible array
const TAPE_DATA_CAPACITY: usize = 8 * 1024 * 1024; // 8M words, 64 MiB

#[repr(C)]
struct TapeWord {
    op: u32,
    imm: u32,
}

/// Returns a shader for interval root tiles
fn interval_root_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += INTERVAL_ROOT_SHADER;
    shader_code += INTERVAL_OPS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += STACK_SHADER;
    shader_code += TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for interval root tile repacking
fn repack_shader() -> String {
    let mut shader_code = String::new();
    shader_code += REPACK_SHADER;
    shader_code += COMMON_SHADER;
    shader_code
}

/// Returns a shader for interval tile sorting
fn sort_shader() -> String {
    let mut shader_code = String::new();
    shader_code += SORT_SHADER;
    shader_code += COMMON_SHADER;
    shader_code
}

/// Returns a shader for interval tile evaluation
fn interval_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += INTERVAL_TILES_SHADER;
    shader_code += INTERVAL_OPS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += STACK_SHADER;
    shader_code += TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for voxel tile evaluation
fn voxel_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += VOXEL_TILES_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += DUMMY_STACK_SHADER;
    shader_code
}

/// Returns a shader for normals evaluation
fn normals_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += NORMALS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += DUMMY_STACK_SHADER;
    shader_code
}

/// Returns a shader for merging images
fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + COMMON_SHADER
}

/// Returns a shader for clearing counters in between strata passes
fn clear_shader() -> String {
    CLEAR_SHADER.to_owned() + COMMON_SHADER
}

/// Helper function to make a read-only buffer binding
fn buffer_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper function to make a read-only buffer binding with dynamic offset
fn buffer_ro_dyn(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper function to make a read-write buffer binding
fn buffer_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Container of multiple pipelines, parameterized by register count
struct RegPipeline(BTreeMap<u8, wgpu::ComputePipeline>);

impl RegPipeline {
    fn build<F: Fn(u8) -> wgpu::ComputePipeline>(builder: F) -> Self {
        let mut out = BTreeMap::new();
        for reg_count in [8, 16, 32, 64, 128, 192, 255] {
            out.insert(reg_count, builder(reg_count));
        }
        Self(out)
    }

    /// Returns the pipeline with sufficient registers to render `reg_count`
    ///
    /// # Panics
    /// If `reg_count` is 256 (which is not allowed in bytecode tapes)
    fn get(&self, reg_count: u8) -> &wgpu::ComputePipeline {
        let (r, v) = self
            .0
            .range(reg_count..)
            .next()
            .expect("bytecode tape cannot use more than 255 registers");
        assert!(*r >= reg_count);
        v
    }
}

/// Root context, which produces a list of 64³ tiles
struct RootContext {
    /// Pipelines for 64³ tile evaluation
    root_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Per-strata offset in the root tiles list
///
/// This must be equivalent to `strata_size_bytes` in the interval root shader
fn strata_size_bytes(render_size: TileRenderSize) -> u64 {
    let nx = u64::from(render_size.nx());
    let ny = u64::from(render_size.ny());
    // Snap to `min_storage_buffer_offset_alignment`
    ((nx * ny + 4) * std::mem::size_of::<u32>() as u64).next_multiple_of(256)
}

impl RootContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tiles_out
                    buffer_rw(1), // tile64_zmax
                ],
            });

        let root_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_root_shader(reg_count);
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        Some(common_bind_group_layout),
                        Some(&bind_group_layout),
                    ],
                    immediate_size: 0u32,
                },
            );
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
        let bind_group = buffers.bind_groups.root(ctx, buffers);
        compute_pass.set_pipeline(self.root_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, bind_group, &[]);

        // Workgroup is 4x4x4, so we divide by 4 here on each axis
        let nx = render_size.nx().div_ceil(4);
        let ny = render_size.ny().div_ceil(4);
        let nz = render_size.nz().div_ceil(4);
        compute_pass.dispatch_workgroups(nx, ny, nz);
    }
}

/// Repack context, which strata-sorts a list of 64³ tiles
struct RepackContext {
    /// Pipeline for 64³ tile packing
    repack_pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl RepackContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tiles_out
                    buffer_ro(1), // tile64_zmin
                    buffer_rw(2), // strata_tiles
                ],
            });

        let repack_pipeline = {
            let shader_code = repack_shader();
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        Some(common_bind_group_layout),
                        Some(&bind_group_layout),
                    ],
                    immediate_size: 0u32,
                },
            );
            let shader_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("repack"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("repack_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            bind_group_layout,
            repack_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        render_size: TileRenderSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.repack(ctx, buffers);

        compute_pass.set_pipeline(&self.repack_pipeline);
        compute_pass.set_bind_group(1, bind_group, &[]);

        // Workgroup is 64x1x1, so we divide on the X axis.  It doesn't matter
        // much; we just need one thread per possible output tile from the
        // previous stage, i.e. `(nx * ny * nz)` total threads.  This could be
        // optimized further with indirect dispatch, but ehhhhhhh
        let nx = render_size.nx().div_ceil(64);
        let ny = render_size.ny();
        let nz = render_size.nz();
        compute_pass.dispatch_workgroups(nx, ny, nz);
    }
}

////////////////////////////////////////////////////////////////////////////////

struct IntervalContext {
    /// Pipeline for 64³ -> 16³ tile evaluation
    interval64_pipeline: RegPipeline,

    /// Pipeline to sort 16³ tiles
    sort16_pipeline: wgpu::ComputePipeline,

    /// Pipeline for 16³ -> 4³ tile evaluation
    interval16_pipeline: RegPipeline,

    /// Pipeline to sort 4³ tiles
    sort4_pipeline: wgpu::ComputePipeline,

    /// Bind group layout for interval pipelines
    interval_bind_group_layout: wgpu::BindGroupLayout,

    /// Bind group layout for sort pipelines
    sort_bind_group_layout: wgpu::BindGroupLayout,
}

impl IntervalContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let interval_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro_dyn(0), // tiles_in
                    buffer_ro(1),     // tile_zmin
                    buffer_rw(2),     // subtiles_out
                    buffer_rw(3),     // subtile_zmin
                    buffer_rw(4),     // subtile_zhist
                ],
            });

        let interval_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(&interval_bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        let interval64_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_tiles_shader(reg_count);
            // SAFETY: the shader is carefully written
            let shader_module = unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                    },
                    wgpu::ShaderRuntimeChecks {
                        bounds_checks: false,
                        force_loop_bounding: false,
                        ray_query_initialization_tracking: false,
                        task_shader_dispatch_tracking: false,
                        mesh_shader_primitive_indices_clamp: false,
                    },
                )
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval64 ({reg_count})")),
                layout: Some(&interval_pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 64.0), ("SUBTILE_SIZE", 16.0)],
                    ..Default::default()
                },
                cache: None,
            })
        });

        let interval16_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_tiles_shader(reg_count);
            // SAFETY: the shader is carefully written
            let shader_module = unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                    },
                    wgpu::ShaderRuntimeChecks {
                        bounds_checks: false,
                        force_loop_bounding: false,
                        ray_query_initialization_tracking: false,
                        task_shader_dispatch_tracking: false,
                        mesh_shader_primitive_indices_clamp: false,
                    },
                )
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval16 ({reg_count})")),
                layout: Some(&interval_pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 16.0), ("SUBTILE_SIZE", 4.0)],
                    ..Default::default()
                },
                cache: None,
            })
        });

        let sort_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // subtiles_out
                    buffer_rw(1), // z_hist
                    buffer_rw(2), // sorted_subtiles
                ],
            });
        let sort_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(&sort_bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        let shader_code = sort_shader();
        // SAFETY: the shader is carefully written
        let shader_module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                },
                wgpu::ShaderRuntimeChecks {
                    bounds_checks: false,
                    force_loop_bounding: false,
                    ray_query_initialization_tracking: false,
                    task_shader_dispatch_tracking: false,
                    mesh_shader_primitive_indices_clamp: false,
                },
            )
        };
        let sort16_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sort16"),
                layout: Some(&sort_pipeline_layout),
                module: &shader_module,
                entry_point: Some("sort_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("SUBTILE_SIZE", 16.0)],
                    ..Default::default()
                },
                cache: None,
            });
        let sort4_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sort4"),
                layout: Some(&sort_pipeline_layout),
                module: &shader_module,
                entry_point: Some("sort_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("SUBTILE_SIZE", 4.0)],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            interval_bind_group_layout,
            sort_bind_group_layout,
            interval64_pipeline,
            sort16_pipeline,
            interval16_pipeline,
            sort4_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        strata: u64,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let strata_bytes = buffers.strata_size_bytes();
        let offset_bytes = strata * strata_bytes;
        let bind_group16 = buffers.bind_groups.interval16(ctx, buffers);
        compute_pass.set_pipeline(self.interval64_pipeline.get(reg_count));
        compute_pass.set_bind_group(
            1,
            bind_group16,
            &[u32::try_from(offset_bytes).unwrap()],
        );
        compute_pass.dispatch_workgroups_indirect(
            &buffers.tile64.strata.data,
            offset_bytes,
        );

        let bind_group_sort16 = buffers.bind_groups.sort16(ctx, buffers);
        compute_pass.set_pipeline(&self.sort16_pipeline);
        compute_pass.set_bind_group(1, bind_group_sort16, &[]);
        compute_pass
            .dispatch_workgroups_indirect(&buffers.tile16.tiles.data, 0);

        let bind_group4 = buffers.bind_groups.interval4(ctx, buffers);
        compute_pass.set_pipeline(self.interval16_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, bind_group4, &[0]);
        compute_pass
            .dispatch_workgroups_indirect(&buffers.tile16.sorted.data, 0);

        let bind_group_sort4 = buffers.bind_groups.sort4(ctx, buffers);
        compute_pass.set_pipeline(&self.sort4_pipeline);
        compute_pass.set_bind_group(1, bind_group_sort4, &[]);
        compute_pass.dispatch_workgroups_indirect(&buffers.tile4.tiles.data, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////

struct VoxelContext {
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for interpreted voxel evaluation
    voxel_pipeline: RegPipeline,
}

impl VoxelContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tiles4_in
                    buffer_ro(1), // tile4_zmin
                    buffer_rw(2), // result
                ],
            });

        let voxel_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = voxel_tiles_shader(reg_count);
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        Some(common_bind_group_layout),
                        Some(&bind_group_layout),
                    ],
                    immediate_size: 0u32,
                },
            );
            // SAFETY: The shader is careful, good luck
            let shader_module = unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                    },
                    wgpu::ShaderRuntimeChecks {
                        bounds_checks: false,
                        force_loop_bounding: false,
                        ray_query_initialization_tracking: false,
                        task_shader_dispatch_tracking: false,
                        mesh_shader_primitive_indices_clamp: false,
                    },
                )
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("voxels ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("voxel_ray_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            voxel_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.voxel(ctx, buffers);
        compute_pass.set_pipeline(self.voxel_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, bind_group, &[]);

        // Each workgroup is 4x4x4, i.e. covering a 4x4 splat of pixels with 4x
        // workers in the Z direction.
        compute_pass
            .dispatch_workgroups_indirect(&buffers.tile4.sorted.data, 0);
    }
}

struct NormalsContext {
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    /// Pipeline for normal evaluation
    normals_pipeline: RegPipeline,
}

impl NormalsContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // image_heightmap
                    buffer_rw(1), // image_out
                ],
            });

        let normals_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = normals_shader(reg_count);
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        Some(common_bind_group_layout),
                        Some(&bind_group_layout),
                    ],
                    immediate_size: 0u32,
                },
            );
            let shader_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("normals ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("normals_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            normals_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.normals(ctx, buffers);
        compute_pass.set_pipeline(self.normals_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, bind_group, &[]);

        compute_pass.dispatch_workgroups(
            buffers.image_size.width().div_ceil(8),
            buffers.image_size.height().div_ceil(8),
            1,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 3D (combined heightmap and normal) rendering
pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    has_timestamps: bool,

    /// Bind group layout for the common bind group (used by all stages)
    bind_group_layout: wgpu::BindGroupLayout,

    root_ctx: RootContext,
    repack_ctx: RepackContext,
    interval_ctx: IntervalContext,
    voxel_ctx: VoxelContext,
    normals_ctx: NormalsContext,
    merge_ctx: MergeContext,
    reset_ctx: ResetContext,
    clear_ctx: ClearContext,
}

struct TileBuffers<const N: u64> {
    /// Tiles written by the stage outputing N^3 tiles
    tiles: Buffer,
    /// Sorted version of [`tiles`](Self::tiles)
    sorted: Buffer,
    /// Minimum Z height at each XY tile
    zmin: Buffer,
    /// Histogram of Z values (used when sorting)
    z_hist: wgpu::Buffer,
}

impl<const N: u64> TileBuffers<N> {
    /// Returns a new `TileBuffers` object
    fn new(
        device: &wgpu::Device,
        render_size: TileRenderSize,
    ) -> Result<Self, TileBuffersError> {
        let tile_buf_size = Self::tile_buf_size(render_size);
        let tiles = Buffer::new(
            device,
            format!("active_tile{N}"),
            tile_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        )
        .map_err(|err| TileBuffersError {
            buf: TileBufferName::Tiles,
            err,
        })?;
        let sorted = Buffer::new(
            device,
            format!("sorted_tile{N}"),
            tile_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        )
        .map_err(|err| TileBuffersError {
            buf: TileBufferName::Sorted,
            err,
        })?;
        let zmin = Buffer::new(
            device,
            format!("tile{N}_zmin"),
            Self::zmin_buf_size(render_size),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| TileBuffersError {
            buf: TileBufferName::Zmin,
            err,
        })?;

        // z_hist never changes size, since it's based on `N`
        let z_hist = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("tile{N}_zhist")),
            size: (64 / N) * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            tiles,
            sorted,
            zmin,
            z_hist,
        })
    }

    fn tile_buf_size(render_size: TileRenderSize) -> u64 {
        let nx = u64::from(render_size.width()) / N;
        let ny = u64::from(render_size.height()) / N;
        let nz = 64 / N;
        // wg_dispatch: [u32; 3]
        // count: u32,
        (4 + nx * ny * nz) * std::mem::size_of::<u32>() as u64
    }

    fn zmin_buf_size(render_size: TileRenderSize) -> u64 {
        let nx = u64::from(render_size.width()) / N;
        let ny = u64::from(render_size.height()) / N;
        (nx * ny) * std::mem::size_of::<u32>() as u64
    }

    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        render_size: TileRenderSize,
    ) -> Result<(), TileBuffersError> {
        let TileBuffers {
            tiles,
            sorted,
            zmin,
            z_hist: _, // never changes, it'll always be like this
        } = self;
        let tile_buf_size = Self::tile_buf_size(render_size);
        tiles.grow_to_fit(device, tile_buf_size).map_err(|err| {
            TileBuffersError {
                buf: TileBufferName::Tiles,
                err,
            }
        })?;
        sorted.grow_to_fit(device, tile_buf_size).map_err(|err| {
            TileBuffersError {
                buf: TileBufferName::Sorted,
                err,
            }
        })?;
        zmin.grow_to_fit(device, Self::zmin_buf_size(render_size))
            .map_err(|err| TileBuffersError {
                buf: TileBufferName::Zmin,
                err,
            })?;

        Ok(())
    }

    /// Returns the number of bytes in use by these buffers
    ///
    /// See [`self.capacity`](Self::capacity) for total bytes allocated
    pub fn size(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let TileBuffers {
            tiles,
            sorted,
            zmin,
            z_hist,
        } = self;
        tiles.size() + sorted.size() + zmin.size() + z_hist.size()
    }

    /// Returns the number of bytes allocated by these buffers
    pub fn capacity(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let TileBuffers {
            tiles,
            sorted,
            zmin,
            z_hist,
        } = self;
        tiles.capacity() + sorted.capacity() + zmin.capacity() + z_hist.size()
    }
}

/// Root tile buffers store strata-packed tile lists
struct RootTileBuffers {
    /// Initial output tiles
    tiles: Buffer,
    /// Strata-sorted output tiles
    strata: Buffer,
    zmin: Buffer,
    zmax: Buffer,
}

impl RootTileBuffers {
    /// Build a new root tiles buffer, which stores strata-packed tile lists
    fn new(
        device: &wgpu::Device,
        render_size: TileRenderSize,
    ) -> Result<Self, RootTileBuffersError> {
        // Root tile buffers are always 64³ voxels
        const N: usize = 64;

        // Allocate enough words to write all of the output tiles
        let tiles = Buffer::new(
            device,
            format!("tiles_out{N}"),
            Self::tiles_buf_size(render_size),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| RootTileBuffersError {
            buf: RootTileBufferName::Tiles,
            err,
        })?;

        let strata = Buffer::new(
            device,
            format!("strata_tile{N}"),
            Self::strata_buf_size(render_size),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| RootTileBuffersError {
            buf: RootTileBufferName::Strata,
            err,
        })?;

        let z_buf_size = Self::z_buf_size(render_size);
        let zmin = Buffer::new(
            device,
            format!("tile{N}_zmin"),
            z_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| RootTileBuffersError {
            buf: RootTileBufferName::Zmin,
            err,
        })?;
        let zmax = Buffer::new(
            device,
            format!("tile{N}_zmax"),
            z_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| RootTileBuffersError {
            buf: RootTileBufferName::Zmax,
            err,
        })?;
        Ok(Self {
            tiles,
            strata,
            zmin,
            zmax,
        })
    }

    fn tiles_buf_size(render_size: TileRenderSize) -> u64 {
        let nx = u64::from(render_size.nx());
        let ny = u64::from(render_size.ny());
        let nz = u64::from(render_size.nz());
        // wg_dispatch: [u32; 3] (unused)
        // count: u32,
        (4 + nx * ny * nz) * std::mem::size_of::<u32>() as u64
    }

    fn strata_buf_size(render_size: TileRenderSize) -> u64 {
        let nz = u64::from(render_size.nz());
        let strata_size = strata_size_bytes(render_size);
        strata_size * nz
    }

    fn z_buf_size(render_size: TileRenderSize) -> u64 {
        let nx = u64::from(render_size.nx());
        let ny = u64::from(render_size.ny());
        nx * ny * std::mem::size_of::<u32>() as u64
    }

    /// Grows all of the buffers to fit a particular render size
    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        render_size: TileRenderSize,
    ) -> Result<(), RootTileBuffersError> {
        // Destructure to make sure we take all members into account
        let RootTileBuffers {
            tiles,
            strata,
            zmin,
            zmax,
        } = self;
        tiles
            .grow_to_fit(device, Self::tiles_buf_size(render_size))
            .map_err(|err| RootTileBuffersError {
                buf: RootTileBufferName::Tiles,
                err,
            })?;
        strata
            .grow_to_fit(device, Self::strata_buf_size(render_size))
            .map_err(|err| RootTileBuffersError {
                buf: RootTileBufferName::Strata,
                err,
            })?;

        let z_buf_size = Self::z_buf_size(render_size);
        zmin.grow_to_fit(device, z_buf_size).map_err(|err| {
            RootTileBuffersError {
                buf: RootTileBufferName::Zmin,
                err,
            }
        })?;
        zmax.grow_to_fit(device, z_buf_size).map_err(|err| {
            RootTileBuffersError {
                buf: RootTileBufferName::Zmax,
                err,
            }
        })?;

        Ok(())
    }

    /// Returns the number of bytes in use by buffers
    pub fn size(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let RootTileBuffers {
            tiles,
            strata,
            zmin,
            zmax,
        } = self;
        tiles.size() + strata.size() + zmin.size() + zmax.size()
    }

    /// Returns the number of bytes allocated to buffers
    pub fn capacity(&self) -> u64 {
        // Destructure to make sure we take all members into account
        let RootTileBuffers {
            tiles,
            strata,
            zmin,
            zmax,
        } = self;
        tiles.capacity() + strata.capacity() + zmin.capacity() + zmax.capacity()
    }
}

/// Shape for rendering
///
/// This object is constructed by [`Context::shape`] and may only be used with
/// that particular [`Context`].
pub struct RenderShape {
    axes: [u32; 3],
    bytecode: Bytecode,
}

/// Error type when constructing a [`RenderShape`]
#[derive(Debug, thiserror::Error)]
pub enum RenderShapeError {
    /// The shape doesn't fit in the GPU tape buffer
    #[error(
        "shape bytecode is {0} tape words (8 bytes each), which exceeds \
        buffer capacity of {TAPE_DATA_CAPACITY} tape words"
    )]
    TooLong(usize),
    /// The shape uses a reserved register
    #[error(transparent)]
    RegisterError(#[from] ReservedRegister),
    /// The shape uses variables other than X, Y, Z
    #[error("shape may only use X, Y, Z variables")]
    XyzOnly,
}

impl RenderShape {
    fn new(shape: &VmShape) -> Result<Self, RenderShapeError> {
        // Generate bytecode for the root tape
        let vars = shape.inner().vars();
        if vars.iter().any(|(v, _)| matches!(v, Var::V(..))) {
            return Err(RenderShapeError::XyzOnly);
        }
        let bytecode = Bytecode::new(shape.inner().data())?;
        if bytecode.len() / 2 > TAPE_DATA_CAPACITY {
            return Err(RenderShapeError::TooLong(bytecode.len() / 2));
        }

        // Create the 4x4 transform matrix
        let axes = shape
            .axes()
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX));

        Ok(Self { axes, bytecode })
    }
}

/// Buffers for rendering, which control the rendered image size
///
/// This object is constructed by [`Context::buffers`] and may only be used with
/// that particular [`Context`].
pub struct Buffers {
    /// Image render size
    ///
    /// Note that the tile buffers below round up to the nearest root tile
    /// (64³ voxels).
    image_size: VoxelSize,

    /// Config and tape data buffer
    config_buf: wgpu::Buffer,

    /// Map from tile to the relevant tape (as a start index)
    tile_tapes: Buffer,

    /// Root tile Z heights (64³)
    tile64: RootTileBuffers,

    /// Z heights for filled first-stage tiles (16^3)
    tile16: TileBuffers<16>,

    /// Z heights for filled second-stage tiles (4^3)
    tile4: TileBuffers<4>,

    /// Z heights for voxels
    voxels: Buffer,

    /// Combined Z height buffer, at the target image size
    heightmap: Buffer,

    /// Buffer of [`GeometryPixel`] data, generated by the normal pass
    geom: Buffer,

    /// Result buffer that can be read back from the host
    ///
    /// This is mostly image pixels (as [`GeometryPixel`] values), but also
    /// contains two trailing `u64` values for timestamps.
    image: Buffer,

    /// Query set for timestamps
    ///
    /// This must be present if and only if the parent context has timestamps
    /// enabled (per [`Context::has_timestamps`])
    timestamps: Option<wgpu::QuerySet>,

    /// Buffer into which we resolve the query
    ts_buf: wgpu::Buffer,

    /// Cached bind groups
    bind_groups: BindGroups,
}

/// Cached bind groups (constructed on-demand)
#[derive(Default)]
struct BindGroups {
    common: std::cell::OnceCell<wgpu::BindGroup>,
    merge: std::cell::OnceCell<wgpu::BindGroup>,
    root: std::cell::OnceCell<wgpu::BindGroup>,
    repack: std::cell::OnceCell<wgpu::BindGroup>,
    interval16: std::cell::OnceCell<wgpu::BindGroup>,
    sort16: std::cell::OnceCell<wgpu::BindGroup>,
    interval4: std::cell::OnceCell<wgpu::BindGroup>,
    sort4: std::cell::OnceCell<wgpu::BindGroup>,
    voxel: std::cell::OnceCell<wgpu::BindGroup>,
    normals: std::cell::OnceCell<wgpu::BindGroup>,
    clear: std::cell::OnceCell<wgpu::BindGroup>,
}

impl BindGroups {
    fn common(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.common.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.bind_group_layout,
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

    fn clear(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.clear.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.clear_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile16.tiles.data.slice(0..16).into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers
                            .tile16
                            .sorted
                            .data
                            .slice(0..16)
                            .into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile16.z_hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile4.tiles.data.slice(0..16).into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.tile4.sorted.data.slice(0..16).into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers.tile4.z_hist.as_entire_binding(),
                    },
                ],
            })
        })
    }

    fn merge(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.merge.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.merge_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile64.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile16.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile4.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.voxels.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.heightmap.bind_active(),
                    },
                ],
            })
        })
    }

    fn root(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.root.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.root_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile64.tiles.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile64.zmax.bind_active(),
                    },
                ],
            })
        })
    }

    fn repack(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.repack.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.repack_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile64.tiles.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile64.zmax.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile64.strata.bind_active(),
                    },
                ],
            })
        })
    }

    fn interval16(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        let strata_bytes = buffers.strata_size_bytes();
        self.interval16.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.interval_ctx.interval_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers
                            .tile64
                            .strata
                            .data
                            .slice(0..strata_bytes) // dynamic offset!
                            .into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile64.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile16.tiles.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile16.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.tile16.z_hist.as_entire_binding(),
                    },
                ],
            })
        })
    }

    fn sort16(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.sort16
            .get_or_init(|| Self::sort_bind_group(ctx, &buffers.tile16))
    }

    fn sort4(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.sort4
            .get_or_init(|| Self::sort_bind_group(ctx, &buffers.tile4))
    }

    fn sort_bind_group<const N: u64>(
        ctx: &Context,
        tile_bufs: &TileBuffers<N>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.interval_ctx.sort_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_bufs.tiles.bind_active(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_bufs.z_hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_bufs.sorted.bind_active(),
                },
            ],
        })
    }

    fn interval4(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.interval4.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.interval_ctx.interval_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile16.sorted.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile16.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile4.tiles.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile4.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.tile4.z_hist.as_entire_binding(),
                    },
                ],
            })
        })
    }

    fn voxel(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.voxel.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.voxel_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile4.sorted.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile4.zmin.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.voxels.bind_active(),
                    },
                ],
            })
        })
    }

    fn normals(&self, ctx: &Context, buffers: &Buffers) -> &wgpu::BindGroup {
        self.normals.get_or_init(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.normals_ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.heightmap.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.geom.bind_active(),
                    },
                ],
            })
        })
    }
}

impl Buffers {
    /// Returns the current image size
    pub fn image_size(&self) -> VoxelSize {
        self.image_size
    }

    /// Returns the image storage buffer and its valid size (in bytes)
    ///
    /// This is intended for subsequent shaders which want to use the
    /// [`GeometryPixel`] image data without copying to the host.
    ///
    /// The buffer is configured with  `STORAGE | COPY_SRC | COPY_DST`; note
    /// that it is not valid for `MAP_READ` operations.  To map the
    /// CPU-accessible image buffer, see [`Context::map_image`] and
    /// [`Context::map_image_async`].
    ///
    /// The buffer is necessarily larger than the image data; at the very least,
    /// it contains two extra `u64` timestamp values, and it may be even larger
    /// if the [`Buffers`] object has been resized over time.  The caller should
    /// use the second member in the tuple when binding the buffer, and may also
    /// want to use [`Buffers::image_size`].
    pub fn image_storage_buffer(&self) -> (&wgpu::Buffer, u64) {
        (&self.geom.data, self.geom.size)
    }
}

/// Handle around a growable GPU buffer which pretends to be smaller
struct Buffer {
    /// Current active size, which may be smaller than the buffer's capacity
    size: u64,
    /// Actual GPU buffer
    data: wgpu::Buffer,
    /// Buffer label (to be used when reallocating)
    name: String,
}

impl Buffer {
    fn new(
        device: &wgpu::Device,
        name: String,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<Self, BufferSizeError> {
        assert_eq!(size % 4, 0);
        Self::check_size(usage, size)?;
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name.as_str()),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self { data, size, name })
    }

    fn check_size(
        usage: wgpu::BufferUsages,
        size: u64,
    ) -> Result<(), BufferSizeError> {
        let buf_ty = if usage.contains(wgpu::BufferUsages::STORAGE) {
            BufferType::Storage
        } else if usage.contains(wgpu::BufferUsages::UNIFORM) {
            BufferType::Uniform
        } else {
            BufferType::Generic
        };
        buf_ty.check(size)
    }

    /// Grows the buffer to fit a particular size in bytes
    ///
    /// If the buffer already fits that size, then no allocation is performed,
    /// but we always update the internal `size` member (e.g. so that
    /// [`bind_active`](Self::bind_active) returns the correct
    /// subset of the buffer).
    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        size: u64,
    ) -> Result<(), BufferSizeError> {
        assert_eq!(size % 4, 0);
        if size > self.capacity() {
            let usage = self.data.usage();
            Self::check_size(usage, size)?;
            self.data = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.name.as_str()),
                size,
                usage,
                mapped_at_creation: false,
            });
        }
        self.size = size;
        Ok(())
    }

    /// Returns a binding resource for the active slice of the buffer
    fn bind_active(&self) -> wgpu::BindingResource<'_> {
        self.data.slice(0..self.size).into()
    }

    /// Returns the active buffer size
    fn size(&self) -> u64 {
        self.size
    }

    /// Returns the total buffer capacity, which may be larger than its size
    fn capacity(&self) -> u64 {
        self.data.size()
    }

    /// Maps the active portion of the buffer for reading
    fn map_async(
        &self,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>)
        + wgpu::WasmNotSend
        + 'static,
    ) -> wgpu::BufferSlice<'_> {
        let slice = self.data.slice(0..self.size);
        slice.map_async(wgpu::MapMode::Read, callback);
        slice
    }

    /// Clears the active portion of the buffer
    fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.data, 0, Some(self.size));
    }
}

impl Buffers {
    fn new(
        device: &wgpu::Device,
        image_size: VoxelSize,
        has_timestamps: bool,
    ) -> Result<Self, BuffersError> {
        // The config buffer is statically sized, so we can check it here
        static_assertions::const_assert!(
            (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>())
                as u64
                <= BufferType::Storage.max_size()
        );
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_size = TileRenderSize::from(image_size);
        let voxels = Buffer::new(
            device,
            "voxels".to_string(),
            Self::voxels_buf_size(render_size),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| BuffersError {
            requested: image_size,
            buf: BufferName::Voxels,
            err,
        })?;
        let tile_tapes = Buffer::new(
            device,
            "tile tape".to_string(),
            Self::tile_tapes_buf_size(render_size),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| BuffersError {
            requested: image_size,
            buf: BufferName::TileTapes,
            err,
        })?;

        let heightmap = Buffer::new(
            device,
            "heightmap".to_string(),
            Self::heightmap_buf_size(image_size),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| BuffersError {
            requested: image_size,
            buf: BufferName::Heightmap,
            err,
        })?;
        let geom = Buffer::new(
            device,
            "geom".to_string(),
            Self::geom_buf_size(image_size),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )
        .map_err(|err| BuffersError {
            requested: image_size,
            buf: BufferName::Geom,
            err,
        })?;

        let image = Buffer::new(
            device,
            "image".to_string(),
            Self::image_buf_size(image_size),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )
        .map_err(|err| BuffersError {
            requested: image_size,
            buf: BufferName::Image,
            err,
        })?;

        let ts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts"),
            size: 2 * std::mem::size_of::<u64>() as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tile64 =
            RootTileBuffers::new(device, render_size).map_err(|e| {
                BuffersError {
                    requested: image_size,
                    buf: BufferName::Tile64(e.buf),
                    err: e.err,
                }
            })?;
        let tile16 = TileBuffers::new(device, render_size).map_err(|e| {
            BuffersError {
                requested: image_size,
                buf: BufferName::Tile16(e.buf),
                err: e.err,
            }
        })?;
        let tile4 = TileBuffers::new(device, render_size).map_err(|e| {
            BuffersError {
                requested: image_size,
                buf: BufferName::Tile4(e.buf),
                err: e.err,
            }
        })?;

        let timestamps = if has_timestamps {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamp query set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }))
        } else {
            None
        };

        Ok(Self {
            config_buf,
            image_size,
            tile_tapes,
            tile64,
            tile16,
            tile4,
            voxels,
            heightmap,
            geom,
            image,
            timestamps,
            ts_buf,
            bind_groups: Default::default(),
        })
    }

    fn render_size(&self) -> TileRenderSize {
        self.image_size.into()
    }

    /// Returns the size of one strata (in bytes)
    fn strata_size_bytes(&self) -> u64 {
        strata_size_bytes(self.render_size())
    }

    /// Returns the number of bytes in the `tile_tapes` buffer
    ///
    /// The tile tape array is... complicated
    ///
    /// The first `nx * ny * nz` words are tape indices for the root tiles
    /// (64³), densely allocated in x / y / z order.  This is
    /// straight-forward.
    ///
    /// After that point, it gets weirder.  At any given point in time, we're
    /// evaluating a single strata (i.e. a 64-voxel deep slice of the image).
    /// We allocated enough tape words for that strata, also in x / y / z
    /// order, but z is limited to either 0..4 for the 16^3 subtiles, or
    /// 0..16 for 4^3 subtiles.
    ///
    /// In other words, it looks something like this:
    ///
    /// ```text
    /// | index | index | index | ... |     densely packed 64³ tape indices
    /// | index | index | index | ... |     16² XY tiles × 4  Z positions
    /// | index | index | index | ... |     4²  XY tiles × 16 Z positions
    /// ```
    fn tile_tapes_buf_size(render_size: TileRenderSize) -> u64 {
        let nx = u64::from(render_size.nx());
        let ny = u64::from(render_size.ny());
        let nz = u64::from(render_size.nz());
        (nx * ny * nz + (nx * ny) * ((64u64 / 16).pow(3) + (64u64 / 4).pow(3)))
            * std::mem::size_of::<u32>() as u64
    }

    fn voxels_buf_size(render_size: TileRenderSize) -> u64 {
        let render_pixels = render_size.pixels();
        (render_pixels * std::mem::size_of::<u32>()) as u64
    }

    /// Returns the size in bytes for the `geom` buffers
    fn geom_buf_size(image_size: VoxelSize) -> u64 {
        let image_pixels =
            u64::from(image_size.width()) * u64::from(image_size.height());
        image_pixels * std::mem::size_of::<GeometryPixel>() as u64
    }

    fn image_buf_size(image_size: VoxelSize) -> u64 {
        // Allocate an extra 16 bytes for timestamp queries
        Self::geom_buf_size(image_size) + 16
    }

    /// Returns the size in bytes for the `heightmap` buffer
    fn heightmap_buf_size(image_size: VoxelSize) -> u64 {
        let image_pixels =
            u64::from(image_size.width()) * u64::from(image_size.height());

        image_pixels * std::mem::size_of::<u32>() as u64
    }

    /// Resizes to render the target image size
    ///
    /// Internal buffers are resized to fit (only getting larger)
    fn set_image_size(
        &mut self,
        device: &wgpu::Device,
        image_size: VoxelSize,
    ) -> Result<(), BuffersError> {
        let render_size = TileRenderSize::from(image_size);
        let Buffers {
            image_size: image_size_ref,
            config_buf: _,
            tile_tapes,
            tile64,
            tile16,
            tile4,
            voxels,
            heightmap,
            geom,
            image,
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
                requested: image_size,
                buf: BufferName::TileTapes,
                err,
            })?;
        tile64
            .grow_to_fit(device, render_size)
            .map_err(|e| BuffersError {
                requested: image_size,
                buf: BufferName::Tile64(e.buf),
                err: e.err,
            })?;
        tile16
            .grow_to_fit(device, render_size)
            .map_err(|e| BuffersError {
                requested: image_size,
                buf: BufferName::Tile16(e.buf),
                err: e.err,
            })?;
        tile4
            .grow_to_fit(device, render_size)
            .map_err(|e| BuffersError {
                requested: image_size,
                buf: BufferName::Tile4(e.buf),
                err: e.err,
            })?;

        voxels
            .grow_to_fit(device, Self::voxels_buf_size(render_size))
            .map_err(|err| BuffersError {
                requested: image_size,
                buf: BufferName::Voxels,
                err,
            })?;
        heightmap
            .grow_to_fit(device, Self::heightmap_buf_size(image_size))
            .map_err(|err| BuffersError {
                requested: image_size,
                buf: BufferName::Heightmap,
                err,
            })?;
        geom.grow_to_fit(device, Self::geom_buf_size(image_size))
            .map_err(|err| BuffersError {
                requested: image_size,
                buf: BufferName::Geom,
                err,
            })?;
        image
            .grow_to_fit(device, Self::image_buf_size(image_size))
            .map_err(|err| BuffersError {
                requested: image_size,
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
            tile16,
            tile4,
            voxels,
            heightmap,
            geom,
            image,
            timestamps: _,
            ts_buf,
            bind_groups: _,
        } = self;
        config_buf.size()
            + tile_tapes.capacity()
            + tile64.capacity()
            + tile16.capacity()
            + tile4.capacity()
            + voxels.capacity()
            + heightmap.capacity()
            + geom.capacity()
            + image.capacity()
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
            tile16,
            tile4,
            voxels,
            heightmap,
            geom,
            image,
            timestamps: _,
            ts_buf,
            bind_groups: _,
        } = self;
        config_buf.size()
            + tile_tapes.size()
            + tile64.size()
            + tile16.size()
            + tile4.size()
            + voxels.size()
            + heightmap.size()
            + geom.size()
            + image.size()
            + ts_buf.size()
    }
}

impl Context {
    /// Build a new 3D rendering context, given a device and queue
    ///
    /// If render timestamps are desirable, then the device should be
    /// initialized with [`wgpu::Features::TIMESTAMP_QUERY`].
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let has_timestamps =
            device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if !has_timestamps {
            log::warn!(
                "WGPU device is missing `TIMESTAMP_QUERY`; \
                 timestamps are disabled"
            );
        }

        // Create bind group layout and bind group
        let common_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // config (including tape buffer)
                    buffer_rw(1), // tile_tape (hierarchical)
                ],
            });

        let root_ctx = RootContext::new(&device, &common_bind_group_layout);
        let repack_ctx = RepackContext::new(&device, &common_bind_group_layout);
        let interval_ctx =
            IntervalContext::new(&device, &common_bind_group_layout);
        let voxel_ctx = VoxelContext::new(&device, &common_bind_group_layout);
        let normals_ctx =
            NormalsContext::new(&device, &common_bind_group_layout);
        let merge_ctx = MergeContext::new(&device, &common_bind_group_layout);
        let reset_ctx = ResetContext::new();
        let clear_ctx = ClearContext::new(&device, &common_bind_group_layout);

        Self {
            device,
            queue,
            has_timestamps,
            bind_group_layout: common_bind_group_layout,
            root_ctx,
            repack_ctx,
            interval_ctx,
            voxel_ctx,
            normals_ctx,
            merge_ctx,
            reset_ctx,
            clear_ctx,
        }
    }

    /// Builds a new [`Buffers`] object for the given render size
    ///
    /// An image rendered with the resulting buffers will have the given width
    /// and height; `image_size.depth()` sets the number of voxels to evaluate
    /// within each pixel of the image (stacked into a column going into the
    /// screen).
    pub fn buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<Buffers, BuffersError> {
        Buffers::new(&self.device, image_size, self.has_timestamps)
    }

    /// Builds a new [`RenderShape`] object for the given shape
    pub fn shape(
        &self,
        shape: &VmShape,
    ) -> Result<RenderShape, RenderShapeError> {
        RenderShape::new(shape)
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is not present when built for the `wasm32` target
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &self,
        shape: &RenderShape,
        buffers: &mut Buffers,
        settings: RenderConfig,
    ) -> Image {
        self.submit(shape, buffers, &settings);
        let image = self.map_image(buffers);
        image.image()
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is only relevant for the web target
    #[cfg(any(target_arch = "wasm32", doc))]
    pub async fn run_async(
        &self,
        shape: &RenderShape,
        buffers: &mut Buffers,
        settings: RenderConfig,
    ) -> Image {
        self.submit(shape, buffers, &settings);
        let image = self.map_image_async(buffers).await;
        image.image()
    }

    /// Submits a single image to be rendered using GPU acceleration
    pub fn submit(
        &self,
        shape: &RenderShape,
        buffers: &Buffers,
        settings: &RenderConfig,
    ) {
        let render_size = TileRenderSize::from(buffers.image_size);

        let mat =
            settings.world_to_model * buffers.image_size.screen_to_world();

        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes,
            render_size: [
                render_size.width(),
                render_size.height(),
                render_size.depth(),
            ],
            tape_data_capacity: TAPE_DATA_CAPACITY.try_into().unwrap(),
            image_size: [
                buffers.image_size.width(),
                buffers.image_size.height(),
                buffers.image_size.depth(),
            ],
            tape_data_offset: shape.bytecode.len().try_into().unwrap(),
            root_tape_len: shape.bytecode.len().try_into().unwrap(),
        };

        {
            // We load the `Config` and shape tape data.
            let config_len = std::mem::size_of_val(&config);
            let mut writer = self
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

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
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
        let bind_group = buffers.bind_groups.common(self, buffers);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // Populate root tiles (64x64x64, densely packed)
        self.root_ctx.run(
            self,
            buffers,
            shape.bytecode.reg_count(),
            render_size,
            &mut compute_pass,
        );
        // Repack root tiles into strata
        self.repack_ctx
            .run(self, buffers, render_size, &mut compute_pass);

        // Evaluate tiles in reverse-Z order by strata (64 voxels deep)
        let strata_count = u64::from(render_size.depth()).div_ceil(64);
        for strata in 0..strata_count {
            self.interval_ctx.run(
                self,
                buffers,
                strata,
                shape.bytecode.reg_count(),
                &mut compute_pass,
            );
            self.voxel_ctx.run(
                self,
                buffers,
                shape.bytecode.reg_count(),
                &mut compute_pass,
            );

            // Merge filled tiles from large -> small, populating the heightmap
            self.merge_ctx.run(self, buffers, &mut compute_pass);
            self.normals_ctx.run(
                self,
                buffers,
                shape.bytecode.reg_count(),
                &mut compute_pass,
            );

            self.clear_ctx.run(self, buffers, &mut compute_pass);
        }
        drop(compute_pass);

        // Resolve the raw GPU ticks into the resolve buffer, then copy them
        // into the last 16 bytes of the image buffer
        if let Some(timestamps) = &buffers.timestamps {
            encoder.resolve_query_set(timestamps, 0..2, &buffers.ts_buf, 0);
            encoder.copy_buffer_to_buffer(
                &buffers.ts_buf,
                0,
                &buffers.image.data,
                buffers.geom.size(), // offset past the image data
                buffers.ts_buf.size(),
            );
        }

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &buffers.geom.data,
            0,
            &buffers.image.data,
            0,
            buffers.geom.size(),
        );

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));
    }

    /// Synchronously maps the image buffer
    ///
    /// The buffers are borrowed exclusively to avoid double-mapping
    ///
    /// This is a blocking function suitable for use on the desktop
    #[cfg(not(target_arch = "wasm32"))]
    pub fn map_image<'a>(&self, buffers: &'a mut Buffers) -> MappedImage<'a> {
        let slice = buffers.image.map_async(|_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        MappedImage {
            buffers,
            slice,
            ns_per_tick: if self.has_timestamps {
                Some(self.queue.get_timestamp_period())
            } else {
                None
            },
        }
    }

    /// Asynchronously maps the image buffer
    ///
    /// The buffers are borrowed exclusively to avoid double-mapping
    ///
    /// This is an `async` function suitable for use in WebAssembly.
    #[cfg(any(target_arch = "wasm32", doc))]
    pub async fn map_image_async<'a>(
        &self,
        buffers: &'a mut Buffers,
    ) -> MappedImage<'a> {
        let (tx, rx) = flume::bounded(0);
        let slice = buffers.image.map_async(move |_| tx.send(()).unwrap());
        rx.recv_async().await.unwrap();
        MappedImage {
            buffers,
            slice,
            ns_per_tick: if self.has_timestamps {
                Some(self.queue.get_timestamp_period())
            } else {
                None
            },
        }
    }

    /// Debug function to read a buffer to a `Vec<T>`
    #[allow(unused)]
    fn read_buffer<T: FromBytes + Immutable + Clone + Copy>(
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

    /// Resizes buffers to the given image size
    ///
    /// Buffer allocations may grow but do not shrink; delete and recreate
    /// buffers if their capacity exceeds their size to a significant degree.
    pub fn set_buffers_image_size(
        &self,
        buffers: &mut Buffers,
        image_size: VoxelSize,
    ) -> Result<(), BuffersError> {
        buffers.set_image_size(&self.device, image_size)
    }
}

/// Handle to a mapped image, which unmaps the image when dropped
pub struct MappedImage<'a> {
    buffers: &'a Buffers,
    slice: wgpu::BufferSlice<'a>,

    /// Nanoseconds per tick, for resolving timestamps
    ns_per_tick: Option<f32>,
}

impl Drop for MappedImage<'_> {
    fn drop(&mut self) {
        self.buffers.image.data.unmap();
    }
}

impl MappedImage<'_> {
    /// Returns the image's data
    pub fn image(&self) -> Image {
        // Get the pixel-populated image
        let result = <[GeometryPixel]>::ref_from_bytes(
            &self.slice.get_mapped_range()[..self.image_bytes()],
        )
        .unwrap()
        .to_owned();
        Image::build(result, self.buffers.image_size).unwrap()
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
        (self.buffers.image_size.width() as usize)
            * (self.buffers.image_size.height() as usize)
            * std::mem::size_of::<GeometryPixel>()
    }
}

struct ClearContext {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ClearContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tile16_count
                    buffer_rw(1), // tile16_sort
                    buffer_rw(2), // tile16_zhist
                    buffer_rw(3), // tile4_count
                    buffer_rw(4), // tile4_sort
                    buffer_rw(5), // tile4_zhist
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        // Compile the shader
        let shader_code = clear_shader();
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("clear"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("clear_main"),
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
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group = buffers.bind_groups.clear(ctx, buffers);
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(1, bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}

struct MergeContext {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl MergeContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader_code = merge_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tile64_zmin
                    buffer_rw(1), // tile16_zmin
                    buffer_rw(2), // tile4_zmin
                    buffer_ro(3), // voxels
                    buffer_rw(4), // heightmap
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(common_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0u32,
            });

        // Compile the shader
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
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
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let render_size = buffers.render_size();
        let bind_group = buffers.bind_groups.merge(ctx, buffers);
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(1, bind_group, &[]);
        compute_pass.dispatch_workgroups(
            render_size.width().div_ceil(8),
            render_size.height().div_ceil(8),
            1,
        );
    }
}

struct ResetContext;

impl ResetContext {
    fn new() -> Self {
        ResetContext
    }

    fn run(&self, encoder: &mut wgpu::CommandEncoder, buffers: &Buffers) {
        // Clear only the `count` member of the tile64 `tiles_out` buffer
        encoder.clear_buffer(&buffers.tile64.tiles.data, 12, Some(4));

        // Per-strata counters may now be at a different location in memory if
        // we're using the buffers for multiple renders of different sizes!  To
        // be safe, we'll clear them here, rather than in a render pass.
        let strata_size_bytes = buffers.strata_size_bytes();
        for s in 0..buffers.render_size().nz() {
            encoder.clear_buffer(
                &buffers.tile64.strata.data,
                u64::from(s) * strata_size_bytes,
                Some(16),
            );
        }

        // Clear all of the heightmaps and output maps
        buffers.tile64.zmin.clear(encoder);
        buffers.tile64.zmax.clear(encoder);
        buffers.tile16.zmin.clear(encoder);
        buffers.tile4.zmin.clear(encoder);
        buffers.voxels.clear(encoder);
        buffers.heightmap.clear(encoder);
        buffers.geom.clear(encoder);

        // Clear the whole tile tape map (TODO is this needed?)
        buffers.tile_tapes.clear(encoder);

        // tiles / sorted counters and z_hist are reset in clear shader
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use heck::ToShoutySnakeCase;

    #[test]
    fn shader_has_all_ops() {
        for (op, _) in fidget_bytecode::iter_ops() {
            let op = format!("OP_{}", op.to_shouty_snake_case());
            assert!(
                TAPE_INTERPRETER.contains(&op),
                "tape interpreter is missing {op}"
            );
            assert!(
                TAPE_SIMPLIFY.contains(&op),
                "tape simplification is missing {op}"
            );
        }
    }

    #[test]
    fn compile_shaders() {
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        for (src, desc) in [
            (interval_root_shader(16), "interval root"),
            (interval_tiles_shader(16), "interval tiles"),
            (voxel_tiles_shader(16), "voxel tiles"),
            (normals_shader(16), "normals tiles"),
            (repack_shader(), "repack"),
            (sort_shader(), "sort"),
            (merge_shader(), "merge"),
            (clear_shader(), "clear"),
        ] {
            // This isn't the best formatting, but it will at least include the
            // relevant text.
            let m = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                if let Some(i) = e.location(&src) {
                    let pos = i.offset as usize..(i.offset + i.length) as usize;
                    panic!(
                        "shader compilation failed\n{src}\n{}",
                        e.emit_to_string_with_path(&src[pos], desc)
                    );
                } else {
                    panic!(
                        "shader compilation failed\n{src}\n{}",
                        e.emit_to_string(desc)
                    );
                }
            });
            if let Err(e) = v.validate(&m) {
                let (pos, desc) = e.spans().next().unwrap();
                panic!(
                    "shader compilation failed\n{src}\n{}",
                    e.emit_to_string_with_path(
                        &src[pos.to_range().unwrap()],
                        desc
                    )
                );
            }
        }
    }
}
