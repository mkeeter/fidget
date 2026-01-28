//! GPU-accelerated 3D rendering
//!
//! # Theory
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
//! After the root tile evaluation, tiles are sparse.  Tiles and tapes uses an
//! atomic bump allocator to claim portions of a fixed buffer.  The tile buffer
//! is always sized to fit all possible tiles; the tape buffer can run out of
//! space, in which case we fall back to the previous (unsimplified) tape.
//!
//! ## Changes versus MPR
//! There are a few notable changes compared to the MPR paper and reference
//! implementation.
//!
//! First, modern GPU APIs support indirect dispatch based on buffers on the GPU
//! itself.  This saves a round-trip: the 64³ shader can compute a dispatch size
//! for the 16³ shader and store it in a buffer.
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
//!   is expensive to build and should be constructed once per thread / worker.
//! - [`RenderShape`] contains a GPU buffer with serialized bytecode to render a
//!   particular shape.  It is somewhat expensive to build and should only be
//!   constructed when a shape changes (i.e. not once per frame)
//! - [`Buffers`] contains GPU buffers needed for rendering at a particular
//!   image size.  It is somewhat expensive to build and should only be
//!   constructed when the target image size changes (i.e. not once per frame).
//!   It may be wise to store a small cache of "image size → `Buffers`".
//! - [`RenderConfig`] sets the transform matrix for rendering.  This is cheap
//!   to construct and could be built once per frame
//!
//! Usage is pretty simple:
//! - Build a [`Context`]
//! - Use [`Context::shape`] to convert from a [`VmShape`] to a [`RenderShape`]
//! - Use [`Context::buffers`] to get [`Buffers`] at a particular image size
//! - Call [`Context::run`] or [`Context::run_async`] to get an image
//!
//! Right now, there's no support for operating solely on the GPU; both `run`
//! functions read back the resulting buffer into CPU memory.
//!
//! ## Sync and async operation

use crate::{opcode_constants, util::new_buffer};
use fidget_bytecode::Bytecode;
use fidget_core::{eval::Function, render::VoxelSize, vm::VmShape};
use fidget_raster::{GeometryBuffer, GeometryPixel};
use std::collections::BTreeMap;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const VOXEL_TILES_SHADER: &str = include_str!("shaders/voxel_tiles.wgsl");
const STACK_SHADER: &str = include_str!("shaders/stack.wgsl");
const DUMMY_STACK_SHADER: &str = include_str!("shaders/dummy_stack.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_OPS_SHADER: &str = include_str!("shaders/interval_ops.wgsl");
const BACKFILL_SHADER: &str = include_str!("shaders/backfill.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const NORMALS_SHADER: &str = include_str!("shaders/normals.wgsl");
const TAPE_INTERPRETER: &str = include_str!("shaders/tape_interpreter.wgsl");
const TAPE_SIMPLIFY: &str = include_str!("shaders/tape_simplify.wgsl");

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
/// Fields are carefully ordered to require no padding
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
struct RenderSize(VoxelSize);

impl From<VoxelSize> for RenderSize {
    fn from(image_size: VoxelSize) -> Self {
        let nx = image_size.width().div_ceil(64);
        let ny = image_size.height().div_ceil(64);
        let nz = image_size.depth().div_ceil(64);
        Self(VoxelSize::new(nx, ny, nz))
    }
}

impl RenderSize {
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

    /// Number of voxels in the X axis
    fn width(&self) -> u32 {
        self.0.width() * 64
    }

    /// Number of voxels in the Y axis
    fn height(&self) -> u32 {
        self.0.height() * 64
    }

    /// Number of voxels in the Z axis
    fn depth(&self) -> u32 {
        self.0.depth() * 64
    }

    /// Number of pixels in total
    fn pixels(&self) -> usize {
        self.width() as usize * self.height() as usize
    }
}

/// Number of `u32` words in the tape data flexible array
const TAPE_DATA_CAPACITY: usize = 1024 * 1024 * 16; // 16M words, 64 MiB

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

/// Returns a shader for interval tile evaluation
fn voxel_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += VOXEL_TILES_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += DUMMY_STACK_SHADER;
    shader_code
}

/// Returns a shader for interval tile evaluation
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

/// Returns a shader for backfilling tile `zmin` values
fn backfill_shader() -> String {
    BACKFILL_SHADER.to_owned() + COMMON_SHADER
}

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
    fn get(&self, reg_count: u8) -> &wgpu::ComputePipeline {
        let (r, v) = self.0.range(reg_count..).next().unwrap();
        assert!(*r >= reg_count);
        v
    }
}

struct RootContext {
    /// Pipelines for 64^3 tile evaluation
    root_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Per-strata offset in the root tiles list
///
/// This must be equivalent to `strata_size_bytes` in the interval root shader
fn strata_size_bytes(render_size: RenderSize) -> usize {
    let nx = render_size.nx() as usize;
    let ny = render_size.ny() as usize;
    // Snap to `min_storage_buffer_offset_alignment`
    ((nx * ny + 4) * std::mem::size_of::<u32>()).next_multiple_of(256)
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
                ],
            });

        let root_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_root_shader(reg_count);
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        common_bind_group_layout,
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[],
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
        render_size: RenderSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.tile64.tiles.as_entire_binding(),
                }],
            });

        compute_pass.set_pipeline(self.root_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // Workgroup is 4x4x4, so we divide by 4 here on each axis
        let nx = render_size.nx().div_ceil(4);
        let ny = render_size.ny().div_ceil(4);
        let nz = render_size.nz().div_ceil(4);
        compute_pass.dispatch_workgroups(nx, ny, nz);
    }
}

////////////////////////////////////////////////////////////////////////////////

struct IntervalContext {
    /// Pipeline for 64^3 -> 16^3 tile evaluation
    interval64_pipeline: RegPipeline,

    /// Pipeline for 16^3 -> 4^3 tile evaluation
    interval16_pipeline: RegPipeline,

    /// Bind group layout (same for both pipelines)
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IntervalContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tiles_in
                    buffer_ro(1), // tile_zmin
                    buffer_rw(2), // subtiles_out
                    buffer_rw(3), // subtile_zmin
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
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
                    },
                )
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval64 ({reg_count})")),
                layout: Some(&pipeline_layout),
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
                    },
                )
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("interval16 ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 16.0), ("SUBTILE_SIZE", 4.0)],
                    ..Default::default()
                },
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            interval64_pipeline,
            interval16_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        strata: usize,
        reg_count: u8,
        render_size: RenderSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let offset_bytes = (strata * strata_size_bytes(render_size)) as u64;
        let bind_group16 =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers
                            .tile64
                            .tiles
                            .slice(offset_bytes..)
                            .into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile64.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile16.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile16.zmin.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_pipeline(self.interval64_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group16, &[]);
        compute_pass.dispatch_workgroups_indirect(
            &buffers.tile64.tiles,
            offset_bytes as u64,
        );

        let bind_group4 =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile16.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile16.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile4.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile4.zmin.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_pipeline(self.interval16_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group4, &[]);
        compute_pass.dispatch_workgroups_indirect(&buffers.tile16.tiles, 0);
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
                        common_bind_group_layout,
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[],
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
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile4.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile4.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.result.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.voxel_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // Each workgroup is 4x4x4, i.e. covering a 4x4 splat of pixels with 4x
        // workers in the Z direction.
        compute_pass.dispatch_workgroups_indirect(&buffers.tile4.tiles, 0);
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
                    buffer_ro(0), // tiles_in
                    buffer_ro(1), // image_heightmap
                    buffer_rw(2), // image_out
                ],
            });

        let normals_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = normals_shader(reg_count);
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        common_bind_group_layout,
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[],
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
        strata: usize,
        reg_count: u8,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let render_size = buffers.render_size();
        let offset_bytes = (strata * strata_size_bytes(render_size)) as u64;

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers
                            .tile64
                            .tiles
                            .slice(offset_bytes..)
                            .into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.merged.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.geom.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.normals_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        compute_pass.dispatch_workgroups_indirect(
            &buffers.tile64.tiles,
            offset_bytes as u64,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 3D (combined heightmap and normal) rendering
pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Bind group layout for the common bind group (used by all stages)
    bind_group_layout: wgpu::BindGroupLayout,

    root_ctx: RootContext,
    interval_ctx: IntervalContext,
    voxel_ctx: VoxelContext,
    normals_ctx: NormalsContext,
    backfill_ctx: BackfillContext,
    merge_ctx: MergeContext,
    clear_ctx: ClearContext,
}

struct TileBuffers<const N: usize> {
    tiles: wgpu::Buffer,
    zmin: wgpu::Buffer,
}

impl<const N: usize> TileBuffers<N> {
    /// Returns a new `TileBuffers` object
    fn new(device: &wgpu::Device, render_size: RenderSize) -> Self {
        let nx = render_size.width() as usize / N;
        let ny = render_size.height() as usize / N;
        let nz = 64 / N;

        let tiles = new_buffer::<u32>(
            device,
            format!("active_tile{N}"),
            // wg_dispatch: [u32; 3]
            // count: u32,
            4 + nx * ny * nz,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let zmin = new_buffer::<u32>(
            device,
            format!("tile{N}_zmin"),
            nx * ny,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        Self { tiles, zmin }
    }
}

/// Root tile buffers store strata-packed tile lists
struct RootTileBuffers<const N: usize> {
    tiles: wgpu::Buffer,
    zmin: wgpu::Buffer,
}

impl<const N: usize> RootTileBuffers<N> {
    /// Build a new root tiles buffer, which stores strata-packed tile lists
    fn new(device: &wgpu::Device, render_size: RenderSize) -> Self {
        assert_eq!(N, 64);
        let nx = render_size.nx() as usize;
        let ny = render_size.ny() as usize;
        let nz = render_size.nz() as usize;

        let strata_size = strata_size_bytes(render_size);
        let total_size = strata_size * nz;

        let tiles = new_buffer::<u8>(
            // bytes
            device,
            format!("active_tile{N}"),
            // wg_dispatch: [u32; 3]
            // count: u32,
            total_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let zmin = new_buffer::<u32>(
            device,
            format!("tile{N}_zmin"),
            nx * ny,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        Self { tiles, zmin }
    }
}

/// Shape for rendering
///
/// This object is constructed by [`Context::shape`] and may only be used with
/// that particular [`Context`].
pub struct RenderShape {
    config_buf: wgpu::Buffer,
    axes: [u32; 3],
    bytecode_len: u32,
    reg_count: u8,
}

impl RenderShape {
    fn new(device: &wgpu::Device, shape: &VmShape) -> Self {
        // The config buffer is statically sized
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<u32>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        // Generate bytecode for the root tape
        let vars = shape.inner().vars();
        let bytecode = Bytecode::new(shape.inner().data());
        assert!(bytecode.len() < TAPE_DATA_CAPACITY);

        // Create the 4x4 transform matrix
        let axes = shape
            .axes()
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX));

        let offset = std::mem::size_of::<Config>();
        {
            let mut buffer_view =
                config_buf.slice(offset as u64..).get_mapped_range_mut();
            let bytes = bytecode.as_bytes();
            buffer_view[..bytes.len()].copy_from_slice(bytes);
        }
        config_buf.unmap();

        // TODO send shape bytecode to GPU
        Self {
            config_buf,
            axes,
            bytecode_len: bytecode.len().try_into().unwrap(),
            reg_count: bytecode.reg_count(),
        }
    }
}

/// Buffers for rendering, which control the rendered image size
///
/// This object is constructed by [`Context::buffers`] and may only be used with
/// that particular [`Context`].
pub struct Buffers {
    /// Image render size
    image_size: VoxelSize,

    /// Map from tile to the relevant tape (as a start index)
    tile_tapes: wgpu::Buffer,

    /// Root tiles (64^3)
    tile64: RootTileBuffers<64>,

    /// First stage output tiles (16^3)
    tile16: TileBuffers<16>,

    /// Second stage output tiles (4^3)
    tile4: TileBuffers<4>,

    /// Voxel heightmap data
    ///
    /// This buffer has sizes that are multiples of 64 voxels on each dimension
    ///
    /// (dynamic size, implicit from image size in config)
    result: wgpu::Buffer,

    /// Combined result buffer, at the target image size
    ///
    /// Note that this buffer can't be read by the host; it must be copied to
    /// [`image`](Self::image)
    merged: wgpu::Buffer,

    /// Buffer of `GeometryPixel` equivalent data, generated by the normal pass
    geom: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    ///
    /// (dynamic size, implicit from image size in config)
    image: wgpu::Buffer,
}

impl Buffers {
    fn new(device: &wgpu::Device, image_size: VoxelSize) -> Self {
        let render_size = RenderSize::from(image_size);
        let render_pixels = render_size.pixels();
        let result = new_buffer::<u32>(
            device,
            "result",
            render_pixels,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // Resize tile_tape_buf, which must be big enough to contain all of our
        // root tapes, followed by a full strata's worth of subsequent tapes (in
        // a hierarchy).
        let nx = render_size.nx() as usize;
        let ny = render_size.ny() as usize;
        let nz = render_size.nz() as usize;

        // The tile tape array is... complicated
        //
        // The first `nx * ny * nz` words are tape indices for the root tiles
        // (64^3), densely allocated in x / y / z order.  This is
        // straight-forward.
        //
        // After that point, it gets weirder.  At any given point in time, we're
        // evaluating a single strata (i.e. a 64-voxel deep slice of the image).
        // We allocated enough tape words for that strata, also in x / y / z
        // order, but z is limited to either 0..4 for the 16^3 subtiles, or
        // 0..16 for 4^3 subtiles.  We also need to track *which* strata is
        // represented by the tape word, so we double this allocation.
        //
        // In other words, it looks something like this:
        //
        // | index | index | index | ... | densely packed 64^3 tape indices
        // | index | z | index | z | ... | 16^3 data
        // | index | z | index | z | ... | 4^3 data
        let tile_tape_words = nx * ny * nz
            + 2 * (nx * ny * (64usize / 16).pow(3)
                + nx * ny * (64usize / 4).pow(3));
        let tile_tapes = new_buffer::<u32>(
            device,
            "tile tape",
            tile_tape_words,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let image_pixels =
            image_size.width() as usize * image_size.height() as usize;
        let merged = new_buffer::<u32>(
            device,
            "merged",
            image_pixels,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let geom = new_buffer::<GeometryPixel>(
            device,
            "geom",
            image_pixels,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let image = new_buffer::<GeometryPixel>(
            device,
            "image",
            image_pixels,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        let tile64 = RootTileBuffers::new(device, render_size);
        let tile16 = TileBuffers::new(device, render_size);
        let tile4 = TileBuffers::new(device, render_size);

        Self {
            image_size,
            tile_tapes,
            tile64,
            tile16,
            tile4,
            result,
            image,
            merged,
            geom,
        }
    }

    fn render_size(&self) -> RenderSize {
        self.image_size.into()
    }
}

impl Context {
    /// Build a new 3D rendering context, given a device and queue
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
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
        let interval_ctx =
            IntervalContext::new(&device, &common_bind_group_layout);
        let voxel_ctx = VoxelContext::new(&device, &common_bind_group_layout);
        let normals_ctx =
            NormalsContext::new(&device, &common_bind_group_layout);
        let backfill_ctx =
            BackfillContext::new(&device, &common_bind_group_layout);
        let merge_ctx = MergeContext::new(&device, &common_bind_group_layout);
        let clear_ctx = ClearContext::new();

        Self {
            device,
            queue,
            bind_group_layout: common_bind_group_layout,
            root_ctx,
            interval_ctx,
            voxel_ctx,
            normals_ctx,
            backfill_ctx,
            merge_ctx,
            clear_ctx,
        }
    }

    /// Builds a new [`Buffers`] object for the given render size
    ///
    /// An image rendered with the resulting buffers will have the given width
    /// and height; `image_size.depth()` sets the number of voxels to evaluate
    /// within each pixel of the image (stacked into a column going into the
    /// screen).
    pub fn buffers(&self, image_size: VoxelSize) -> Buffers {
        Buffers::new(&self.device, image_size)
    }

    /// Builds a new [`RenderShape`] object for the given shape
    pub fn shape(&self, shape: &VmShape) -> RenderShape {
        RenderShape::new(&self.device, shape)
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is not present when built for the `wasm32` target
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &mut self,
        shape: &RenderShape,
        buffers: &Buffers,
        settings: RenderConfig,
    ) -> GeometryBuffer {
        self.submit(shape, buffers, &settings);
        let buffer_slice = self.map_image(buffers);
        self.read_mapped_image(buffers, &buffer_slice)
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    ///
    /// This function is only relevant for the web target
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(
        &mut self,
        shape: &RenderShape,
        buffers: &Buffers,
        settings: RenderConfig,
    ) -> GeometryBuffer {
        self.submit(shape, buffers, &settings);
        let buffer_slice = self.map_image_async(buffers).await;
        self.read_mapped_image(buffers, &buffer_slice)
    }

    /// Submits a single image to be rendered using GPU acceleration
    fn submit(
        &mut self,
        shape: &RenderShape,
        buffers: &Buffers,
        settings: &RenderConfig,
    ) {
        let render_size = RenderSize::from(buffers.image_size);

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
            tape_data_offset: shape.bytecode_len,
            root_tape_len: shape.bytecode_len,
        };

        {
            // We load the `Config` and the initial tape; the rest of the tape
            // buffer is uninitialized (filled in by the GPU)
            let config_len = std::mem::size_of_val(&config);
            let mut writer = self
                .queue
                .write_buffer_with(
                    &shape.config_buf,
                    0,
                    (config_len as u64).try_into().unwrap(),
                )
                .unwrap();
            writer[0..config_len].copy_from_slice(config.as_bytes());
        }

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        // Initial image clearing pass
        self.clear_ctx.run(&mut encoder, buffers);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

        // Build the common config buffer
        let bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shape.config_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile_tapes.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Populate root tiles (64x64x64, densely packed)
        self.root_ctx.run(
            self,
            buffers,
            shape.reg_count,
            render_size,
            &mut compute_pass,
        );

        // Evaluate tiles in reverse-Z order by strata (64 voxels deep)
        for strata in (0..render_size.depth() as usize / 64).rev() {
            self.interval_ctx.run(
                self,
                buffers,
                strata,
                shape.reg_count,
                render_size,
                &mut compute_pass,
            );
            self.voxel_ctx.run(
                self,
                buffers,
                shape.reg_count,
                &mut compute_pass,
            );

            // It's somewhat overkill to run `merge` after each layer, but it's
            // also very cheap (and we need it to prep for normal_ctx dispatch)
            self.merge_ctx.run(self, buffers, strata, &mut compute_pass);
            self.normals_ctx.run(
                self,
                buffers,
                strata,
                shape.reg_count,
                &mut compute_pass,
            );

            self.backfill_ctx
                .run(self, buffers, strata, &mut compute_pass);
        }
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &buffers.geom,
            0,
            &buffers.image,
            0,
            buffers.geom.size(),
        );

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));
    }

    /// Synchronously maps the image buffer
    ///
    /// This is a blocking function suitable for use on the desktop
    #[cfg(not(target_arch = "wasm32"))]
    fn map_image<'a>(&self, buffers: &'a Buffers) -> wgpu::BufferSlice<'a> {
        // Map result buffer and read back the data
        let buffer_slice = buffers.image.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        buffer_slice
    }

    /// Asynchronously maps the image buffer
    #[cfg(target_arch = "wasm32")]
    async fn map_image_async<'a>(
        &self,
        buffers: &'a Buffers,
    ) -> wgpu::BufferSlice<'a> {
        // Map result buffer and read back the data
        let buffer_slice = buffers.image.slice(..);
        let (tx, rx) = flume::bounded(0); // rendezvous! TODO is this good?
        buffer_slice
            .map_async(wgpu::MapMode::Read, move |_| tx.send(()).unwrap());
        rx.recv_async().await.unwrap();
        buffer_slice
    }

    /// Reads a mapped image from `self.buffers.image`
    ///
    /// # Panics
    /// If we have not yet called `submit` (to submit the job to the GPU) and
    /// either `map_image` or `map_image_async` (to map the image buffer).
    fn read_mapped_image(
        &self,
        buffers: &Buffers,
        buffer_slice: &wgpu::BufferSlice,
    ) -> GeometryBuffer {
        // Get the pixel-populated image
        let result =
            <[GeometryPixel]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        buffers.image.unmap();

        GeometryBuffer::build(result, buffers.image_size).unwrap()
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
}

struct BackfillContext {
    bind_group_layout: wgpu::BindGroupLayout,

    pipeline4: wgpu::ComputePipeline,
    pipeline16: wgpu::ComputePipeline,
    pipeline64: wgpu::ComputePipeline,
}

impl BackfillContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader_code = backfill_shader();
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // subtile_zmin
                    buffer_rw(1), // tile_zmin
                    buffer_rw(2), // count_clear
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // Compile the shader
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        let make_pipeline = |tile_size: u32| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("backfill{tile_size}")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("backfill_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", tile_size as f64)],
                    ..Default::default()
                },
                cache: None,
            })
        };

        let pipeline4 = make_pipeline(4);
        let pipeline16 = make_pipeline(16);
        let pipeline64 = make_pipeline(64);

        Self {
            pipeline64,
            pipeline16,
            pipeline4,
            bind_group_layout,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        buffers: &Buffers,
        strata: usize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let render_size = buffers.render_size();
        let offset_bytes = (strata * strata_size_bytes(render_size)) as u64;

        let nx = render_size.nx() as usize;
        let ny = render_size.ny() as usize;

        let bind_group4 = self.create_bind_group(
            ctx,
            &buffers.result,
            &buffers.tile4.zmin,
            buffers.tile4.tiles.as_entire_binding(),
        );
        compute_pass.set_pipeline(&self.pipeline4);
        compute_pass.set_bind_group(1, &bind_group4, &[]);
        compute_pass.dispatch_workgroups(
            (nx * ny * 16usize.pow(2)).div_ceil(64) as u32,
            1,
            1,
        );

        let bind_group16 = self.create_bind_group(
            ctx,
            &buffers.tile4.zmin,
            &buffers.tile16.zmin,
            buffers.tile16.tiles.as_entire_binding(),
        );
        compute_pass.set_pipeline(&self.pipeline16);
        compute_pass.set_bind_group(1, &bind_group16, &[]);
        compute_pass.dispatch_workgroups(
            (nx * ny * 4usize.pow(2)).div_ceil(64) as u32,
            1,
            1,
        );

        let bind_group64 = self.create_bind_group(
            ctx,
            &buffers.tile16.zmin,
            &buffers.tile64.zmin,
            buffers
                .tile64
                .tiles
                .slice(offset_bytes..offset_bytes + 16)
                .into(),
        );
        compute_pass.set_pipeline(&self.pipeline64);
        compute_pass.set_bind_group(1, &bind_group64, &[]);
        compute_pass.dispatch_workgroups((nx * ny).div_ceil(64) as u32, 1, 1);
    }

    fn create_bind_group(
        &self,
        ctx: &Context,
        subtile_zmin: &wgpu::Buffer,
        tile_zmin: &wgpu::Buffer,
        tile_clear: wgpu::BindingResource,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: subtile_zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_clear,
                },
            ],
        })
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
                    buffer_ro(0), // tile64_zmin
                    buffer_ro(1), // tile16_zmin
                    buffer_ro(2), // tile4_zmin
                    buffer_ro(3), // result
                    buffer_rw(4), // merged
                    buffer_rw(5), // tiles
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
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
        strata: usize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let image_size = buffers.image_size;
        let render_size = buffers.render_size();
        let offset_bytes = (strata * strata_size_bytes(render_size)) as u64;

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.tile64.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile16.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile4.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.merged.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers
                            .tile64
                            .tiles
                            .slice(offset_bytes..)
                            .into(),
                    },
                ],
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            image_size.width().div_ceil(8),
            image_size.height().div_ceil(8),
            1,
        );
    }
}

struct ClearContext;

impl ClearContext {
    fn new() -> Self {
        ClearContext
    }

    fn run(&self, encoder: &mut wgpu::CommandEncoder, buffers: &Buffers) {
        encoder.clear_buffer(&buffers.tile64.zmin, 0, None);
        encoder.clear_buffer(&buffers.tile16.zmin, 0, None);
        encoder.clear_buffer(&buffers.tile4.zmin, 0, None);
        encoder.clear_buffer(&buffers.result, 0, None);
        encoder.clear_buffer(&buffers.merged, 0, None);
        encoder.clear_buffer(&buffers.geom, 0, None);
        encoder.clear_buffer(&buffers.tile_tapes, 0, None);
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
            (merge_shader(), "merge"),
            (backfill_shader(), "backfill"),
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
