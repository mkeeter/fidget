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
const REPACK_SHADER: &str = include_str!("shaders/repack.wgsl");
const VOXEL_TILES_SHADER: &str = include_str!("shaders/voxel_tiles.wgsl");
const STACK_SHADER: &str = include_str!("shaders/stack.wgsl");
const DUMMY_STACK_SHADER: &str = include_str!("shaders/dummy_stack.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_OPS_SHADER: &str = include_str!("shaders/interval_ops.wgsl");
const BACKFILL_SHADER: &str = include_str!("shaders/backfill.wgsl");
const NORMALS_SHADER: &str = include_str!("shaders/normals.wgsl");
const TAPE_INTERPRETER: &str = include_str!("shaders/tape_interpreter.wgsl");
const TAPE_SIMPLIFY: &str = include_str!("shaders/tape_simplify.wgsl");

const DEBUG_FLAGS: wgpu::BufferUsages = if cfg!(test) {
    wgpu::BufferUsages::COPY_SRC
} else {
    wgpu::BufferUsages::empty()
};

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
/// zerocopy derives)
#[derive(Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
#[repr(C)]
struct Config {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],
    _padding1: u32,

    /// Render size, rounded up to the nearest multiple of 64
    render_size: [u32; 3],
    _padding2: u32,

    /// Image size (not rounded)
    image_size: [u32; 3],
    max_tiles_per_dispatch: u32,
}

/// Indirect dispatch plan for a round of interval tile dispatch
#[derive(Copy, Clone, Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
#[repr(C)]
struct Dispatch {
    /// Indirect dispatch size
    tile_dispatch: [u32; 3],
    _padding1: u32,
    backfill_dispatch: [u32; 3],
    _padding2: u32,
}

/// Z value and tape index packed into a single `u32` word
///
/// The top 12 bits are the Z value; the bottom 20 are a tape index.
#[repr(C)]
#[derive(Copy, Clone, Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
struct Voxel(u32);

/// A render size is rounded up to the next multiple of 64 on every axis
///
/// The internal `VoxelSize` stores divided-by-64 values, so that the render
/// size cannot be constructed with an invalid state.
#[derive(Copy, Clone, Debug)]
struct RenderSize(VoxelSize);

impl From<VoxelSize> for RenderSize {
    fn from(image_size: VoxelSize) -> Self {
        assert!(image_size.width() <= 4096);
        assert!(image_size.height() <= 4096);
        assert!(image_size.depth() <= 4096);
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
}

/// Number of [`TapeWord`] objects in the tape data flexible array
///
/// This is limited based on how tapes are packed in the `Voxel` data type, we
/// only have 20 bits available.
const TAPE_DATA_CAPACITY: usize = 1024 * 1024; // 1M TapeWord objects

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

/// Returns a shader for repacking tiles in per-dispatch Z-sorted order
fn repack_shader() -> String {
    REPACK_SHADER.to_owned() + COMMON_SHADER
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

fn uniform_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
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

/// Root context, which produces a list of 64³ tiles
struct RootContext {
    /// Pipelines for 64³ tile evaluation
    root_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
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
                    buffer_rw(0), // tape_data
                    buffer_rw(1), // tile64_out
                    buffer_rw(2), // tile64_zmin
                    buffer_rw(3), // tile64_hist
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
        shape: &RenderShape,
        buffers: &Buffers,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shape.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile64.out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile64.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile64.hist.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.root_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // Workgroup is 4x4x4, so we divide by 4 here on each axis
        let render_size = buffers.render_size();
        let nx = render_size.nx().div_ceil(4);
        let ny = render_size.ny().div_ceil(4);
        let nz = render_size.nz().div_ceil(4);
        compute_pass.dispatch_workgroups(nx, ny, nz);
    }
}

struct RepackContext<const N: usize> {
    repack_pipeline: wgpu::ComputePipeline,
    repack_bind_group_layout: wgpu::BindGroupLayout,
}

impl<const N: usize> RepackContext<N> {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let repack_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tiles
                    buffer_rw(1), // tiles_hist
                    buffer_rw(2), // tile_z_offset
                    buffer_rw(3), // tiles_out
                    buffer_rw(4), // dispatch
                ],
            });

        let repack_pipeline = {
            let shader_code = repack_shader();
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        common_bind_group_layout,
                        &repack_bind_group_layout,
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
                label: Some(&format!("repack{N}")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("repack_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", N as f64)],
                    ..Default::default()
                },
                cache: None,
            })
        };

        Self {
            repack_pipeline,
            repack_bind_group_layout,
        }
    }

    fn run<const M: usize>(
        &self,
        ctx: &Context,
        buffers: &TileBuffers<N>,
        prev: Option<(usize, &TileBuffers<M>)>,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let repack_bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.repack_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.z_to_offset.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.sorted.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.dispatch.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(&self.repack_pipeline);
        compute_pass.set_bind_group(1, &repack_bind_group, &[]);

        if let Some((dispatch, prev)) = prev {
            // If this is an intermediate repacking step, then launch enough
            // workgroups to have one thread per generated tile (which is the
            // same dispatch size as the previous **interval tiles** pass)
            assert_eq!(M, N * 4);
            compute_pass.dispatch_workgroups_indirect(
                &prev.dispatch,
                (dispatch * std::mem::size_of::<Dispatch>()) as u64,
            );
        } else {
            // Otherwise, just launch workgroups based on the number of tiles in
            // our own output buffer
            compute_pass.dispatch_workgroups(
                // Convert from bytes to tiles
                ((buffers.out.size() - 4) / 8)
                    .div_ceil(64)
                    .try_into()
                    .unwrap(),
                1,
                1,
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Single-stage interval evaluation context
///
/// This pipeline accepts a list of tiles of size `(N * 4)³` and produces tiles
/// of size `N³`.
struct IntervalContext<const N: usize> {
    interval_pipeline: RegPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl<const N: usize> IntervalContext<N> {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // dispatch_count
                    buffer_ro(1), // tiles_in
                    buffer_ro(2), // tile_zmin
                    buffer_rw(3), // tape_data
                    buffer_rw(4), // subtiles_out
                    buffer_rw(5), // subtiles_zmin
                    buffer_rw(6), // subtiles_hist
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

        let interval_pipeline = RegPipeline::build(|reg_count| {
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
                label: Some(&format!("interval{N} ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("TILE_SIZE", N as f64 * 4.0),
                        ("SUBTILE_SIZE", N as f64),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            interval_pipeline,
        }
    }

    fn run<const M: usize>(
        &self,
        ctx: &Context,
        shape: &RenderShape,
        tiles_in: &TileBuffers<M>,
        tiles_out: &TileBuffers<N>,
        dispatch: usize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        assert_eq!(N * 4, M);
        let bind_group16 =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    // TODO should the previous stage zero dispatch_count?
                    // Right now, this stage zeroes it on the last dispatch
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tiles_in.dispatch_counter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tiles_in.sorted.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tiles_in.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shape.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tiles_out.out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: tiles_out.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: tiles_out.hist.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_pipeline(self.interval_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group16, &[]);
        compute_pass.dispatch_workgroups_indirect(
            &tiles_in.dispatch,
            (dispatch * std::mem::size_of::<Dispatch>()) as u64,
        );
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
                    buffer_ro(0), // tape_data
                    buffer_ro(1), // tiles4_in
                    buffer_ro(2), // tile4_zmin
                    buffer_rw(3), // result
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
        shape: &RenderShape,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shape.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile4.sorted.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile4.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.voxels.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.voxel_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // There's only a single dispatch for voxel rendering, because looping
        // is done in the kernel itself (if there are more tiles than can be
        // produced in a single dispatch)
        compute_pass.dispatch_workgroups_indirect(&buffers.tile4.dispatch, 0);
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
                    buffer_ro(0), // tape_data
                    buffer_ro(1), // tile64_zmin
                    buffer_ro(2), // tile16_zmin
                    buffer_ro(3), // tile4_zmin
                    buffer_ro(4), // vnoxels
                    buffer_rw(5), // image_out
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
        shape: &RenderShape,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shape.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.tile64.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile16.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile4.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers.geom.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.normals_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

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

    /// Common bind group
    bind_group: wgpu::BindGroup,

    /// Uniform buffer for render configuration
    ///
    /// Written at the start of each rendering operation
    config_buf: wgpu::Buffer,

    /// Context for clearing buffers
    clear_ctx: ClearContext,

    /// Interval root context, generating 64³ tiles
    root_ctx: RootContext,

    /// Cumulative sum and repacking into dispatches for 64³ tiles
    repack64_ctx: RepackContext<64>,

    /// Interval tile context, going from 64³ to 16³ tiles
    interval16_ctx: IntervalContext<16>,

    /// Cumulative sum and repacking into dispatches for 16³ tiles
    repack16_ctx: RepackContext<16>,

    /// Interval tile context, going from 64³ to 4³ tiles
    interval4_ctx: IntervalContext<4>,

    /// Cumulative sum and repacking into dispatches for 4³ tiles
    repack4_ctx: RepackContext<4>,

    /// Voxel evaluation context
    voxel_ctx: VoxelContext,

    /// Backfill context (voxel → 4³ tiles)
    backfill4_ctx: BackfillContext<4>,

    /// Backfill context (4³ → 16³ tiles)
    backfill16_ctx: BackfillContext<16>,

    /// Backfill context (16³ → 64³ tiles)
    backfill64_ctx: BackfillContext<64>,

    /// Normal context for merging `zmin` buffers and computing normals
    normals_ctx: NormalsContext,
}

impl Context {
    /// Build a new 3D rendering context, given a device and queue
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        // Create bind group layout and bind group
        let common_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[uniform_ro(0)],
            });

        let config_buf = new_buffer::<Config>(
            &device,
            "config",
            1,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Build the common config buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &common_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            }],
        });

        let clear_ctx = ClearContext::new();
        let root_ctx = RootContext::new(&device, &common_bind_group_layout);
        let repack64_ctx =
            RepackContext::new(&device, &common_bind_group_layout);
        let interval16_ctx =
            IntervalContext::new(&device, &common_bind_group_layout);
        let repack16_ctx =
            RepackContext::new(&device, &common_bind_group_layout);
        let interval4_ctx =
            IntervalContext::new(&device, &common_bind_group_layout);
        let repack4_ctx =
            RepackContext::new(&device, &common_bind_group_layout);
        let voxel_ctx = VoxelContext::new(&device, &common_bind_group_layout);
        let backfill4_ctx =
            BackfillContext::new(&device, &common_bind_group_layout);
        let backfill16_ctx =
            BackfillContext::new(&device, &common_bind_group_layout);
        let backfill64_ctx =
            BackfillContext::new(&device, &common_bind_group_layout);
        let normals_ctx =
            NormalsContext::new(&device, &common_bind_group_layout);

        Self {
            device,
            queue,
            bind_group,
            config_buf,
            clear_ctx,
            root_ctx,
            repack64_ctx,
            interval16_ctx,
            repack16_ctx,
            interval4_ctx,
            voxel_ctx,
            repack4_ctx,
            backfill4_ctx,
            backfill16_ctx,
            backfill64_ctx,
            normals_ctx,
        }
    }

    /// Builds a new [`Buffers`] object for the given render size
    ///
    /// An image rendered with the resulting buffers will have the given width
    /// and height; `image_size.depth()` sets the number of voxels to evaluate
    /// within each pixel of the image (stacked into a column going into the
    /// screen).
    pub fn buffers(&self, image_size: VoxelSize, spec: GpuSpec) -> Buffers {
        Buffers::new(&self.device, image_size, spec)
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
        &self,
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

    fn load_config(
        &self,
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
            max_tiles_per_dispatch: buffers.max_tiles_per_dispatch as u32,
            image_size: [
                buffers.image_size.width(),
                buffers.image_size.height(),
                buffers.image_size.depth(),
            ],
            _padding1: 0u32,
            _padding2: 0u32,
        };

        let config_len = std::mem::size_of_val(&config);
        let mut writer = self
            .queue
            .write_buffer_with(
                &self.config_buf,
                0,
                (config_len as u64).try_into().unwrap(),
            )
            .unwrap();
        writer[0..config_len].copy_from_slice(config.as_bytes());
    }

    /// Helper function to build a new command encoder
    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            })
    }

    /// Begins a compute pass with the common bind group already bound
    fn begin_compute_pass<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> wgpu::ComputePass<'a> {
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass
    }

    /// Submits a single image to be rendered using GPU acceleration
    fn submit(
        &self,
        shape: &RenderShape,
        buffers: &Buffers,
        settings: &RenderConfig,
    ) {
        self.load_config(shape, buffers, settings);

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.new_encoder();

        // Initial image clearing pass
        self.clear_ctx.run(buffers, shape, &mut encoder);

        let mut compute_pass = self.begin_compute_pass(&mut encoder);

        // Populate root tiles (64x64x64, densely packed)
        self.root_ctx.run(self, shape, buffers, &mut compute_pass);
        self.repack64_ctx.run::<0>(
            self,
            &buffers.tile64,
            None,
            &mut compute_pass,
        );
        for i in 0..buffers.tile64.max_dispatch_count() {
            // Dispatching is indirect based on tile64.dispatch buffers
            self.interval16_ctx.run(
                self,
                shape,
                &buffers.tile64,
                &buffers.tile16,
                i,
                &mut compute_pass,
            );
            self.repack16_ctx.run(
                self,
                &buffers.tile16,
                Some((i, &buffers.tile64)),
                &mut compute_pass,
            );
            for j in 0..buffers.tile16.max_dispatch_count() {
                // Dispatching interval and repacking is indirect based on
                // the tile16.dispatch buffers.
                self.interval4_ctx.run(
                    self,
                    shape,
                    &buffers.tile16,
                    &buffers.tile4,
                    j,
                    &mut compute_pass,
                );
                self.repack4_ctx.run(
                    self,
                    &buffers.tile4,
                    Some((j, &buffers.tile16)),
                    &mut compute_pass,
                );

                // Voxel evaluation is *philosophically* another loop of
                // multiple dispatches (with a max wg count), but we do the
                // looping in the compute shader instead for efficiency.
                self.voxel_ctx.run(self, buffers, shape, &mut compute_pass);
                // end of voxel_ctx virtual loop

                self.backfill4_ctx.run(
                    self,
                    buffers.render_size(),
                    &buffers.voxels,
                    &buffers.tile4,
                    Some((j, &buffers.tile16)),
                    &mut compute_pass,
                );
            }
            self.backfill16_ctx.run(
                self,
                buffers.render_size(),
                &buffers.tile4.zmin,
                &buffers.tile16,
                Some((i, &buffers.tile64)),
                &mut compute_pass,
            );
        }
        self.backfill64_ctx.run::<0>(
            self,
            buffers.render_size(),
            &buffers.tile16.zmin,
            &buffers.tile64,
            None,
            &mut compute_pass,
        );
        self.normals_ctx
            .run(self, buffers, shape, &mut compute_pass);
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &buffers.geom,
            0,
            &buffers.image,
            0,
            buffers.geom.size(),
        );

        // We're all done; submit the commands
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

/// Tile buffers for a stage which produces tiles of size `N`
#[derive(Clone)]
struct TileBuffers<const N: usize> {
    /// Maximum number of output tiles that can be stored in these buffers
    max_tile_count: usize,

    /// Maximum number of tiles processed in one dispatch
    ///
    /// This value should be copied from [`Buffers::max_tiles_per_dispatch`] and
    /// must be the same as [`Config::max_tiles_per_dispatch`].
    max_tiles_per_dispatch: usize,

    /// Unsorted `ActiveTile` list
    ///
    /// This fits `max_tile_count` plus a leading counter word
    ///
    /// Written by the **interval root** or **interval tiles** pass.  Cleared by
    /// writing 0 to the counter word in the backfill pass.
    out: wgpu::Buffer,

    /// Number of tiles in each Z position
    ///
    /// Stores one `u32` per `render_size.z / N`
    ///
    /// Written by the **interval root** or **interval tile** pass, then
    /// modified during the **repack** pass.  Cleared by the backfill pass.
    hist: wgpu::Buffer,

    /// Map from tile Z position to relative buffer offset
    ///
    /// This is scratch space used during the repacking pass.  The repacking
    /// pass writes the value; the backfill pass resets it back to 0.
    ///
    /// Stores one `u32` per `render_size.z / N`
    z_to_offset: wgpu::Buffer,

    /// Array of dispatch objects generated by this stage
    ///
    /// The maximum number of dispatches depends on `N` and the render size
    ///
    /// This buffer is fully written by the **repack** pass.
    dispatch: wgpu::Buffer,

    /// Array of counters to keep track of dispatch for the **next** stage
    ///
    /// There should be one `u32` here for each workgroup
    ///
    /// This value is both written and reset by the **interval tile** pass
    dispatch_counter: wgpu::Buffer,

    /// Output tuple of `(Z, tape index)` for each N³ tile
    ///
    /// This stores one tuple for each tile in the product of
    /// `(render_size / N).[x,y]`
    ///
    /// Written by the tile rendering passes; cleared by the initial clearing
    /// pass.
    zmin: wgpu::Buffer,

    /// Sorted array of `ActiveTile` objects
    ///
    /// This is fully written by repack, and does not need to be cleared.
    sorted: wgpu::Buffer,
}

impl<const N: usize> TileBuffers<N> {
    /// Returns a new `TileBuffers` object
    fn new<const M: usize>(
        device: &wgpu::Device,
        render_size: RenderSize,
        prev: Option<&TileBuffers<M>>,
        max_tiles_per_dispatch: usize,
    ) -> Self {
        let nx = render_size.width() as usize / N;
        let ny = render_size.height() as usize / N;
        let nz = render_size.depth() as usize / N;

        // Figure out how many tiles can be produced by this stage
        let max_tile_count = if let Some(prev) = prev {
            // The previous stage can generate some number of tiles.  It will
            // also plan some number of dispatches; each dispatch can handle up
            // to `max_tiles_per_dispatch`, so that's an upper bound on how many
            // tiles we need to handle.  Each tile from the previous stage can
            // generate up to 64 tiles in this stage.
            assert_eq!(M, N * 4);
            prev.max_tiles_per_dispatch() * 64
        } else {
            // This is a root buffer, so it evaluates every tile in the region
            nx * ny * nz
        };

        // Special case for when the next stage is voxel rendering
        // TODO consolidate with `max_dispatch_count()` function?
        let max_dispatch_count = if N == 4 {
            1
        } else {
            max_tile_count.div_ceil(max_tiles_per_dispatch)
        };

        let out = new_buffer::<u32>(
            device,
            format!("tile{N}_out"),
            // count, then 2 words per tile
            1 + max_tile_count * 2,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );
        let z_to_offset = new_buffer::<u32>(
            device,
            format!("tile{N}_z_to_offset"),
            nz,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );
        let hist = new_buffer::<u32>(
            device,
            format!("tile{N}_hist"),
            nz,
            wgpu::BufferUsages::STORAGE | DEBUG_FLAGS,
        );
        let zmin = new_buffer::<Voxel>(
            device,
            format!("tile{N}_zmin"),
            nx * ny,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );
        let dispatch = new_buffer::<Dispatch>(
            device,
            format!("dispatch{N}"),
            max_dispatch_count,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | DEBUG_FLAGS,
        );
        let dispatch_counter = new_buffer::<u32>(
            device,
            format!("dispatch_counter{N}"),
            max_tile_count.min(max_tiles_per_dispatch),
            wgpu::BufferUsages::STORAGE | DEBUG_FLAGS,
        );
        let sorted = new_buffer::<u32>(
            device,
            format!("tile{N}_sorted"),
            // count, then 2 words per tile
            1 + max_tile_count * 2,
            wgpu::BufferUsages::STORAGE | DEBUG_FLAGS,
        );
        Self {
            max_tile_count,
            max_tiles_per_dispatch,
            out,
            hist, // TODO merge this and zmin into one buffer to minimize binds?
            zmin,
            dispatch,
            dispatch_counter,
            z_to_offset,
            sorted,
        }
    }

    /// Maximum number of tiles to be evaluated per generated dispatch
    ///
    /// This applies to the subsequent rendering stage
    fn max_tiles_per_dispatch(&self) -> usize {
        self.max_tile_count.min(self.max_tiles_per_dispatch)
    }

    /// Maximum number of generated dispatch stages
    ///
    /// This applies to the subsequent rendering stage
    fn max_dispatch_count(&self) -> usize {
        if N == 4 {
            1
        } else {
            self.max_tile_count.div_ceil(self.max_tiles_per_dispatch)
        }
    }
}

/// Shape for rendering
///
/// This object is constructed by [`Context::shape`] and may only be used with
/// that particular [`Context`].
#[derive(Clone)]
pub struct RenderShape {
    /// Tape buffer, containing offset, bytecode length, capacity, and tape data
    ///
    /// The first two words are offset and capacity; subsequent words are tape
    /// data.  Offset is cleared by the initial clearing pass and incremented
    /// whenever shaders write to the tape.  The full tape is located at offset
    /// 0 and is never modified.
    tape_buf: wgpu::Buffer,
    axes: [u32; 3],
    reg_count: u8,
}

/// Doppelganger of the WGSL `struct TapeWord`
#[repr(C)]
struct TapeWord {
    op: u32,
    imm: u32,
}

impl RenderShape {
    fn new(device: &wgpu::Device, shape: &VmShape) -> Self {
        // The config buffer is statically sized
        let tape_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tape_data"),
            size: 12
                + (TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
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

        {
            let mut buffer_view = tape_buf.slice(..).get_mapped_range_mut();
            let bytes = bytecode.as_bytes();
            buffer_view[0..4] // offset
                .copy_from_slice((bytecode.len() as u32 / 2).as_bytes());
            buffer_view[4..8] // base_len
                .copy_from_slice((bytecode.len() as u32 / 2).as_bytes());
            buffer_view[8..12] // capacity
                .copy_from_slice((TAPE_DATA_CAPACITY as u32).as_bytes());
            buffer_view[12..][..bytes.len()].copy_from_slice(bytes);
        }
        tape_buf.unmap();

        Self {
            tape_buf,
            axes,
            reg_count: bytecode.reg_count(),
        }
    }
}

/// Buffers for rendering, which control the rendered image size
///
/// This object is constructed by [`Context::buffers`] and may only be used with
/// that particular [`Context`].
#[derive(Clone)]
pub struct Buffers {
    /// Maximum number of tiles processed in one dispatch
    max_tiles_per_dispatch: usize,

    /// Image render size
    ///
    /// Note that the tile buffers below round up to the nearest root tile
    /// (64³ voxels).
    image_size: VoxelSize,

    /// Buffers for the 64³ pass
    tile64: TileBuffers<64>,

    /// Buffers for the 64³ pass
    tile16: TileBuffers<16>,

    /// Buffers for the 4³ pass
    tile4: TileBuffers<4>,

    /// Buffer of voxel heights, populated by the voxel pass
    voxels: wgpu::Buffer,

    /// Buffer of `GeometryPixel` equivalent data, generated by the normal pass
    geom: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    image: wgpu::Buffer,
}

#[derive(Copy, Clone, Debug)]
pub enum GpuSpec {
    Low,
    Medium,
    High,
    Custom(u32),
}

impl GpuSpec {
    pub fn max_tiles_per_dispatch(&self) -> usize {
        match self {
            GpuSpec::Low => 512,
            GpuSpec::Medium => 4096,
            GpuSpec::High => 32768,
            GpuSpec::Custom(i) => *i as usize,
        }
    }
}

impl Buffers {
    fn new(
        device: &wgpu::Device,
        image_size: VoxelSize,
        spec: GpuSpec,
    ) -> Self {
        let render_size = RenderSize::from(image_size);
        let image_pixels =
            image_size.width() as usize * image_size.height() as usize;
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
        let max_tiles_per_dispatch = spec.max_tiles_per_dispatch();
        let tile64 = TileBuffers::new::<0>(
            device,
            render_size,
            None,
            max_tiles_per_dispatch,
        );
        let tile16 = TileBuffers::new(
            device,
            render_size,
            Some(&tile64),
            max_tiles_per_dispatch,
        );
        let tile4 = TileBuffers::new(
            device,
            render_size,
            Some(&tile16),
            max_tiles_per_dispatch,
        );
        let voxels = new_buffer::<Voxel>(
            device,
            "voxels",
            render_size.width() as usize * render_size.height() as usize,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );

        Self {
            max_tiles_per_dispatch,
            image_size,
            tile64,
            tile16,
            tile4,
            geom,
            image,
            voxels,
        }
    }

    fn render_size(&self) -> RenderSize {
        self.image_size.into()
    }
}

/// Backfills from tiles of size `N` to the parent tiles (of size `N * 4`)
struct BackfillContext<const N: usize> {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl<const N: usize> BackfillContext<N> {
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
                    buffer_rw(2), // tiles_out (to clear count)
                    buffer_rw(3), // tile_hist (to clear counts)
                    buffer_rw(4), // tile_z_offset (to clear counts)
                    buffer_rw(5), // dispatch (to clear counts)
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
                label: Some(&format!("backfill{N}")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("backfill_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", N as f64)],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    fn run<const M: usize>(
        &self,
        ctx: &Context,
        render_size: RenderSize,
        subtile_zmin: &wgpu::Buffer,
        tile_buf: &TileBuffers<N>,
        prev: Option<(usize, &TileBuffers<M>)>,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
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
                        resource: tile_buf.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile_buf.out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tile_buf.hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tile_buf.z_to_offset.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: tile_buf.dispatch.as_entire_binding(),
                    },
                ],
            });

        // Dispatched once per parent tile, on the X axis only
        let nx = render_size.width() as usize / N;
        let ny = render_size.height() as usize / N;
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        if let Some((dispatch, prev)) = prev {
            // If this is an intermediate backfill step, then it may not need to
            // be dispatched at all; we'll use the `backfill_dispatch` member to
            // do an indirect dispatch (or not)
            assert_eq!(M, N * 4);
            compute_pass.dispatch_workgroups_indirect(
                &prev.dispatch,
                (dispatch * std::mem::size_of::<Dispatch>()
                    + std::mem::offset_of!(Dispatch, backfill_dispatch))
                    as u64,
            );
        } else {
            compute_pass.dispatch_workgroups(
                (nx * ny).div_ceil(64) as u32,
                1,
                1,
            );
        }
    }
}

struct ClearContext;

impl ClearContext {
    fn new() -> Self {
        ClearContext
    }

    fn run(
        &self,
        buffers: &Buffers,
        shape: &RenderShape,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        encoder.clear_buffer(&shape.tape_buf, 0, Some(4));
        self.clear_tile_buffers(&buffers.tile64, encoder);
        self.clear_tile_buffers(&buffers.tile16, encoder);
        self.clear_tile_buffers(&buffers.tile4, encoder);
        encoder.clear_buffer(&buffers.voxels, 0, None);
        encoder.clear_buffer(&buffers.geom, 0, None);
    }

    fn clear_tile_buffers<const N: usize>(
        &self,
        buffers: &TileBuffers<N>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        encoder.clear_buffer(&buffers.zmin, 0, None);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use heck::ToShoutySnakeCase;
    use zerocopy::TryFromBytes;

    #[derive(
        Copy, Clone, Debug, IntoBytes, Immutable, FromBytes, KnownLayout,
    )]
    #[repr(C)]
    struct ActiveTile {
        /// Tile position, with x/y/z values packed into a single `u32`
        tile: u32,
        /// Start of this tile's tape in the tape data array
        tape_index: u32,
    }

    /// Bonus methods to deal with bitpacked voxels
    impl Voxel {
        fn z(&self) -> u32 {
            self.0 >> 20
        }
        fn tape_index(&self) -> u32 {
            (self.0 & ((1 << 20) - 1)) * 2
        }
    }

    #[derive(TryFromBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct TileList {
        count: u32,
        tiles: [ActiveTile],
    }

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

    fn compile_shader(src: &str, desc: &str) {
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        // This isn't the best formatting, but it will at least include the
        // relevant text.
        let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
            if let Some(i) = e.location(src) {
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
                e.emit_to_string_with_path(&src[pos.to_range().unwrap()], desc)
            );
        }
    }

    #[test]
    fn check_interval_root_shader() {
        compile_shader(&interval_root_shader(16), "interval root")
    }

    #[test]
    fn check_interval_tile_shader() {
        compile_shader(&interval_tiles_shader(16), "interval tile")
    }

    #[test]
    fn check_voxel_tile_shader() {
        compile_shader(&voxel_tiles_shader(16), "voxel")
    }

    #[test]
    fn check_repack_shader() {
        compile_shader(&repack_shader(), "repack")
    }

    #[test]
    fn check_backfill_shader() {
        compile_shader(&backfill_shader(), "backfill")
    }

    #[test]
    fn check_normals_shader() {
        compile_shader(&normals_shader(8), "normals")
    }

    /// GPU-specific tests
    mod gpu {
        use super::*;
        use fidget_core::{context::Tree, vm::VmShape};
        use std::collections::BTreeSet;

        thread_local! {
            pub static CTX: Context = get_context();
        }

        fn get_context() -> Context {
            let instance = wgpu::Instance::default();
            let (device, queue) = pollster::block_on(async move {
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference:
                            wgpu::PowerPreference::HighPerformance,
                        ..wgpu::RequestAdapterOptions::default()
                    })
                    .await
                    .unwrap();
                adapter
                    .request_device(&wgpu::DeviceDescriptor::default())
                    .await
                    .unwrap()
            });
            Context::new(device, queue)
        }

        // Inner test function to avoid rightward drift
        fn step_by_step_x_(ctx: &Context) {
            const MAX_TILES_PER_DISPATCH: u32 = 32768;

            // 16x16x16 tiles in the image
            let size = VoxelSize::from(1024);
            let buffers =
                ctx.buffers(size, GpuSpec::Custom(MAX_TILES_PER_DISPATCH));
            let shape = ctx.shape(&VmShape::from(Tree::x()));
            let settings = RenderConfig::default();

            ////////////////////////////////////////////////////////////////////
            // First pass: interval root tile evaluator
            //
            // This should produce 512 output tiles with
            //  x = [7, 8]
            //  y = 0..15
            //  z= 0..15
            ctx.load_config(&shape, &buffers, &settings);

            let mut encoder = ctx.new_encoder();
            ctx.clear_ctx.run(&buffers, &shape, &mut encoder);
            ctx.queue.submit(Some(encoder.finish()));

            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.root_ctx.run(ctx, &shape, &buffers, &mut compute_pass);
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            // Make sure that the output tiles are just at the X = 0 origin
            // (so have tile X = 7 or 8) and we have the expected number.
            let out = ctx.read_buffer::<u8>(&buffers.tile64.out);
            let out = TileList::try_ref_from_bytes(&out).unwrap();
            assert_eq!(out.count, 16 * 16 * 2); // Y-Z slice at the origin
            assert_eq!(out.tiles.len(), 16usize.pow(3));
            let mut tiles = BTreeSet::new();
            for t in out.tiles.iter().take(out.count as usize) {
                assert_eq!(t.tape_index, 0);
                let x = t.tile % 16;
                let y = (t.tile / 16) % 16;
                let z = (t.tile / 16) / 16;
                assert!(
                    x == 7 || x == 8,
                    "bad x value {x} in tile {x}, {y}, {z}"
                );
                assert!(tiles.insert(t.tile));
            }
            assert_eq!(tiles.len(), 16 * 16 * 2); // no duplicates

            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile64.zmin);
            assert_eq!(zmin.len(), 16 * 16);
            for x in 0..16 {
                for y in 0..16 {
                    let t = zmin[x + y * 16];
                    assert_eq!(t.tape_index(), 0);
                    let expected = if x < 7 { 1023 } else { 0 };
                    assert_eq!(t.z(), expected, "bad z at tile {x}, {y}");
                }
            }

            let hist = ctx.read_buffer::<u32>(&buffers.tile64.hist);
            assert_eq!(hist.len(), 16);
            for n in hist {
                assert_eq!(n, 32); // strip of 2x16 tiles
            }

            ////////////////////////////////////////////////////////////////////
            // Next up: the repacking pass
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.repack64_ctx.run::<0>(
                ctx,
                &buffers.tile64,
                None,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let tile_z_to_offset =
                ctx.read_buffer::<u32>(&buffers.tile64.z_to_offset);
            assert_eq!(tile_z_to_offset.len(), 16);
            for (i, z) in tile_z_to_offset.iter().enumerate() {
                assert_eq!(*z as usize, 32, "bad offset at {z}, {i}");
            }

            let dispatch =
                ctx.read_buffer::<Dispatch>(&buffers.tile64.dispatch);
            assert_eq!(dispatch.len(), 1);
            assert_eq!(dispatch[0].tile_dispatch, [32 * 16, 1, 1]);
            assert_eq!(
                dispatch[0].backfill_dispatch,
                [16 * 16 * 16 / 64, 1, 1]
            );

            let sorted = ctx.read_buffer::<u8>(&buffers.tile64.sorted);
            let sorted = TileList::try_ref_from_bytes(&sorted).unwrap();
            assert_eq!(sorted.tiles.len(), 16usize.pow(3));
            assert_eq!(sorted.count, 16 * 16 * 2); // Y-Z slice at the origin
            for (i, t) in sorted.tiles[..32 * 16].chunks(32).enumerate() {
                let ys = (0..16).collect::<BTreeSet<u32>>();
                let mut ys = [ys.clone(), ys];
                for t in t {
                    assert_eq!(t.tape_index, 0);
                    let x = t.tile % 16;
                    let y = (t.tile / 16) % 16;
                    let z = (t.tile / 16) / 16;

                    // Ensure that we're in reverse-z order
                    assert_eq!(z as usize, 16 - i - 1);
                    // Only middle tiles should be active
                    assert!(x == 7 || x == 8);
                    // All Y values must be represented
                    assert!(ys[x as usize - 7].remove(&y));
                }
                assert!(ys[0].is_empty() && ys[1].is_empty());
            }

            // Repacking uses `hist` for a cumulative sum
            let hist = ctx.read_buffer::<u32>(&buffers.tile64.hist);
            assert_eq!(hist.len(), 16);
            for (i, n) in hist.iter().enumerate() {
                assert_eq!(*n, (32 * (i + 1) as u32) | (1 << 31));
            }

            ////////////////////////////////////////////////////////////////////
            // At this point, we've got a Z-sorted list of 512 tiles with
            //  x = [7, 8]
            //  y = 0..=15
            //  z = 0..=15
            //
            // We'll do a single dispatch of 64->16 tile refinement, which
            // should produce 8192 tiles with
            //  x = [31, 32]
            //  y = 0..=63
            //  z = 0..=63
            assert_eq!(buffers.tile64.max_dispatch_count(), 1);
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.interval16_ctx.run(
                ctx,
                &shape,
                &buffers.tile64,
                &buffers.tile16,
                0,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let dispatch_counter =
                ctx.read_buffer::<u32>(&buffers.tile64.dispatch_counter);
            assert_eq!(dispatch_counter.len(), 4096);
            // all dispatch counters are either left as 0 or reset
            for (i, d) in dispatch_counter.iter().enumerate() {
                assert_eq!(*d, 0, "bad dispatch at {i}");
            }

            let out = ctx.read_buffer::<u8>(&buffers.tile16.out);
            let out = TileList::try_ref_from_bytes(&out).unwrap();
            assert_eq!(out.count, 8192); // Y-Z slice at the origin
            assert_eq!(out.tiles.len(), 64usize.pow(3));
            let mut tiles = BTreeSet::new();
            for t in out.tiles.iter().take(out.count as usize) {
                assert_eq!(t.tape_index, 0);
                let x = t.tile % 64;
                let y = (t.tile / 64) % 64;
                let z = (t.tile / 64) / 64;
                assert!(
                    x == 31 || x == 32,
                    "bad x value {x} in tile {x}, {y}, {z}"
                );
                assert!(tiles.insert(t.tile));
            }
            assert_eq!(tiles.len(), 8192); // no duplicates

            // Check that the zmin buffer is correctly filled out
            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile16.zmin);
            assert_eq!(zmin.len(), 64 * 64);
            for x in 0..64 {
                for y in 0..64 {
                    // The previous stage only produced tiles with x = [7,8]
                    if x / 4 < 7 || x / 4 > 8 {
                        continue;
                    }
                    let t = zmin[x + y * 64];
                    assert_eq!(t.tape_index(), 0);
                    let expected = if x < 31 { 1023 } else { 0 };
                    assert_eq!(t.z(), expected, "bad z at tile {x}, {y}");
                }
            }

            let hist = ctx.read_buffer::<u32>(&buffers.tile16.hist);
            assert_eq!(hist.len(), 64);
            for n in hist {
                assert_eq!(n, 128); // strip of 2x64 tiles
            }

            ////////////////////////////////////////////////////////////////////
            // Next up: the repacking pass for 16³ tiles
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.repack16_ctx.run(
                ctx,
                &buffers.tile16,
                Some((0, &buffers.tile64)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let tile_z_to_offset =
                ctx.read_buffer::<u32>(&buffers.tile16.z_to_offset);
            assert_eq!(tile_z_to_offset.len(), 64);
            for (i, z) in tile_z_to_offset.iter().enumerate() {
                assert_eq!(*z as usize, 128, "bad offset at {z}, {i}");
            }

            // Only dispatch 0 is used, because we don't have > MAX_DISPATH_SIZE
            // tiles generated in this stage.
            let dispatch =
                ctx.read_buffer::<Dispatch>(&buffers.tile16.dispatch);
            assert_eq!(dispatch.len(), buffers.tile16.max_dispatch_count());
            assert_eq!(dispatch[0].tile_dispatch, [2 * 64 * 64, 1, 1]);
            assert_eq!(
                dispatch[0].backfill_dispatch,
                [64 * 64 * 16 / 64, 1, 1]
            );
            for d in &dispatch[1..] {
                assert_eq!(d.tile_dispatch, [0, 0, 0]);
                assert_eq!(d.backfill_dispatch, [0, 0, 0]);
            }

            let sorted = ctx.read_buffer::<u8>(&buffers.tile16.sorted);
            let sorted = TileList::try_ref_from_bytes(&sorted).unwrap();
            for (i, t) in sorted.tiles[..2 * 64 * 64].chunks(128).enumerate() {
                let ys = (0..64).collect::<BTreeSet<u32>>();
                let mut ys = [ys.clone(), ys];
                for t in t {
                    assert_eq!(t.tape_index, 0);
                    let x = t.tile % 64;
                    let y = (t.tile / 64) % 64;
                    let z = (t.tile / 64) / 64;

                    // Ensure that we're in reverse-z order
                    assert_eq!(z as usize, 64 - i - 1);
                    // Only middle tiles should be active
                    assert!(x == 31 || x == 32);
                    // All Y values must be represented
                    assert!(ys[x as usize - 31].remove(&y));
                }
                assert!(ys[0].is_empty() && ys[1].is_empty());
            }

            // Repacking uses `hist` to compute a cumulative sum
            let hist = ctx.read_buffer::<u32>(&buffers.tile16.hist);
            assert_eq!(hist.len(), 64);
            for (i, n) in hist.iter().enumerate() {
                assert_eq!(*n, (128 * (i + 1) as u32) | (1 << 31));
            }

            // Now we're onto the 16³ -> 4³ tile pass
            //
            // We'll do a single dispatch of 16³ -> 4³ tile refinement, which
            // should produce 131072 tiles with
            //  x = [127,128]
            //  y = 0..=256
            //  z = 0..=256

            // The previous stage could plan up to 8 dispatches; in practice,
            // we've asserted that only the first dispatch is used.
            assert_eq!(buffers.tile16.max_dispatch_count(), 8);
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.interval4_ctx.run(
                ctx,
                &shape,
                &buffers.tile16,
                &buffers.tile4,
                0,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let dispatch_counter =
                ctx.read_buffer::<u32>(&buffers.tile16.dispatch_counter);
            assert_eq!(
                dispatch_counter.len(),
                buffers.tile16.max_tiles_per_dispatch()
            );
            // all dispatch counters are either left as 0 or reset
            for (i, d) in dispatch_counter.iter().enumerate() {
                assert_eq!(*d, 0, "bad dispatch at {i}");
            }

            let out = ctx.read_buffer::<u8>(&buffers.tile4.out);
            let out = TileList::try_ref_from_bytes(&out).unwrap();
            assert_eq!(
                out.tiles.len(),
                buffers.tile16.max_tiles_per_dispatch() * 64
            );
            assert_eq!(out.count, 131072); // Y-Z slice at the origin
            let mut tiles = BTreeSet::new();
            for t in out.tiles.iter().take(out.count as usize) {
                assert_eq!(t.tape_index, 0);
                let x = t.tile % 256;
                let y = (t.tile / 256) % 256;
                let z = (t.tile / 256) / 256;
                assert!(
                    x == 127 || x == 128,
                    "bad x value {x} in tile {x}, {y}, {z}"
                );
                assert!(tiles.insert(t.tile));
            }
            assert_eq!(tiles.len(), 131072); // no duplicates

            // Check that the zmin buffer is correctly filled out
            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile4.zmin);
            assert_eq!(zmin.len(), 256 * 256);
            for x in 0..256 {
                for y in 0..256 {
                    // The previous stage only produced tiles with x = [7,8]
                    if x / 4 < 31 || x / 4 > 32 {
                        continue;
                    }
                    let t = zmin[x + y * 256];
                    assert_eq!(t.tape_index(), 0);
                    let expected = if x < 127 { 1023 } else { 0 };
                    assert_eq!(t.z(), expected, "bad z at tile {x}, {y}");
                }
            }

            let hist = ctx.read_buffer::<u32>(&buffers.tile4.hist);
            assert_eq!(hist.len(), 256);
            for n in hist {
                assert_eq!(n, 512); // strip of 2x256 tiles
            }

            ////////////////////////////////////////////////////////////////////
            // Almost done!  Time for the repacking pass for 4³ tiles
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.repack4_ctx.run(
                ctx,
                &buffers.tile4,
                Some((0, &buffers.tile16)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let tile_z_to_offset =
                ctx.read_buffer::<u32>(&buffers.tile4.z_to_offset);
            assert_eq!(tile_z_to_offset.len(), 256);
            for (i, z) in tile_z_to_offset.iter().enumerate() {
                assert_eq!(*z as usize, 512, "bad offset at {z}, {i}");
            }

            let dispatch = ctx.read_buffer::<Dispatch>(&buffers.tile4.dispatch);
            assert_eq!(dispatch.len(), buffers.tile4.max_dispatch_count());
            assert_eq!(
                dispatch[0].tile_dispatch,
                [MAX_TILES_PER_DISPATCH, 1, 1]
            );
            assert_eq!(dispatch[0].backfill_dispatch, [65536 * 16 / 64, 1, 1]);
            for d in &dispatch[1..] {
                assert_eq!(d.tile_dispatch, [0, 0, 0]);
                assert_eq!(d.backfill_dispatch, [0, 0, 0]);
            }

            let sorted = ctx.read_buffer::<u8>(&buffers.tile4.sorted);
            let sorted = TileList::try_ref_from_bytes(&sorted).unwrap();
            assert_eq!(sorted.count, 256 * 256 * 2);
            for (i, t) in sorted.tiles[..sorted.count as usize]
                .chunks(256 * 2)
                .enumerate()
            {
                let ys = (0..256).collect::<BTreeSet<u32>>();
                let mut ys = [ys.clone(), ys];
                for t in t {
                    assert_eq!(t.tape_index, 0);
                    let x = t.tile % 256;
                    let y = (t.tile / 256) % 256;
                    let z = (t.tile / 256) / 256;

                    // Ensure that we're in reverse-z order
                    assert_eq!(z as usize, 256 - i - 1);
                    // Only middle tiles should be active
                    assert!(x == 127 || x == 128);
                    // All Y values must be represented
                    assert!(ys[x as usize - 127].remove(&y));
                }
                assert!(ys[0].is_empty() && ys[1].is_empty());
            }

            // Repacking uses `hist` to compute a cumulative sum
            let hist = ctx.read_buffer::<u32>(&buffers.tile4.hist);
            assert_eq!(hist.len(), 256);
            for (i, n) in hist.iter().enumerate() {
                assert_eq!(*n, (512 * (i + 1) as u32) | (1 << 31));
            }

            ////////////////////////////////////////////////////////////////////
            // It's morging^W voxel time!
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.voxel_ctx.run(ctx, &buffers, &shape, &mut compute_pass);
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let voxels = ctx.read_buffer::<Voxel>(&buffers.voxels);
            assert_eq!(voxels.len(), 1024 * 1024);
            for (y, vs) in voxels.chunks(1024).enumerate() {
                for (x, v) in vs.iter().enumerate() {
                    assert_eq!(v.tape_index(), 0);
                    let expected =
                        if (508..512).contains(&x) { 1023 } else { 0 };
                    assert_eq!(v.z(), expected, "bad value at {x}, {y}");
                }
            }

            ////////////////////////////////////////////////////////////////////
            // Let's run the backfill shader and see how it does
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.backfill4_ctx.run(
                ctx,
                buffers.render_size(),
                &buffers.voxels,
                &buffers.tile4,
                Some((0, &buffers.tile16)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            // Now, the tile4 zmin buffer should be populated all the way up to
            // x = 128 (previously it was x < 128).
            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile4.zmin);
            assert_eq!(zmin.len(), 256 * 256);
            for x in 0..256 {
                for y in 0..256 {
                    // The previous stage only produced tiles with x = [7,8]
                    if x / 4 < 31 || x / 4 > 32 {
                        continue;
                    }
                    let t = zmin[x + y * 256];
                    assert_eq!(t.tape_index(), 0);

                    // this condition changed!
                    let expected = if x <= 127 { 1023 } else { 0 };
                    assert_eq!(t.z(), expected, "bad z at tile {x}, {y}");
                }
            }

            ////////////////////////////////////////////////////////////////////
            // Normals!
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.normals_ctx
                .run(ctx, &buffers, &shape, &mut compute_pass);
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let geom = ctx.read_buffer::<[f32; 4]>(&buffers.geom);
            assert_eq!(geom.len(), 1024 * 1024);
            for (y, vs) in geom.chunks(1024).enumerate() {
                for (x, v) in vs.iter().enumerate() {
                    let expected = if x <= 511 {
                        [1023.0, 1.0 / 512.0, 0.0, 0.0]
                    } else {
                        [0.0; 4]
                    };
                    assert_eq!(v, &expected, "bad pixel at {x}, {y}");
                }
            }
        }

        #[test]
        fn step_by_step_x() {
            CTX.with(step_by_step_x_);
        }

        // Inner test function to avoid rightward drift
        fn step_by_step_y_(ctx: &Context) {
            const MAX_TILES_PER_DISPATCH: u32 = 32768;

            // 1 root tile in the image
            let size = VoxelSize::from(64);
            let buffers =
                ctx.buffers(size, GpuSpec::Custom(MAX_TILES_PER_DISPATCH));
            let shape = ctx.shape(&VmShape::from(Tree::y()));
            let settings = RenderConfig::default();

            ////////////////////////////////////////////////////////////////////
            // First pass: interval root tile evaluator
            //
            // This should produce 1 output tile at 0,0,0
            ctx.load_config(&shape, &buffers, &settings);

            let mut encoder = ctx.new_encoder();
            ctx.clear_ctx.run(&buffers, &shape, &mut encoder);
            ctx.queue.submit(Some(encoder.finish()));

            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.root_ctx.run(ctx, &shape, &buffers, &mut compute_pass);
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let out = ctx.read_buffer::<u8>(&buffers.tile64.out);
            let out = TileList::try_ref_from_bytes(&out).unwrap();
            assert_eq!(out.count, 1);
            assert_eq!(out.tiles.len(), 1);
            assert_eq!(out.tiles[0].tape_index, 0);
            let x = out.tiles[0].tile % 16;
            let y = (out.tiles[0].tile / 16) % 16;
            let z = (out.tiles[0].tile / 16) / 16;
            assert_eq!([x, y, z], [0, 0, 0], "bad value in tile {x}, {y}, {z}");

            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile64.zmin);
            assert_eq!(zmin.len(), 1);
            assert_eq!(zmin[0].tape_index(), 0);
            assert_eq!(zmin[0].z(), 0);

            let hist = ctx.read_buffer::<u32>(&buffers.tile64.hist);
            assert_eq!(hist.len(), 1);
            assert_eq!(hist[0], 1);

            ////////////////////////////////////////////////////////////////////
            // Next up: the repacking pass
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.repack64_ctx.run::<0>(
                ctx,
                &buffers.tile64,
                None,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let tile_z_to_offset =
                ctx.read_buffer::<u32>(&buffers.tile64.z_to_offset);
            assert_eq!(tile_z_to_offset.len(), 1);
            for (i, z) in tile_z_to_offset.iter().enumerate() {
                assert_eq!(*z as usize, 1, "bad offset at {z}, {i}");
            }

            let dispatch =
                ctx.read_buffer::<Dispatch>(&buffers.tile64.dispatch);
            assert_eq!(dispatch.len(), 1);
            for d in dispatch {
                assert_eq!(d.tile_dispatch, [1, 1, 1]);
                assert_eq!(d.backfill_dispatch, [1, 1, 1]);
            }

            let sorted = ctx.read_buffer::<u8>(&buffers.tile64.sorted);
            let sorted = TileList::try_ref_from_bytes(&sorted).unwrap();
            assert_eq!(sorted.tiles.len(), 1);
            assert_eq!(sorted.count, 1); // Y-Z slice at the origin
            assert_eq!(sorted.tiles[0].tape_index, 0);
            assert_eq!(sorted.tiles[0].tile, 0);

            // Repacking uses `hist` for a cumulative sum
            let hist = ctx.read_buffer::<u32>(&buffers.tile64.hist);
            assert_eq!(hist.len(), 1);
            assert_eq!(hist[0], 1 | (1 << 31));

            ////////////////////////////////////////////////////////////////////
            // At this point, we've got a Z-sorted list of 1 tile (0,0,0)
            //
            // We'll do a single dispatch of 64->16 tile refinement, which
            // should produce 16 tiles with
            //  x = 0..4
            //  y = 1
            //  z = 0..4
            assert_eq!(buffers.tile64.max_dispatch_count(), 1);
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.interval16_ctx.run(
                ctx,
                &shape,
                &buffers.tile64,
                &buffers.tile16,
                0,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let dispatch_counter =
                ctx.read_buffer::<u32>(&buffers.tile64.dispatch_counter);
            assert_eq!(dispatch_counter.len(), 1);
            // all dispatch counters are either left as 0 or reset
            for (i, d) in dispatch_counter.iter().enumerate() {
                assert_eq!(*d, 0, "bad dispatch at {i}");
            }

            let out = ctx.read_buffer::<u8>(&buffers.tile16.out);
            let out = TileList::try_ref_from_bytes(&out).unwrap();
            assert_eq!(out.count, 16); // Y-Z slice at the origin
            assert_eq!(out.tiles.len(), 4usize.pow(3));
            let mut tiles = BTreeSet::new();
            for t in out.tiles.iter().take(out.count as usize) {
                assert_eq!(t.tape_index, 0);
                let x = t.tile % 4;
                let y = (t.tile / 4) % 4;
                let z = (t.tile / 4) / 4;
                assert_eq!(y, 1, "bad value in tile {x}, {y}, {z}");
                assert!(tiles.insert(t.tile));
            }
            assert_eq!(tiles.len(), 16); // no duplicates

            // Check that the zmin buffer is correctly filled out
            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile16.zmin);
            assert_eq!(zmin.len(), 4 * 4);
            for y in 0..4 {
                for x in 0..4 {
                    let t = zmin[x + y * 4];
                    assert_eq!(t.tape_index(), 0);
                    let expected = if y > 1 { 63 } else { 0 };
                    assert_eq!(t.z(), expected, "bad z at tile {x}, {y}");
                }
            }

            let hist = ctx.read_buffer::<u32>(&buffers.tile16.hist);
            assert_eq!(hist.len(), 4);
            for n in hist {
                assert_eq!(n, 4); // strip of 1x4 tiles
            }

            ////////////////////////////////////////////////////////////////////
            // Next up: the repacking pass for 16³ tiles
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.repack16_ctx.run(
                ctx,
                &buffers.tile16,
                Some((0, &buffers.tile64)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let tile_z_to_offset =
                ctx.read_buffer::<u32>(&buffers.tile16.z_to_offset);
            assert_eq!(tile_z_to_offset.len(), 4);
            for (i, z) in tile_z_to_offset.iter().enumerate() {
                assert_eq!(*z as usize, 4, "bad offset at {z}, {i}");
            }

            // Only dispatch 0 is used, because we don't have > MAX_DISPATH_SIZE
            // tiles generated in this stage.
            let dispatch =
                ctx.read_buffer::<Dispatch>(&buffers.tile16.dispatch);
            assert_eq!(dispatch.len(), buffers.tile16.max_dispatch_count());
            assert_eq!(dispatch[0].tile_dispatch, [16, 1, 1]);
            assert_eq!(dispatch[0].backfill_dispatch, [4, 1, 1]);
            for d in &dispatch[1..] {
                assert_eq!(d.tile_dispatch, [0, 0, 0]);
                assert_eq!(d.backfill_dispatch, [0, 0, 0]);
            }

            let sorted = ctx.read_buffer::<u8>(&buffers.tile16.sorted);
            let sorted = TileList::try_ref_from_bytes(&sorted).unwrap();
            for (i, t) in sorted.tiles[..16].chunks(4).enumerate() {
                let mut xs = (0..4).collect::<BTreeSet<u32>>();
                for t in t {
                    assert_eq!(t.tape_index, 0);
                    let x = t.tile % 4;
                    let y = (t.tile / 4) % 4;
                    let z = (t.tile / 4) / 4;

                    // Ensure that we're in reverse-z order
                    assert_eq!(z as usize, 4 - i - 1);
                    // Only middle tiles should be active
                    assert_eq!(y, 1);
                    // All Y values must be represented
                    assert!(xs.remove(&x));
                }
                assert!(xs.is_empty());
            }

            // Repacking uses `hist` to compute a cumulative sum
            let hist = ctx.read_buffer::<u32>(&buffers.tile16.hist);
            assert_eq!(hist.len(), 4);
            for (i, n) in hist.iter().enumerate() {
                assert_eq!(*n, (4 * (i + 1) as u32) | (1 << 31));
            }

            // Now we're onto the 16³ -> 4³ tile pass
            //
            // We'll do a single dispatch of 16³ -> 4³ tile refinement, which
            // should produce 256 tiles with
            //  x = 0..16
            //  y = 7
            //  z = 0..16

            assert_eq!(buffers.tile16.max_dispatch_count(), 1);
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.interval4_ctx.run(
                ctx,
                &shape,
                &buffers.tile16,
                &buffers.tile4,
                0,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let dispatch_counter =
                ctx.read_buffer::<u32>(&buffers.tile16.dispatch_counter);
            assert_eq!(
                dispatch_counter.len(),
                buffers.tile16.max_tiles_per_dispatch()
            );
            // all dispatch counters are either left as 0 or reset
            for (i, d) in dispatch_counter.iter().enumerate() {
                assert_eq!(*d, 0, "bad dispatch at {i}");
            }

            let out = ctx.read_buffer::<u8>(&buffers.tile4.out);
            let out = TileList::try_ref_from_bytes(&out).unwrap();
            assert_eq!(
                out.tiles.len(),
                buffers.tile16.max_tiles_per_dispatch() * 64
            );
            assert_eq!(out.count, 256); // X-Z slice at the origin
            let mut tiles = BTreeSet::new();
            for t in out.tiles.iter().take(out.count as usize) {
                assert_eq!(t.tape_index, 0);
                let x = t.tile % 16;
                let y = (t.tile / 16) % 16;
                let z = (t.tile / 16) / 16;
                assert_eq!(y, 7, "bad value in tile {x}, {y}, {z}");
                assert!(tiles.insert(t.tile));
            }
            assert_eq!(tiles.len(), 256); // no duplicates

            // Check that the zmin buffer is correctly filled out
            let zmin = ctx.read_buffer::<Voxel>(&buffers.tile4.zmin);
            assert_eq!(zmin.len(), 16 * 16);
            for y in 0..16 {
                for x in 0..16 {
                    let t = zmin[x + y * 4];
                    // The previous stage only produced tiles with y = 1
                    if y / 4 == 1 {
                        assert_eq!(t.tape_index(), 0);
                        let expected = if y > 7 { 63 } else { 0 };
                        assert_eq!(t.z(), expected, "bad z at tile {x}, {y}");
                    }
                }
            }

            let hist = ctx.read_buffer::<u32>(&buffers.tile4.hist);
            assert_eq!(hist.len(), 16);
            for n in hist {
                assert_eq!(n, 16); // strip of 1x16 tiles
            }

            ////////////////////////////////////////////////////////////////////
            // Almost done!  Time for the repacking pass for 4³ tiles
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.repack4_ctx.run(
                ctx,
                &buffers.tile4,
                Some((0, &buffers.tile16)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let tile_z_to_offset =
                ctx.read_buffer::<u32>(&buffers.tile4.z_to_offset);
            assert_eq!(tile_z_to_offset.len(), 16);
            for (i, z) in tile_z_to_offset.iter().enumerate() {
                assert_eq!(*z as usize, 16, "bad offset at {z}, {i}");
            }

            let dispatch = ctx.read_buffer::<Dispatch>(&buffers.tile4.dispatch);
            assert_eq!(dispatch.len(), buffers.tile4.max_dispatch_count());
            assert_eq!(dispatch[0].tile_dispatch, [256, 1, 1]);
            assert_eq!(
                dispatch[0].backfill_dispatch,
                [16 * 16 * 16 / 64, 1, 1]
            );
            for d in &dispatch[1..] {
                assert_eq!(d.tile_dispatch, [0, 0, 0]);
                assert_eq!(d.backfill_dispatch, [0, 0, 0]);
            }

            let sorted = ctx.read_buffer::<u8>(&buffers.tile4.sorted);
            let sorted = TileList::try_ref_from_bytes(&sorted).unwrap();
            assert_eq!(sorted.count, 256);
            for (i, t) in
                sorted.tiles[..sorted.count as usize].chunks(16).enumerate()
            {
                let mut xs = (0..16).collect::<BTreeSet<u32>>();
                for t in t {
                    assert_eq!(t.tape_index, 0);
                    let x = t.tile % 16;
                    let y = (t.tile / 16) % 16;
                    let z = (t.tile / 16) / 16;

                    // Ensure that we're in reverse-z order
                    assert_eq!(z as usize, 16 - i - 1);
                    // Only middle tiles should be active
                    assert_eq!(y, 7);
                    // All Y values must be represented
                    assert!(xs.remove(&x));
                }
                assert!(xs.is_empty());
            }

            // Repacking uses `hist` to compute a cumulative sum
            let hist = ctx.read_buffer::<u32>(&buffers.tile4.hist);
            assert_eq!(hist.len(), 16);
            for (i, n) in hist.iter().enumerate() {
                assert_eq!(*n, (16 * (i + 1) as u32) | (1 << 31));
            }

            ////////////////////////////////////////////////////////////////////
            // It's morging^W voxel time!
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.voxel_ctx.run(ctx, &buffers, &shape, &mut compute_pass);
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let voxels = ctx.read_buffer::<Voxel>(&buffers.voxels);
            assert_eq!(voxels.len(), 64 * 64);
            for (y, vs) in voxels.chunks(64).enumerate() {
                for (x, v) in vs.iter().enumerate() {
                    assert_eq!(v.tape_index(), 0);
                    assert_eq!(v.z(), 0, "bad value at {x}, {y}");
                }
            }

            // Do all of the backfill passes
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.backfill4_ctx.run(
                ctx,
                buffers.render_size(),
                &buffers.voxels,
                &buffers.tile4,
                Some((0, &buffers.tile16)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.backfill16_ctx.run(
                ctx,
                buffers.render_size(),
                &buffers.tile4.zmin,
                &buffers.tile16,
                Some((0, &buffers.tile64)),
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.backfill64_ctx.run::<0>(
                ctx,
                buffers.render_size(),
                &buffers.tile16.zmin,
                &buffers.tile64,
                None,
                &mut compute_pass,
            );
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            ////////////////////////////////////////////////////////////////////
            // Normals!
            let mut encoder = ctx.new_encoder();
            let mut compute_pass = ctx.begin_compute_pass(&mut encoder);
            ctx.normals_ctx
                .run(ctx, &buffers, &shape, &mut compute_pass);
            drop(compute_pass);
            ctx.queue.submit(Some(encoder.finish()));

            let geom = ctx.read_buffer::<[f32; 4]>(&buffers.geom);
            assert_eq!(geom.len(), 64 * 64);
            for (y, vs) in geom.chunks(64).enumerate() {
                for (x, v) in vs.iter().enumerate() {
                    let expected = if y > 31 {
                        [63.0, 0.0, -1.0 / 32.0, 0.0]
                    } else {
                        [0.0; 4]
                    };
                    assert_eq!(v, &expected, "bad pixel at {x}, {y}");
                }
            }
        }

        #[test]
        fn step_by_step_y() {
            CTX.with(step_by_step_y_);
        }

        fn repeated_xy_(ctx: &Context) {
            let size = VoxelSize::from(64);
            let buffers = ctx.buffers(size, GpuSpec::High);
            let shape_x = ctx.shape(&VmShape::from(Tree::x()));
            let shape_y = ctx.shape(&VmShape::from(Tree::y()));
            let settings = RenderConfig::default();

            for j in 0..10 {
                let shape = if j % 2 == 0 { &shape_x } else { &shape_y };
                let out = ctx.run(shape, &buffers, settings);
                println!();
                for y in 0..64 {
                    for x in 0..64 {
                        let p = &out[(y, x)];
                        print!("{}", if p.depth == 0.0 { ".." } else { "##" });
                    }
                    println!();
                }
                for y in 0..64 {
                    for x in 0..64 {
                        let p = &out[(y, x)];
                        let g = if j % 2 == 0 {
                            if x <= 31 {
                                GeometryPixel {
                                    depth: 63.0,
                                    normal: [1.0 / 32.0, 0.0, 0.0],
                                }
                            } else {
                                GeometryPixel {
                                    depth: 0.0,
                                    normal: [0.0, 0.0, 0.0],
                                }
                            }
                        } else if y > 31 {
                            GeometryPixel {
                                depth: 63.0,
                                normal: [0.0, -1.0 / 32.0, 0.0],
                            }
                        } else {
                            GeometryPixel {
                                depth: 0.0,
                                normal: [0.0, 0.0, 0.0],
                            }
                        };
                        assert_eq!(*p, g, "bad pixel at {x}, {y} [{j}]");
                    }
                }
            }
        }

        #[test]
        fn repeated_xy() {
            CTX.with(repeated_xy_);
        }

        fn repeated_x_(ctx: &Context) {
            let size = VoxelSize::from(64);
            let buffers = ctx.buffers(size, GpuSpec::High);
            let shape = ctx.shape(&VmShape::from(Tree::x()));
            let settings = RenderConfig::default();

            for j in 0..5 {
                let out = ctx.run(&shape, &buffers, settings);
                println!();
                for y in 0..64 {
                    for x in 0..64 {
                        let p = &out[(y, x)];
                        print!("{}", if p.depth == 0.0 { ".." } else { "##" });
                    }
                    println!();
                }
                for y in 0..64 {
                    for x in 0..64 {
                        let p = &out[(y, x)];
                        let g = if x <= 31 {
                            GeometryPixel {
                                depth: 63.0,
                                normal: [1.0 / 32.0, 0.0, 0.0],
                            }
                        } else {
                            GeometryPixel {
                                depth: 0.0,
                                normal: [0.0, 0.0, 0.0],
                            }
                        };
                        assert_eq!(*p, g, "bad pixel at {x}, {y} [{j}]");
                    }
                }
            }
        }

        #[test]
        fn repeated_x() {
            CTX.with(repeated_x_);
        }

        #[test]
        fn repeated_x_with_fresh_context() {
            let size = VoxelSize::from(64);
            let settings = RenderConfig::default();

            for j in 0..5 {
                let ctx = get_context();
                let buffers = ctx.buffers(size, GpuSpec::High);
                let shape = ctx.shape(&VmShape::from(Tree::x()));
                let out = ctx.run(&shape, &buffers, settings);
                println!();
                for y in 0..64 {
                    for x in 0..64 {
                        let p = &out[(y, x)];
                        print!("{}", if p.depth == 0.0 { ".." } else { "##" });
                    }
                    println!();
                }
                for y in 0..64 {
                    for x in 0..64 {
                        let p = &out[(y, x)];
                        let g = if x <= 31 {
                            GeometryPixel {
                                depth: 63.0,
                                normal: [1.0 / 32.0, 0.0, 0.0],
                            }
                        } else {
                            GeometryPixel {
                                depth: 0.0,
                                normal: [0.0, 0.0, 0.0],
                            }
                        };
                        assert_eq!(*p, g, "bad pixel at {x}, {y} [{j}]");
                    }
                }
            }
        }
    }
}
