//! GPU-accelerated 3D rendering
use crate::{Error, opcode_constants, util::resize_buffer_with};
use fidget_bytecode::{Bytecode, BytecodeOp};
use fidget_core::{eval::Function, render::VoxelSize, vm::VmShape};
use fidget_raster::{GeometryBuffer, GeometryPixel};
use std::collections::BTreeMap;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const DEBUG_MODE: bool = false;

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const VOXEL_TILES_SHADER: &str = include_str!("shaders/voxel_tiles.wgsl");
const STACK_SHADER: &str = include_str!("shaders/stack.wgsl");
const DUMMY_STACK_SHADER: &str = include_str!("shaders/dummy_stack.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_OPS_SHADER: &str = include_str!("shaders/interval_ops.wgsl");
const BACKFILL_SHADER: &str = include_str!("shaders/backfill.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const CLEAR_SHADER: &str = include_str!("shaders/clear.wgsl");
const NORMALS_SHADER: &str = include_str!("shaders/normals.wgsl");
const TAPE_INTERPRETER: &str = include_str!("shaders/tape_interpreter.wgsl");
const TAPE_SIMPLIFY: &str = include_str!("shaders/tape_simplify.wgsl");

/// Settings for 3D rendering
#[derive(Copy, Clone)]
pub struct RenderConfig {
    /// Render size
    ///
    /// The resulting image will have the given width and height; depth sets the
    /// number of voxels to evaluate within each pixel of the image (stacked
    /// into a column going into the screen).
    pub image_size: VoxelSize,

    /// World-to-model transform
    pub world_to_model: nalgebra::Matrix4<f32>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            image_size: VoxelSize::from(512),
            world_to_model: nalgebra::Matrix4::identity(),
        }
    }
}

impl RenderConfig {
    /// Returns the combined screen-to-model transform matrix
    pub fn mat(&self) -> nalgebra::Matrix4<f32> {
        self.world_to_model * self.image_size.screen_to_world()
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

    /// Next empty position in `tile_tapes`
    tile_tapes_offset: u32,

    /// Image size (not rounded)
    image_size: [u32; 3],

    /// Number of words in the trailing tape buffer
    tape_data_capacity: u32,

    /// Length of the root tape
    root_tape_len: u32,
}

/// Number of `u32` words in the tape data flexible array
const TAPE_DATA_CAPACITY: usize = 1024 * 1024 * 16; // 16M words, 64 MiB

/// Returns a shader for interval root tiles
pub fn interval_root_shader(reg_count: u8) -> String {
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
pub fn interval_tiles_shader(reg_count: u8) -> String {
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
pub fn voxel_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += VOXEL_TILES_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += DUMMY_STACK_SHADER;
    shader_code
}

/// Returns a shader for interval tile evaluation
pub fn normals_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += NORMALS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += DUMMY_STACK_SHADER;
    shader_code
}

/// Returns a shader for merging images
pub fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + COMMON_SHADER
}

/// Returns a shader for clearing images
pub fn clear_shader() -> String {
    CLEAR_SHADER.to_owned() + COMMON_SHADER
}

/// Returns a shader for backfilling tile `zmin` values
pub fn backfill_shader() -> String {
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
    /// Input tiles (64^3)
    tile64_buffers: TileBuffers<64>,

    /// Pipelines for 64^3 tile evaluation
    root_pipeline: RegPipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Per-strata offset in the root tiles list
///
/// This must be equivalent to `strata_size_bytes` in the interval root shader
fn strata_size_bytes(render_size: VoxelSize) -> usize {
    assert!(render_size.width().is_multiple_of(64));
    assert!(render_size.height().is_multiple_of(64));
    let nx = (render_size.width() / 64) as usize;
    let ny = (render_size.height() / 64) as usize;
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
                    buffer_rw(0), // tile_zmin
                    buffer_rw(1), // tiles_out
                ],
            });

        let tile64_buffers = TileBuffers::new(device);

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
            tile64_buffers,
            root_pipeline,
        }
    }

    fn reset(&mut self, device: &wgpu::Device, render_size: VoxelSize) {
        self.tile64_buffers.reset_root(device, render_size)
    }

    fn run(
        &self,
        ctx: &Context,
        reg_count: u8,
        render_size: VoxelSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.tile64_buffers.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.tile64_buffers.tiles.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.root_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);
        let nx = (render_size.width() / 64).div_ceil(4);
        let ny = (render_size.height() / 64).div_ceil(4);
        let nz = (render_size.depth() / 64).div_ceil(4);
        compute_pass.dispatch_workgroups(nx, ny, nz);
    }
}

////////////////////////////////////////////////////////////////////////////////

struct IntervalContext {
    /// First stage output tiles (16^3)
    tile16_buffers: TileBuffers<16>,

    /// First stage output tiles (4^3)
    tile4_buffers: TileBuffers<4>,

    /// Pipeline for 64^3 -> 16^3 tile evaluation
    interval64_pipeline: RegPipeline,

    /// Pipeline for 16^3 -> 4^3 tile evaluation
    interval16_pipeline: RegPipeline,

    /// Bind group layout
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
                    buffer_rw(0), // tiles_in (rw so we can update wg size)
                    buffer_ro(1), // tile_zmin
                    buffer_rw(2), // subtiles_out
                    buffer_rw(3), // subtile_zmin
                ],
            });

        let tile16_buffers = TileBuffers::new(device);
        let tile4_buffers = TileBuffers::new(device);

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
            tile16_buffers,
            tile4_buffers,
            interval64_pipeline,
            interval16_pipeline,
        }
    }

    fn reset(&mut self, device: &wgpu::Device, render_size: VoxelSize) {
        self.tile16_buffers.reset(device, render_size);
        self.tile4_buffers.reset(device, render_size);
    }

    fn run(
        &self,
        ctx: &Context,
        strata: usize,
        reg_count: u8,
        render_size: VoxelSize,
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
                        resource: ctx
                            .root_ctx
                            .tile64_buffers
                            .tiles
                            .slice(offset_bytes..)
                            .into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx
                            .root_ctx
                            .tile64_buffers
                            .zmin
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.tile16_buffers.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.tile16_buffers.zmin.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_pipeline(self.interval64_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group16, &[]);
        compute_pass.dispatch_workgroups_indirect(
            &ctx.root_ctx.tile64_buffers.tiles,
            offset_bytes as u64,
        );

        let bind_group4 =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.tile16_buffers.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.tile16_buffers.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.tile4_buffers.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.tile4_buffers.zmin.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_pipeline(self.interval16_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group4, &[]);
        compute_pass
            .dispatch_workgroups_indirect(&self.tile16_buffers.tiles, 0);
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
                        resource: ctx
                            .interval_ctx
                            .tile4_buffers
                            .tiles
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx
                            .interval_ctx
                            .tile4_buffers
                            .zmin
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.buffers.result.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.voxel_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // Each workgroup is 4x4x4, i.e. covering a 4x4 splat of pixels with 4x
        // workers in the Z direction.
        compute_pass.dispatch_workgroups_indirect(
            &ctx.interval_ctx.tile4_buffers.tiles,
            0,
        );
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
        strata: usize,
        reg_count: u8,
        render_size: VoxelSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let offset_bytes = (strata * strata_size_bytes(render_size)) as u64;

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx
                            .root_ctx
                            .tile64_buffers
                            .tiles
                            .slice(offset_bytes..)
                            .into(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buffers.merged.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.buffers.geom.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.normals_pipeline.get(reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        compute_pass.dispatch_workgroups_indirect(
            &ctx.root_ctx.tile64_buffers.tiles,
            offset_bytes as u64,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 3D (voxel) rendering
pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: DynamicBuffers,

    config_buf: wgpu::Buffer,
    tile_tape_buf: wgpu::Buffer,
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
    fn new(device: &wgpu::Device) -> Self {
        Self {
            tiles: scratch_buffer(device),
            zmin: scratch_buffer(device),
        }
    }

    /// Reset buffers, allocating room for densely packed tiles
    fn reset(&mut self, device: &wgpu::Device, render_size: VoxelSize) {
        let nx = render_size.width() as usize / N;
        let ny = render_size.height() as usize / N;
        let nz = 64 / N;

        resize_buffer_with::<u32>(
            device,
            &mut self.tiles,
            &format!("active_tile{N}"),
            // wg_dispatch: [u32; 3]
            // count: u32,
            4 + nx * ny * nz,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        self.reset_zmin(device, render_size);
    }

    /// Reset a root tiles buffer, which stores strata-packed tile lists
    fn reset_root(&mut self, device: &wgpu::Device, render_size: VoxelSize) {
        assert_eq!(N, 64);
        let nz = render_size.depth() as usize / N;

        let strata_size = strata_size_bytes(render_size);
        let total_size = strata_size * nz;

        resize_buffer_with::<u8>(
            // bytes
            device,
            &mut self.tiles,
            &format!("active_tile{N}"),
            // wg_dispatch: [u32; 3]
            // count: u32,
            total_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        self.reset_zmin(device, render_size);
    }

    fn reset_zmin(&mut self, device: &wgpu::Device, render_size: VoxelSize) {
        // zero out the zmin buffer
        let nx = render_size.width() as usize / N;
        let ny = render_size.height() as usize / N;
        resize_buffer_with::<u32>(
            device,
            &mut self.zmin,
            &format!("tile{N}_zmin"),
            nx * ny,
            wgpu::BufferUsages::STORAGE,
        );
    }
}

/// Buffers which must be dynamically sized
struct DynamicBuffers {
    /// Result buffer written by the voxel compute shader
    ///
    /// This buffer has sizes that are multiples of 64 voxels on each dimension
    ///
    /// (dynamic size, implicit from image size in config)
    result: wgpu::Buffer,

    /// Combined result buffer, at the target image size
    ///
    /// Note that this buffer can't be read by the host; it must be copied to
    /// 'image_buf`
    merged: wgpu::Buffer,

    /// Buffer of `GeometryPixel` equivalent data
    geom: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    ///
    /// (dynamic size, implicit from image size in config)
    image: wgpu::Buffer,
}

fn scratch_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

impl DynamicBuffers {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            result: scratch_buffer(device),
            image: scratch_buffer(device),
            merged: scratch_buffer(device),
            geom: scratch_buffer(device),
        }
    }

    fn reset(&mut self, device: &wgpu::Device, image_size: VoxelSize) {
        let nx = image_size.width().next_multiple_of(64) as usize;
        let ny = image_size.height().next_multiple_of(64) as usize;
        let render_pixels = nx * ny;
        resize_buffer_with::<u32>(
            device,
            &mut self.result,
            "result",
            render_pixels,
            wgpu::BufferUsages::STORAGE,
        );

        let image_pixels =
            image_size.width() as usize * image_size.height() as usize;
        resize_buffer_with::<u32>(
            device,
            &mut self.merged,
            "merged",
            image_pixels,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        resize_buffer_with::<GeometryPixel>(
            device,
            &mut self.geom,
            "geom",
            image_pixels,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        resize_buffer_with::<GeometryPixel>(
            device,
            &mut self.image,
            "image",
            image_pixels,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
    }
}

impl Context {
    /// Build a new 3D rendering context
    ///
    /// This is async due to choices made by WebGPU; to use in an otherwise-sync
    /// program, consider a minimal async runtime like `pollster`.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<Self, Error> {
        // The config buffer is statically sized
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<u32>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | if DEBUG_MODE {
                    wgpu::BufferUsages::COPY_SRC
                } else {
                    wgpu::BufferUsages::empty()
                },
            mapped_at_creation: false,
        });
        log::info!("got config buf");
        let tile_tape_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile tape (dummy)"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

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
        let clear_ctx = ClearContext::new(&device, &common_bind_group_layout);
        let buffers = DynamicBuffers::new(&device);

        Ok(Self {
            device,
            config_buf,
            tile_tape_buf,
            buffers,
            queue,
            bind_group_layout: common_bind_group_layout,
            root_ctx,
            interval_ctx,
            voxel_ctx,
            normals_ctx,
            backfill_ctx,
            merge_ctx,
            clear_ctx,
        })
    }

    /// Renders a single image using GPU acceleration
    ///
    /// Returns a heightmap
    pub fn run(
        &mut self,
        shape: &VmShape, // XXX add ShapeVars here
        settings: RenderConfig,
    ) -> GeometryBuffer {
        // Generate bytecode for the root tape
        let vars = shape.inner().vars();
        let bc = Bytecode::new(shape.inner().data());
        assert!(bc.len() < TAPE_DATA_CAPACITY);

        // Create the 4x4 transform matrix
        let shape = shape.with_transform(settings.mat());
        let mat = shape.transform();
        let axes = shape
            .axes()
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX));
        let render_size = VoxelSize::new(
            settings.image_size.width().next_multiple_of(64),
            settings.image_size.height().next_multiple_of(64),
            settings.image_size.depth().next_multiple_of(64),
        );

        let nx = render_size.width() as u64 / 64;
        let ny = render_size.height() as u64 / 64;
        let nz = render_size.depth() as u64 / 64;

        let bytecode_len: u32 = bc.len().try_into().unwrap();
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes,
            tile_tapes_offset: (nx * ny * nz).try_into().unwrap(),
            render_size: [
                render_size.width(),
                render_size.height(),
                render_size.depth(),
            ],
            tape_data_capacity: TAPE_DATA_CAPACITY.try_into().unwrap(),
            image_size: [
                settings.image_size.width(),
                settings.image_size.height(),
                settings.image_size.depth(),
            ],
            tape_data_offset: bytecode_len,
            root_tape_len: bytecode_len,
        };

        {
            // We load the `Config` and the initial tape; the rest of the tape
            // buffer is uninitialized (filled in by the GPU)
            let config_len = std::mem::size_of_val(&config);
            let bc_len = bc.as_bytes().len();
            let config_load_size = (config_len + bc_len).next_multiple_of(16);
            let mut writer = self
                .queue
                .write_buffer_with(
                    &self.config_buf,
                    0,
                    (config_load_size as u64).try_into().unwrap(),
                )
                .unwrap();
            writer[0..config_len].copy_from_slice(config.as_bytes());
            writer[config_len..][..bc_len].copy_from_slice(bc.as_bytes());
        }

        // Resize tile_tape_buf, which must be big enough to contain all of our
        // root tapes, followed by a full strata's worth of subsequent tapes (in
        // a hierarchy).
        let tile_tape_words = nx * ny * nz  // Root tapes
            + nx * ny * 4u64.pow(3)   // Tapes for one strata of 16^3 tiles
            + nx * ny * 16u64.pow(3); // Tapes for one strata of 4^3 tiles
        let tile_tape_bytes =
            tile_tape_words * std::mem::size_of::<u32>() as u64;
        if self.tile_tape_buf.size() != tile_tape_bytes {
            self.tile_tape_buf =
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tile tape"),
                    size: tile_tape_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
        }

        self.buffers.reset(&self.device, settings.image_size);
        self.root_ctx.reset(&self.device, render_size);
        self.interval_ctx.reset(&self.device, render_size);

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

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
                        resource: self.config_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.tile_tape_buf.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let reg_count = bc.reg_count();

        // Initial image clearing pass
        self.clear_ctx
            .run(self, settings.image_size, &mut compute_pass);

        // Populate root tiles
        self.root_ctx
            .run(self, reg_count, render_size, &mut compute_pass);

        // Evaluate tiles in reverse-Z order by strata (64 voxels deep)
        for strata in (0..render_size.depth() as usize / 64).rev() {
            self.interval_ctx.run(
                self,
                strata,
                reg_count,
                render_size,
                &mut compute_pass,
            );
            self.voxel_ctx.run(self, reg_count, &mut compute_pass);

            // It's somewhat overkill to run `merge` after each layer, but it's
            // also very cheap.
            self.merge_ctx
                .run(self, settings.image_size, &mut compute_pass);
            self.normals_ctx.run(
                self,
                strata,
                reg_count,
                render_size,
                &mut compute_pass,
            );

            // Backfill from smaller tiles to larger tiles, for culling. This
            // step also resets counters for subsequent strata.  We skip this
            // step if we're in DEBUG_MODE and rendering a single strata,
            // because we want those buffers to remain un-reset.
            if DEBUG_MODE && render_size.depth() == 64 {
                continue;
            }
            self.backfill_ctx
                .run(self, strata, render_size, &mut compute_pass);
        }
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &self.buffers.geom,
            0,
            &self.buffers.image,
            0,
            self.buffers.geom.size(),
        );

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        log::info!("buffer slice: {}", self.buffers.image.size());
        let buffer_slice = self.buffers.image.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // Get the pixel-populated image
        let result =
            <[GeometryPixel]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        self.buffers.image.unmap();

        if DEBUG_MODE {
            let cfg = self.read_buffer::<u8>(&self.config_buf);
            let (cfg, rest) = Config::ref_from_prefix(&cfg).unwrap();
            let rest = <[u32]>::ref_from_bytes(rest).unwrap();
            println!("{cfg:?}");
            for i in (0..cfg.tape_data_offset).step_by(2) {
                if i == cfg.root_tape_len {
                    println!();
                }
                let i = i as usize;
                let f = f32::from_bits(rest[i + 1]);
                let op = rest[i].as_bytes();
                let oooop = BytecodeOp::from_repr(op[0] as usize);
                println!(
                    "{i:>4} | {:02x?} {:>10}: {}",
                    rest[i].as_bytes(),
                    if op[0] == 0xFF {
                        "JUMP"
                    } else {
                        oooop.map(|o| o.into()).unwrap_or("")
                    },
                    if f > -100.0 && f < 100.0 && f.abs() > 0.001 {
                        format!(" {f}")
                    } else {
                        format!(" {}", rest[i + 1])
                    }
                );
                if op[0] == 0xFF && rest[i + 1] == u32::MAX {
                    println!();
                }
            }
        }
        /*
        let tiles =
            self.read_buffer::<u32>(&self.root_ctx.tile64_buffers.tiles);
        println!("{tiles:x?}");
        */

        GeometryBuffer::build(result, settings.image_size).unwrap()
    }

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
        strata: usize,
        render_size: VoxelSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let offset_bytes = (strata * strata_size_bytes(render_size)) as u64;

        let tile4_zmin = &ctx.interval_ctx.tile4_buffers.zmin;
        let tile16_zmin = &ctx.interval_ctx.tile16_buffers.zmin;
        let tile64_zmin = &ctx.root_ctx.tile64_buffers.zmin;

        let nx = render_size.width().div_ceil(64) as usize;
        let ny = render_size.height().div_ceil(64) as usize;

        let bind_group4 = self.create_bind_group(
            ctx,
            &ctx.buffers.result,
            tile4_zmin,
            ctx.interval_ctx.tile4_buffers.tiles.as_entire_binding(),
        );
        compute_pass.set_pipeline(&self.pipeline4);
        compute_pass.set_bind_group(1, &bind_group4, &[]);
        compute_pass.dispatch_workgroups(
            (nx * ny * 16).div_ceil(64) as u32,
            1,
            1,
        );

        let bind_group16 = self.create_bind_group(
            ctx,
            tile4_zmin,
            tile16_zmin,
            ctx.interval_ctx.tile16_buffers.tiles.as_entire_binding(),
        );
        compute_pass.set_pipeline(&self.pipeline16);
        compute_pass.set_bind_group(1, &bind_group16, &[]);
        compute_pass.dispatch_workgroups(
            (nx * ny * 4).div_ceil(64) as u32,
            1,
            1,
        );

        let bind_group64 = self.create_bind_group(
            ctx,
            tile16_zmin,
            tile64_zmin,
            ctx.root_ctx
                .tile64_buffers
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
        image_size: VoxelSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let tile4_zmin = &ctx.interval_ctx.tile4_buffers.zmin;
        let tile16_zmin = &ctx.interval_ctx.tile16_buffers.zmin;
        let tile64_zmin = &ctx.root_ctx.tile64_buffers.zmin;

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tile64_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile16_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile4_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.buffers.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.buffers.merged.as_entire_binding(),
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

struct ClearContext {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ClearContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader_code = clear_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tile64_zmin
                    buffer_rw(1), // tile16_zmin
                    buffer_rw(2), // tile4_zmin
                    buffer_rw(3), // voxel_result
                    buffer_rw(4), // voxel_merged
                    buffer_rw(5), // geom
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
        image_size: VoxelSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let result = &ctx.buffers.result;
        let merged = &ctx.buffers.merged;
        let geom = &ctx.buffers.geom;
        let tile64_zmin = &ctx.root_ctx.tile64_buffers.zmin;
        let tile16_zmin = &ctx.interval_ctx.tile16_buffers.zmin;
        let tile4_zmin = &ctx.interval_ctx.tile4_buffers.zmin;

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tile64_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile16_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile4_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: merged.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: geom.as_entire_binding(),
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
