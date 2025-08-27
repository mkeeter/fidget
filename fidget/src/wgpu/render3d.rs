//! GPU-accelerated 3D rendering
use crate::{
    Error,
    eval::Function,
    render::{GeometryBuffer, GeometryPixel, VoxelSize},
    vm::VmShape,
    wgpu::{
        opcode_constants,
        util::{resize_buffer_with, write_storage_buffer},
    },
};

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

use zerocopy::{FromBytes, Immutable, IntoBytes};

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
            world_to_model: crate::render::View3::default().world_to_model(),
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

/// Doppleganger of the WGSL `struct Config`
///
/// Fields are carefully ordered to require no padding
#[derive(Debug, IntoBytes, Immutable)]
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
}

/// Number of `u32` words in the tape data flexible array
const TAPE_DATA_CAPACITY: usize = 65536;

/// Returns a shader for interval root tiles
pub fn interval_root_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERVAL_ROOT_SHADER;
    shader_code += INTERVAL_OPS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += STACK_SHADER;
    shader_code += TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for interval tile evaluation
pub fn interval_tiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERVAL_TILES_SHADER;
    shader_code += INTERVAL_OPS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += STACK_SHADER;
    shader_code
}

/// Returns a shader for interval tile evaluation
pub fn voxel_tiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += VOXEL_TILES_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += DUMMY_STACK_SHADER;
    shader_code
}

/// Returns a shader for interval tile evaluation
pub fn normals_shader() -> String {
    let mut shader_code = opcode_constants();
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

struct RootContext {
    /// Input tiles (64^3)
    tile64_buffers: TileBuffers<64>,

    /// Pipeline for 64^3 tile evaluation
    root_pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Per-strata offset in the root tiles list
///
/// This must be equivalent to `strata_size_bytes` in the interval root shader
fn strata_size_bytes(render_size: VoxelSize) -> usize {
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

        let shader_code = interval_root_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let root_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interval root"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_root_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            bind_group_layout,
            tile64_buffers,
            root_pipeline,
        }
    }

    fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: VoxelSize,
    ) {
        self.tile64_buffers.reset_root(device, queue, render_size)
    }

    fn run(
        &self,
        ctx: &Context,
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

        compute_pass.set_pipeline(&self.root_pipeline);
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

    /// Pipeline for 64^3 -> 16^3 tile evaluation evaluation
    interval64_pipeline: wgpu::ComputePipeline,

    /// Pipeline for 16^3 -> 4^3 tile evaluation evaluation
    interval16_pipeline: wgpu::ComputePipeline,

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

        let shader_code = interval_tiles_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let interval64_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interval"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 64.0), ("SUBTILE_SIZE", 16.0)],
                    ..Default::default()
                },
                cache: None,
            });
        let interval16_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interval"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 16.0), ("SUBTILE_SIZE", 4.0)],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            bind_group_layout,
            tile16_buffers,
            tile4_buffers,
            interval64_pipeline,
            interval16_pipeline,
        }
    }

    fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: VoxelSize,
    ) {
        self.tile16_buffers.reset(device, queue, render_size);
        self.tile4_buffers.reset(device, queue, render_size);
    }

    fn run(
        &self,
        ctx: &Context,
        strata: usize,
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
        compute_pass.set_pipeline(&self.interval64_pipeline);
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
        compute_pass.set_pipeline(&self.interval16_pipeline);
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
    voxel_pipeline: wgpu::ComputePipeline,
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

        let shader_code = voxel_tiles_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let voxel_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxels"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("voxel_ray_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            bind_group_layout,
            voxel_pipeline,
        }
    }

    fn run(&self, ctx: &Context, compute_pass: &mut wgpu::ComputePass) {
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

        compute_pass.set_pipeline(&self.voxel_pipeline);
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
    normals_pipeline: wgpu::ComputePipeline,
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

        let shader_code = normals_shader();
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    common_bind_group_layout,
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let normals_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("normals"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("normals_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            bind_group_layout,
            normals_pipeline,
        }
    }

    fn run(
        &self,
        ctx: &Context,
        image_size: VoxelSize,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.buffers.merged.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buffers.geom.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(&self.normals_pipeline);
        compute_pass.set_bind_group(1, &bind_group, &[]);

        compute_pass.dispatch_workgroups(
            image_size.width().div_ceil(8),
            image_size.height().div_ceil(8),
            1,
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
}

struct TileBuffers<const N: usize> {
    tiles: wgpu::Buffer,
    zmin: wgpu::Buffer,
    zmin_scratch: Vec<u32>,
}

impl<const N: usize> TileBuffers<N> {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            tiles: scratch_buffer(device),
            zmin: scratch_buffer(device),
            zmin_scratch: vec![],
        }
    }

    /// Reset buffers, allocating room for densely packed tiles
    fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: VoxelSize,
    ) {
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
        self.reset_zmin(device, queue, render_size);
    }

    /// Reset a root tiles buffer, which stores strata-packed tile lists
    fn reset_root(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: VoxelSize,
    ) {
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
        self.reset_zmin(device, queue, render_size);
    }

    fn reset_zmin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: VoxelSize,
    ) {
        // zero out the zmin buffer
        let nx = render_size.width() as usize / N;
        let ny = render_size.height() as usize / N;
        if self.zmin_scratch.len() != nx * ny {
            self.zmin_scratch.resize(nx * ny, 0u32);
        }
        write_storage_buffer(
            device,
            queue,
            &mut self.zmin,
            &format!("tile{N}_zmin"),
            &self.zmin_scratch,
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
    result_scratch: Vec<u32>,

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
            result_scratch: vec![],
        }
    }

    fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image_size: VoxelSize,
    ) {
        let nx = image_size.width().next_multiple_of(64) as usize;
        let ny = image_size.height().next_multiple_of(64) as usize;
        let render_pixels = nx * ny;
        resize_buffer_with::<u32>(
            device,
            &mut self.result,
            "result",
            render_pixels,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        if render_pixels != self.result_scratch.len() {
            self.result_scratch.resize(render_pixels, 0u32);
        }
        queue
            .write_buffer_with(
                &self.result,
                0,
                (render_pixels as u64 * 4).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice(self.result_scratch.as_bytes());

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
    pub fn new() -> Result<Self, Error> {
        // Initialize wgpu
        let instance = wgpu::Instance::default();
        let (device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..wgpu::RequestAdapterOptions::default()
                })
                .await
                .map_err(|_| Error::NoAdapter)?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .map_err(Error::NoDevice)
        })?;

        // The config buffer is statically sized
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (std::mem::size_of::<Config>()
                + TAPE_DATA_CAPACITY * std::mem::size_of::<u32>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let bc = shape.to_bytecode();
        assert!(bc.data.len() < TAPE_DATA_CAPACITY);

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
            tape_data_offset: bc.data.len().try_into().unwrap(),
        };

        {
            // We load the `Config` and the initial tape; the rest of the tape
            // buffer is uninitialized (filled in by the GPU)
            let config_len = std::mem::size_of_val(&config);
            let bc_len = bc.data.as_bytes().len();
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
            writer[config_len..][..bc_len].copy_from_slice(bc.data.as_bytes());
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

        self.buffers.reset(&self.device, &self.queue, render_size);
        self.root_ctx.reset(&self.device, &self.queue, render_size);
        self.interval_ctx
            .reset(&self.device, &self.queue, render_size);

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

        // Populate root tiles
        self.root_ctx.run(self, render_size, &mut compute_pass);

        // Evaluate tiles in reverse-Z order by strata (64 voxels deep)
        for strata in (0..render_size.depth() as usize / 64).rev() {
            self.interval_ctx
                .run(self, strata, render_size, &mut compute_pass);
            self.voxel_ctx.run(self, &mut compute_pass);
            self.backfill_ctx.run(
                self,
                strata,
                settings.image_size,
                &mut compute_pass,
            );
        }
        self.merge_ctx
            .run(self, settings.image_size, &mut compute_pass);
        self.normals_ctx
            .run(self, settings.image_size, &mut compute_pass);
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
        let buffer_slice = self.buffers.image.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Get the pixel-populated image
        let result =
            <[GeometryPixel]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        self.buffers.image.unmap();

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
        self.device.poll(wgpu::PollType::Wait).unwrap();

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
        let ny = render_size.width().div_ceil(64) as usize;

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
                .slice(offset_bytes..)
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
            image_size.width().div_ceil(8),
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
        for (op, _) in crate::bytecode::iter_ops() {
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
            (interval_root_shader(), "interval root"),
            (interval_tiles_shader(), "interval tiles"),
            (voxel_tiles_shader(), "voxel tiles"),
            (normals_shader(), "normals tiles"),
            (merge_shader(), "merge"),
            (backfill_shader(), "backfill"),
        ] {
            // This isn't the best formatting, but it will at least include the
            // relevant text.
            let m = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                if let Some(i) = e.location(&src) {
                    let pos = i.offset as usize..(i.offset + i.length) as usize;
                    panic!(
                        "shader conpilation failed\n{src}\n{}",
                        e.emit_to_string_with_path(&src[pos], desc)
                    );
                } else {
                    panic!(
                        "shader conpilation failed\n{src}\n{}",
                        e.emit_to_string(desc)
                    );
                }
            });
            if let Err(e) = v.validate(&m) {
                let (pos, desc) = e.spans().next().unwrap();
                panic!(
                    "shader conpilation failed\n{src}\n{}",
                    e.emit_to_string_with_path(
                        &src[pos.to_range().unwrap()],
                        desc
                    )
                );
            }
        }
    }
}
