//! GPU-accelerated 3D rendering
//!
//! # Theory
//! This module implements a column-per-workgroup algorithm for hierarchical
//! implicit surface rendering. Root tiles (64³) are evaluated densely, then
//! a column shader processes each 16×16 pixel sub-column through all Z layers
//! front-to-back, doing interval evaluation at 4³, voxel evaluation, and
//! inline gradient/normal computation in a single dispatch.
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
//! - [`RenderConfig`] sets the transform matrix for rendering.  This is cheap
//!   to construct and could be built once per frame

use crate::{opcode_constants, util::new_buffer};
use fidget_bytecode::Bytecode;
use fidget_core::{eval::Function, render::VoxelSize, vm::VmShape};
use fidget_raster::{GeometryBuffer, GeometryPixel};
use std::collections::BTreeMap;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const STACK_SHADER: &str = include_str!("shaders/stack.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_OPS_SHADER: &str = include_str!("shaders/interval_ops.wgsl");
const TAPE_INTERPRETER: &str = include_str!("shaders/tape_interpreter.wgsl");
const TAPE_SIMPLIFY: &str = include_str!("shaders/tape_simplify.wgsl");
const INTERVAL_TILE16_SHADER: &str =
    include_str!("shaders/interval_tile16.wgsl");
const COLUMN_SHADER: &str = include_str!("shaders/column.wgsl");

const DEBUG_FLAGS: wgpu::BufferUsages = if cfg!(test) {
    wgpu::BufferUsages::COPY_SRC
} else {
    wgpu::BufferUsages::empty()
};

/// Number of workgroups to dispatch for the column shader
const COLUMN_WORKGROUP_COUNT: u32 = 2048;

/// Per-thread tape budget in TapeWords for local simplification
const TAPE_BUDGET: u32 = 64;

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
#[derive(Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
#[repr(C)]
struct Config {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    axes: [u32; 3],
    _padding1: u32,

    /// Render size, rounded up to the nearest multiple of 64
    render_size: [u32; 3],
    _padding2: u32,

    /// Image size (not rounded)
    image_size: [u32; 3],
    max_tiles_per_dispatch: u32,
}

/// Z value and tape index packed into a single `u32` word
#[repr(C)]
#[derive(Copy, Clone, Debug, IntoBytes, Immutable, FromBytes, KnownLayout)]
struct Voxel(u32);

/// A render size is rounded up to the next multiple of 64 on every axis
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
    fn nx(&self) -> u32 {
        self.0.width()
    }
    fn ny(&self) -> u32 {
        self.0.height()
    }
    fn nz(&self) -> u32 {
        self.0.depth()
    }
    fn width(&self) -> u32 {
        self.0.width() * 64
    }
    fn height(&self) -> u32 {
        self.0.height() * 64
    }
    fn depth(&self) -> u32 {
        self.0.depth() * 64
    }
}

/// Number of [`TapeWord`] objects in the tape data flexible array
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

/// Returns a shader for 16^3 sub-tile evaluation
fn interval_tile16_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += INTERVAL_TILE16_SHADER;
    shader_code += INTERVAL_OPS_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += TAPE_INTERPRETER;
    shader_code += STACK_SHADER;
    shader_code += TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for column-per-workgroup processing
fn column_shader(reg_count: u8) -> String {
    let mut shader_code = String::new();
    shader_code += &opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};\n");
    shader_code += &format!("const TAPE_BUDGET: u32 = {TAPE_BUDGET};\n");
    shader_code += COLUMN_SHADER;
    shader_code += COMMON_SHADER;
    shader_code += STACK_SHADER;
    shader_code
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

    fn get(&self, reg_count: u8) -> &wgpu::ComputePipeline {
        let (r, v) = self.0.range(reg_count..).next().unwrap();
        assert!(*r >= reg_count);
        v
    }
}

/// Root context, which produces a dense status buffer of 64³ tiles
struct RootContext {
    root_pipeline: RegPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl RootContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tape_data
                    buffer_rw(1), // root_status (dense)
                    buffer_rw(2), // tile64_zmin
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
                        resource: buffers.root_status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile64_zmin.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.root_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        let render_size = buffers.render_size();
        let nx = render_size.nx().div_ceil(4);
        let ny = render_size.ny().div_ceil(4);
        let nz = render_size.nz().div_ceil(4);
        compute_pass.dispatch_workgroups(nx, ny, nz);
    }
}

////////////////////////////////////////////////////////////////////////////////

/// 16^3 sub-tile evaluation context
struct Tile16Context {
    tile16_pipeline: RegPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Tile16Context {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_rw(0), // tape_data (read_write for simplification)
                    buffer_ro(1), // root_status
                    buffer_rw(2), // tile16_status
                ],
            });

        let tile16_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = interval_tile16_shader(reg_count);
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
                label: Some(&format!("interval tile16 ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile16_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            tile16_pipeline,
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
                        resource: buffers.root_status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile16_status.as_entire_binding(),
                    },
                ],
            });

        compute_pass
            .set_pipeline(self.tile16_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // One workgroup per root tile; each thread handles one 16^3 sub-tile
        let render_size = buffers.render_size();
        compute_pass.dispatch_workgroups(
            render_size.nx(),
            render_size.ny(),
            render_size.nz(),
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Column processing context — replaces all interval_tiles, voxel, normals,
/// repack, and backfill contexts
struct ColumnContext {
    column_pipeline: RegPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ColumnContext {
    fn new(
        device: &wgpu::Device,
        common_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_ro(0), // tape_data
                    buffer_ro(1), // root_status
                    buffer_ro(2), // tile64_zmin
                    buffer_ro(3), // tile16_status
                    buffer_rw(4), // workgroup_tapes
                    buffer_rw(5), // image_out
                ],
            });

        let column_pipeline = RegPipeline::build(|reg_count| {
            let shader_code = column_shader(reg_count);
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
                label: Some(&format!("column ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("column_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            bind_group_layout,
            column_pipeline,
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
                        resource: buffers.root_status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.tile64_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.tile16_status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.workgroup_tapes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers.geom.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(self.column_pipeline.get(shape.reg_count));
        compute_pass.set_bind_group(1, &bind_group, &[]);

        // Fixed workgroup count, grid-stride loop inside the shader
        let render_size = buffers.render_size();
        let total_work_items = render_size.nx() * render_size.ny() * 16;
        let dispatch_count = total_work_items.min(COLUMN_WORKGROUP_COUNT);
        compute_pass.dispatch_workgroups(dispatch_count, 1, 1);
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
    config_buf: wgpu::Buffer,

    /// Interval root context, generating dense root status
    root_ctx: RootContext,

    /// 16^3 sub-tile evaluation context
    tile16_ctx: Tile16Context,

    /// Column processing context (4^3 interval + voxel eval + inline gradient normals)
    column_ctx: ColumnContext,
}

impl Context {
    /// Build a new 3D rendering context, given a device and queue
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &common_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            }],
        });

        let root_ctx = RootContext::new(&device, &common_bind_group_layout);
        let tile16_ctx =
            Tile16Context::new(&device, &common_bind_group_layout);
        let column_ctx = ColumnContext::new(&device, &common_bind_group_layout);

        Self {
            device,
            queue,
            bind_group,
            config_buf,
            root_ctx,
            tile16_ctx,
            column_ctx,
        }
    }

    /// Builds a new [`Buffers`] object for the given render size
    pub fn buffers(&self, image_size: VoxelSize) -> Buffers {
        Buffers::new(&self.device, image_size)
    }

    /// Builds a new [`RenderShape`] object for the given shape
    pub fn shape(&self, shape: &VmShape) -> RenderShape {
        RenderShape::new(&self.device, shape)
    }

    /// Renders the image, with a blocking wait to read pixel data from the GPU
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &self,
        shape: &RenderShape,
        buffers: &Buffers,
        settings: RenderConfig,
    ) -> GeometryBuffer {
        self.submit(shape, buffers, &settings);
        // self.print_diagnostics(buffers);
        let buffer_slice = self.map_image(buffers);
        self.read_mapped_image(buffers, &buffer_slice)
    }

    /// Read and print diagnostic counters from the GPU
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    fn print_diagnostics(&self, buffers: &Buffers) {
        let diag = self.read_buffer::<u32>(&buffers.diagnostics);
        if diag.iter().any(|&v| v != 0) {
            eprintln!("=== Shader Diagnostics ===");
            eprintln!("  z16 ambiguous (eval'd): {}", diag[0]);
            eprintln!("  z16 empty (skipped):    {}", diag[1]);
            eprintln!("  z16 filled (skipped):   {}", diag[2]);
            eprintln!("  z16 using base tape:    {}", diag[3]);
            eprintln!("  ambiguous 4^3 tiles:    {}", diag[4]);
            eprintln!("  voxel evals:            {}", diag[5]);
            eprintln!("  voxel hits:             {}", diag[6]);
            eprintln!("  z16 using simp'd tape:  {}", diag[7]);
            let total_z16 = diag[0] + diag[1] + diag[2];
            if total_z16 > 0 {
                eprintln!(
                    "  z16 skip rate:          {:.1}%",
                    (diag[1] + diag[2]) as f64 / total_z16 as f64 * 100.0
                );
            }
            if diag[0] > 0 {
                eprintln!(
                    "  z16 simplify rate:      {:.1}%",
                    diag[7] as f64 / diag[0] as f64 * 100.0
                );
            }
            eprintln!("==========================");
        }
    }

    /// Renders the image, with an async wait to read pixel data from the GPU
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
            max_tiles_per_dispatch: 0, // unused in column approach
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

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            })
    }

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

        let mut encoder = self.new_encoder();

        // Clear buffers
        encoder.clear_buffer(&shape.tape_buf, 0, Some(4));
        encoder.clear_buffer(&buffers.root_status, 0, None);
        encoder.clear_buffer(&buffers.tile64_zmin, 0, None);
        encoder.clear_buffer(&buffers.tile16_status, 0, None);
        encoder.clear_buffer(&buffers.diagnostics, 0, None);
        encoder.clear_buffer(&buffers.geom, 0, None);

        // Three-pass rendering: root → tile16 → column (depth + normals)
        let mut compute_pass = self.begin_compute_pass(&mut encoder);

        // Pass 1: Dense root tile evaluation (64^3)
        self.root_ctx.run(self, shape, buffers, &mut compute_pass);

        // Pass 2: 16^3 sub-tile evaluation + simplification
        self.tile16_ctx.run(self, shape, buffers, &mut compute_pass);

        // Pass 3: Column-per-workgroup (4^3 + voxel + inline gradient normals)
        self.column_ctx.run(self, shape, buffers, &mut compute_pass);

        drop(compute_pass);

        // Copy from STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &buffers.geom,
            0,
            &buffers.image,
            0,
            buffers.geom.size(),
        );

        self.queue.submit(Some(encoder.finish()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn map_image<'a>(&self, buffers: &'a Buffers) -> wgpu::BufferSlice<'a> {
        let buffer_slice = buffers.image.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        buffer_slice
    }

    #[cfg(target_arch = "wasm32")]
    async fn map_image_async<'a>(
        &self,
        buffers: &'a Buffers,
    ) -> wgpu::BufferSlice<'a> {
        let buffer_slice = buffers.image.slice(..);
        let (tx, rx) = flume::bounded(0);
        buffer_slice
            .map_async(wgpu::MapMode::Read, move |_| tx.send(()).unwrap());
        rx.recv_async().await.unwrap();
        buffer_slice
    }

    fn read_mapped_image(
        &self,
        buffers: &Buffers,
        buffer_slice: &wgpu::BufferSlice,
    ) -> GeometryBuffer {
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

/// Shape for rendering
#[derive(Clone)]
pub struct RenderShape {
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
        let tape_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tape_data"),
            size: 12
                + (TAPE_DATA_CAPACITY * std::mem::size_of::<TapeWord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
            mapped_at_creation: true,
        });

        let vars = shape.inner().vars();
        let bytecode = Bytecode::new(shape.inner().data());
        eprintln!(
            "Tape: {} TapeWords ({} bytes), {} registers",
            bytecode.len() / 2,
            bytecode.len() * 4,
            bytecode.reg_count()
        );
        assert!(bytecode.len() < TAPE_DATA_CAPACITY);

        let axes = shape
            .axes()
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX));

        {
            let mut buffer_view = tape_buf.slice(..).get_mapped_range_mut();
            let bytes = bytecode.as_bytes();
            buffer_view[0..4]
                .copy_from_slice((bytecode.len() as u32 / 2).as_bytes());
            buffer_view[4..8]
                .copy_from_slice((bytecode.len() as u32 / 2).as_bytes());
            buffer_view[8..12]
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
#[derive(Clone)]
pub struct Buffers {
    /// Image render size
    image_size: VoxelSize,

    /// Dense root status buffer: one u32 per (x, y, z) root tile
    root_status: wgpu::Buffer,

    /// Per-XY packed (z, tape_index) for filled root tiles
    tile64_zmin: wgpu::Buffer,

    /// Dense 16^3 tile status: one u32 per (x, y, z) tile
    tile16_status: wgpu::Buffer,

    /// Per-thread tape storage for local simplification
    workgroup_tapes: wgpu::Buffer,

    /// Diagnostic counters (8 x u32)
    diagnostics: wgpu::Buffer,

    /// Buffer of GeometryPixel data (z, normal_x, normal_y, normal_z)
    geom: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    image: wgpu::Buffer,
}

impl Buffers {
    fn new(device: &wgpu::Device, image_size: VoxelSize) -> Self {
        let render_size = RenderSize::from(image_size);
        let image_pixels =
            image_size.width() as usize * image_size.height() as usize;

        let root_tile_count = render_size.nx() as usize
            * render_size.ny() as usize
            * render_size.nz() as usize;

        let root_status = new_buffer::<u32>(
            device,
            "root_status",
            root_tile_count,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );

        let tile64_xy_count =
            render_size.nx() as usize * render_size.ny() as usize;
        let tile64_zmin = new_buffer::<Voxel>(
            device,
            "tile64_zmin",
            tile64_xy_count,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );

        // 16^3 tiles: 4x more resolution than root tiles on each axis
        let tile16_count = render_size.nx() as usize
            * render_size.ny() as usize
            * render_size.nz() as usize
            * 64; // 4^3 = 64 sub-tiles per root tile
        let tile16_status = new_buffer::<u32>(
            device,
            "tile16_status",
            tile16_count,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | DEBUG_FLAGS,
        );

        // COLUMN_WORKGROUP_COUNT workgroups × 64 threads × TAPE_BUDGET TapeWords
        let workgroup_tape_count = COLUMN_WORKGROUP_COUNT as usize
            * 64
            * TAPE_BUDGET as usize;
        let workgroup_tapes = new_buffer::<TapeWord>(
            device,
            "workgroup_tapes",
            workgroup_tape_count,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let diagnostics = new_buffer::<u32>(
            device,
            "diagnostics",
            8,
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

        Self {
            image_size,
            root_status,
            tile64_zmin,
            tile16_status,
            workgroup_tapes,
            diagnostics,
            geom,
            image,
        }
    }

    fn render_size(&self) -> RenderSize {
        self.image_size.into()
    }
}

/// Backwards-compatible GpuSpec (ignored in column approach)
#[derive(Copy, Clone, Debug)]
pub enum GpuSpec {
    Low,
    Medium,
    High,
    Custom(u32),
}

#[cfg(test)]
mod test {
    use super::*;
    use heck::ToShoutySnakeCase;

    #[test]
    fn shader_has_all_ops() {
        // Check that the column shader has all opcodes (including gradient ops)
        for (op, _) in fidget_bytecode::iter_ops() {
            let op_name = format!("OP_{}", op.to_shouty_snake_case());
            assert!(
                COLUMN_SHADER.contains(&op_name),
                "column shader is missing {op_name}"
            );
        }
    }

    fn compile_shader(src: &str, desc: &str) {
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
            if let Some(i) = e.location(src) {
                let pos = i.offset as usize..(i.offset + i.length) as usize;
                panic!(
                    "shader compilation failed\n{}",
                    e.emit_to_string_with_path(&src[pos], desc)
                );
            } else {
                panic!(
                    "shader compilation failed\n{}",
                    e.emit_to_string(desc)
                );
            }
        });
        if let Err(e) = v.validate(&m) {
            let (pos, desc) = e.spans().next().unwrap();
            panic!(
                "shader compilation failed\n{}",
                e.emit_to_string_with_path(&src[pos.to_range().unwrap()], desc)
            );
        }
    }

    #[test]
    fn check_interval_root_shader() {
        compile_shader(&interval_root_shader(16), "interval root")
    }

    #[test]
    fn check_interval_tile16_shader() {
        compile_shader(&interval_tile16_shader(16), "interval_tile16")
    }

    #[test]
    fn check_column_shader() {
        compile_shader(&column_shader(16), "column")
    }


    /// GPU-specific tests
    mod gpu {
        use super::*;
        use fidget_core::{context::Tree, vm::VmShape};

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

        #[test]
        fn render_sphere() {
            CTX.with(|ctx| {
                let x = Tree::x();
                let y = Tree::y();
                let z = Tree::z();
                let sphere = (x.square() + y.square() + z.square()).sqrt()
                    - Tree::from(0.5);
                let shape = VmShape::from(sphere);
                let shape = ctx.shape(&shape);
                let size = VoxelSize::from(128);
                let buffers = ctx.buffers(size);
                let settings = RenderConfig::default();
                let result = ctx.run(&shape, &buffers, settings);

                // Check that we got some non-zero pixels
                let mut has_pixel = false;
                for y in 0..result.height() {
                    for x in 0..result.width() {
                        if result[(y, x)].depth > 0.0 {
                            has_pixel = true;
                        }
                    }
                }
                assert!(has_pixel, "render produced no visible pixels");
            });
        }
    }
}
