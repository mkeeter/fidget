//! GPU-accelerated 3D rendering
use crate::{
    Error,
    compiler::SsaOp,
    eval::Function,
    render::{GeometryBuffer, GeometryPixel, VoxelSize},
    vm::VmShape,
    wgpu::util::{resize_buffer_with, write_storage_buffer},
};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const VOXEL_TILES_SHADER: &str = include_str!("shaders/voxel_tiles.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const BACKFILL_SHADER: &str = include_str!("shaders/backfill.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");
const NORMALS_SHADER: &str = include_str!("shaders/normals.wgsl");

use std::collections::BTreeMap;
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

#[derive(Debug, IntoBytes, Immutable)]
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
    _padding3: u32,
}

struct IntervalTileContext {
    /// Input tiles (64^3)
    tile64_buffers: TileBuffers<64>,

    /// First stage output tiles (16^3)
    tile16_buffers: TileBuffers<16>,

    /// First stage output tiles (4^3)
    tile4_buffers: TileBuffers<4>,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Shape-specific render pipelines
pub struct ShapePipelines {
    interval_pipeline: wgpu::ComputePipeline,
    voxel_pipeline: wgpu::ComputePipeline,
    normals_pipeline: wgpu::ComputePipeline,
    mat: nalgebra::Matrix4<f32>,
    axes: [u32; 3],
}

impl ShapePipelines {
    fn new(
        device: &wgpu::Device,
        shape: &VmShape,
        interval_ctx: &IntervalTileContext,
        voxel_ctx: &VoxelContext,
        normals_ctx: &NormalsContext,
    ) -> Self {
        let vars = shape.inner().vars();
        let axes = shape
            .axes()
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX));

        let mut shader_code = make_shader(shape, "vec2f", 3, 1);
        shader_code += INTERVAL_TILES_SHADER;
        shader_code += COMMON_SHADER;
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&interval_ctx.bind_group_layout],
                push_constant_ranges: &[],
            });
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });
        let interval_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interval_tile"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut shader_code = make_shader(shape, "f32", 3, 1);
        shader_code += VOXEL_TILES_SHADER;
        shader_code += COMMON_SHADER;
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&voxel_ctx.bind_group_layout],
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

        let mut shader_code = make_shader(shape, "vec4f", 3, 1);
        shader_code += COMMON_SHADER;
        shader_code += NORMALS_SHADER;
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&normals_ctx.bind_group_layout],
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
            interval_pipeline,
            voxel_pipeline,
            normals_pipeline,
            mat: shape.transform().unwrap_or(nalgebra::Matrix4::identity()),
            axes,
        }
    }
}

fn buffer_cfg(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

impl IntervalTileContext {
    fn new(device: &wgpu::Device) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // tiles_in
                    buffer_ro(2),  // tile_zmin
                    buffer_rw(3),  // subtiles_out
                    buffer_rw(4),  // subtile_zmin
                    buffer_rw(5),  // count_clear
                ],
            });

        let tile64_buffers = TileBuffers::new(device);
        let tile16_buffers = TileBuffers::new(device);
        let tile4_buffers = TileBuffers::new(device);

        Self {
            bind_group_layout,
            tile64_buffers,
            tile16_buffers,
            tile4_buffers,
        }
    }

    fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active_tiles: BTreeMap<u32, Vec<u32>>,
        render_size: VoxelSize,
    ) -> Vec<(usize, usize)> {
        self.tile16_buffers.reset(device, queue, render_size);
        self.tile4_buffers.reset(device, queue, render_size);
        self.tile64_buffers.reset_with_sorted_tiles(
            device,
            queue,
            active_tiles,
            render_size,
        )
    }

    fn run(
        &mut self,
        active_offset: usize,
        active_count: usize,
        ctx: &CommonCtx,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group16 = self.create_bind_group(
            ctx,
            &self.tile64_buffers,
            Some((active_offset, active_count)),
            &self.tile16_buffers,
            &self.tile4_buffers.tiles,
        );
        compute_pass.set_pipeline(&ctx.pipelines.interval_pipeline);
        compute_pass.set_bind_group(0, &bind_group16, &[]);
        compute_pass.dispatch_workgroups(active_count as u32, 1, 1);

        let bind_group4 = self.create_bind_group(
            ctx,
            &self.tile16_buffers,
            None,
            &self.tile4_buffers,
            &self.tile64_buffers.tiles, // clear
        );
        compute_pass.set_bind_group(0, &bind_group4, &[]);
        compute_pass
            .dispatch_workgroups_indirect(&self.tile16_buffers.tiles, 0);
    }

    fn create_bind_group<const N: usize, const M: usize>(
        &self,
        ctx: &CommonCtx,
        tile_buffers: &TileBuffers<N>,
        offset_and_count: Option<(usize, usize)>,
        subtile_buffers: &TileBuffers<M>,
        clear: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: if let Some((offset, count)) = offset_and_count {
                        tile_buffers
                            .tiles
                            .slice(
                                offset as u64 * 4
                                    ..((offset + count + 5) as u64) * 4,
                            )
                            .into()
                    } else {
                        tile_buffers.tiles.as_entire_binding()
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_buffers.zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: subtile_buffers.tiles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: subtile_buffers.zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: clear.slice(0..16).into(),
                },
            ],
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

struct VoxelContext {
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl VoxelContext {
    fn new(device: &wgpu::Device) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // tiles4_in
                    buffer_ro(2),  // tile4_zmin
                    buffer_rw(3),  // result
                    buffer_rw(4),  // count_clear
                ],
            });

        Self { bind_group_layout }
    }

    fn run(
        &mut self,
        ctx: &CommonCtx,
        tile4_buffers: &TileBuffers<4>,
        clear: &wgpu::Buffer,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile4_buffers.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile4_buffers.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.buf.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: clear.slice(0..16).into(),
                    },
                ],
            });

        compute_pass.set_pipeline(&ctx.pipelines.voxel_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Each workgroup is 4x4x4, i.e. covering a 4x4 splat of pixels with 4x
        // workers in the Z direction.
        compute_pass.dispatch_workgroups_indirect(&tile4_buffers.tiles, 0);
    }
}

struct NormalsContext {
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl NormalsContext {
    fn new(device: &wgpu::Device) -> Self {
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // image_heightmap
                    buffer_rw(2),  // image_out
                ],
            });

        Self { bind_group_layout }
    }

    fn run(&mut self, ctx: &CommonCtx, compute_pass: &mut wgpu::ComputePass) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buf.merged.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.buf.geom.as_entire_binding(),
                    },
                ],
            });

        compute_pass.set_pipeline(&ctx.pipelines.normals_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(
            ctx.image_size.width().div_ceil(8),
            ctx.image_size.height().div_ceil(8),
            1,
        );
    }
}

fn format_op(op: SsaOp) -> String {
    match op {
        SsaOp::Output(..) => unreachable!(),
        SsaOp::Input(_, i) => format!("inputs[{i}];"),
        SsaOp::CopyReg(_, reg) => format!("r{reg};"),
        SsaOp::NegReg(_, reg) => format!("op_neg(r{reg});"),
        SsaOp::CopyImm(_, imm) => format!("build_imm({imm});"),
        SsaOp::AbsReg(_, reg) => format!("op_abs(r{reg});"),
        SsaOp::RecipReg(_, reg) => format!("op_recip(r{reg});"),
        SsaOp::SquareReg(_, reg) => format!("op_square(r{reg});"),
        SsaOp::SqrtReg(_, reg) => format!("op_sqrt(r{reg});"),
        SsaOp::FloorReg(_, reg) => format!("op_floor(r{reg});"),
        SsaOp::CeilReg(_, reg) => format!("op_ceil(r{reg});"),
        SsaOp::RoundReg(_, reg) => format!("op_round(r{reg});"),
        SsaOp::SinReg(_, reg) => format!("op_sin(r{reg});"),
        SsaOp::CosReg(_, reg) => format!("op_cos(r{reg});"),
        SsaOp::TanReg(_, reg) => format!("op_tan(r{reg});"),
        SsaOp::AsinReg(_, reg) => format!("op_asin(r{reg});"),
        SsaOp::AcosReg(_, reg) => format!("op_acos(r{reg});"),
        SsaOp::AtanReg(_, reg) => format!("op_atan(r{reg});"),
        SsaOp::ExpReg(_, reg) => format!("op_exp(r{reg});"),
        SsaOp::LnReg(_, reg) => format!("op_ln(r{reg});"),
        SsaOp::NotReg(_, reg) => format!("op_not(r{reg});"),
        SsaOp::AddRegImm(_, reg, imm) => {
            format!("op_add(r{reg}, build_imm({imm}));")
        }
        SsaOp::MulRegImm(_, reg, imm) => {
            format!("op_mul(r{reg}, build_imm({imm}));")
        }
        SsaOp::SubRegImm(_, reg, imm) => {
            format!("op_sub(r{reg}, build_imm({imm}));")
        }
        SsaOp::SubImmReg(_, reg, imm) => {
            format!("op_sub(build_imm({imm}), r{reg});")
        }
        SsaOp::DivRegImm(_, reg, imm) => {
            format!("op_div(r{reg}, build_imm({imm}));")
        }
        SsaOp::DivImmReg(_, reg, imm) => {
            format!("op_div(build_imm({imm}), r{reg});")
        }
        SsaOp::MaxRegImm(_, reg, imm) => {
            format!("op_max(r{reg}, build_imm({imm}));")
        }
        SsaOp::MinRegImm(_, reg, imm) => {
            format!("op_min(r{reg}, build_imm({imm}));")
        }
        SsaOp::ModRegImm(_, reg, imm) => {
            format!("op_mod(r{reg}, build_imm({imm}));")
        }
        SsaOp::AtanRegImm(_, reg, imm) => {
            format!("op_atan(r{reg}, build_imm({imm}));")
        }
        SsaOp::CompareRegImm(_, reg, imm) => {
            format!("op_compare(r{reg}, build_imm({imm}));")
        }
        SsaOp::ModImmReg(_, reg, imm) => {
            format!("op_mod(build_imm({imm}), r{reg});")
        }
        SsaOp::AtanImmReg(_, reg, imm) => {
            format!("op_atan(build_imm({imm}), r{reg});")
        }
        SsaOp::CompareImmReg(_, reg, imm) => {
            format!("op_compare(build_imm({imm}), r{reg});")
        }
        SsaOp::AndRegImm(_, reg, imm) => {
            format!("op_and(r{reg}, build_imm({imm}));")
        }
        SsaOp::OrRegImm(_, reg, imm) => {
            format!("op_or(r{reg}, build_imm({imm}));")
        }
        SsaOp::AddRegReg(_, lhs, rhs) => {
            format!("op_add(r{lhs}, r{rhs});")
        }
        SsaOp::SubRegReg(_, lhs, rhs) => {
            format!("op_sub(r{lhs}, r{rhs});")
        }
        SsaOp::MulRegReg(_, lhs, rhs) => {
            format!("op_mul(r{lhs}, r{rhs});")
        }
        SsaOp::DivRegReg(_, lhs, rhs) => {
            format!("op_div(r{lhs}, r{rhs});")
        }
        SsaOp::CompareRegReg(_, lhs, rhs) => {
            format!("op_compare(r{lhs}, r{rhs});")
        }
        SsaOp::AtanRegReg(_, lhs, rhs) => {
            format!("op_atan2(r{lhs}, r{rhs});")
        }
        SsaOp::ModRegReg(_, lhs, rhs) => {
            format!("op_mod(r{lhs}, r{rhs});")
        }
        SsaOp::MinRegReg(_, lhs, rhs) => {
            format!("op_min(r{lhs}, r{rhs});")
        }
        SsaOp::MaxRegReg(_, lhs, rhs) => {
            format!("op_max(r{lhs}, r{rhs});")
        }
        SsaOp::AndRegReg(_, lhs, rhs) => {
            format!("op_and(r{lhs}, r{rhs});")
        }
        SsaOp::OrRegReg(_, lhs, rhs) => {
            format!("op_or(r{lhs}, r{rhs});")
        }
    }
}

fn make_shader(
    f: &VmShape,
    ty: &str,
    input_count: usize,
    output_count: usize,
) -> String {
    let mut s = format!(
        "fn run_tape(inputs: array<{ty}, {input_count}>) -> array<{ty}, {output_count}> {{
    var out: array<{ty}, {output_count}>;
",
    );
    for op in f.inner().data().iter_ssa() {
        if let SsaOp::Output(reg, i) = op {
            s += &format!("out[{i}] = r{reg};\nreturn out;\n}}");
            return s;
        }

        let out = op.output().expect("opcode must have output");
        s += &format!("let r{out} = ");
        s += &format_op(op);
        s += "\n";
    }
    panic!("reached end of tape without an Output");
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 3D (voxel) rendering
pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: DynamicBuffers,

    config: wgpu::Buffer,
    interval_tile_ctx: IntervalTileContext,
    voxel_ctx: VoxelContext,
    normals_ctx: NormalsContext,
    backfill_ctx: BackfillContext,
    merge_ctx: MergeContext,
}

struct TileBuffers<const N: usize> {
    // TileList, so 4 u32s of count then active tile list
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

        if resize_buffer_with::<u32>(
            device,
            &mut self.tiles,
            &format!("active_tile{N}"),
            // wg_dispatch: [u32; 3]
            // count: u32,
            // tile_size: u32,
            5 + nx * ny * nz,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ) {
            queue
                .write_buffer_with(&self.tiles, 16, 4.try_into().unwrap())
                .unwrap()
                .copy_from_slice((N as u32).as_bytes());
        }

        // zero out the zmin buffer
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

    fn reset_with_sorted_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active_tiles: BTreeMap<u32, Vec<u32>>,
        render_size: VoxelSize,
    ) -> Vec<(usize, usize)> {
        let mut packed_tiles = vec![];
        let mut out = vec![];
        for (_z, ts) in active_tiles.iter().rev() {
            // Ensure 256-byte alignment for buffer offsets
            while packed_tiles.len() % 64 != 0 {
                packed_tiles.push(0u32);
            }
            out.push((packed_tiles.len(), ts.len()));
            packed_tiles.extend([0u32, 0, 0, ts.len() as u32, 64]);
            packed_tiles.extend(ts);
        }

        write_storage_buffer(
            device,
            queue,
            &mut self.tiles,
            &format!("active_tile{N}"),
            &packed_tiles,
        );

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

        out
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
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
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

#[derive(Copy, Clone)]
struct CommonCtx<'a> {
    device: &'a wgpu::Device,
    config: &'a wgpu::Buffer,
    buf: &'a DynamicBuffers,
    pipelines: &'a ShapePipelines,
    render_size: VoxelSize,
    image_size: VoxelSize,
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

        let config = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<Config>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dummy shape
        let interval_tile_ctx = IntervalTileContext::new(&device);
        let voxel_ctx = VoxelContext::new(&device);
        let normals_ctx = NormalsContext::new(&device);
        let backfill_ctx = BackfillContext::new(&device);
        let buffers = DynamicBuffers::new(&device);
        let merge_ctx = MergeContext::new(&device);

        Ok(Self {
            device,
            config,
            buffers,
            queue,
            interval_tile_ctx,
            voxel_ctx,
            normals_ctx,
            backfill_ctx,
            merge_ctx,
        })
    }

    /// Builds a [`ShapePipelines`] object for the given shape
    pub fn pipelines(
        &self,
        shape: VmShape, // XXX add ShapeVars here
    ) -> ShapePipelines {
        ShapePipelines::new(
            &self.device,
            &shape,
            &self.interval_tile_ctx,
            &self.voxel_ctx,
            &self.normals_ctx,
        )
    }

    /// Renders a single image using GPU acceleration
    ///
    /// Returns a heightmap
    pub fn run(
        &mut self,
        pipelines: &ShapePipelines,
        settings: RenderConfig,
    ) -> GeometryBuffer {
        // Create the 4x4 transform matrix
        let mat = pipelines.mat * settings.mat();
        let render_size = VoxelSize::new(
            settings.image_size.width().next_multiple_of(64),
            settings.image_size.height().next_multiple_of(64),
            settings.image_size.depth().next_multiple_of(64),
        );

        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: pipelines.axes,
            _padding1: 0u32,
            render_size: [
                render_size.width(),
                render_size.height(),
                render_size.depth(),
            ],
            _padding2: 0u32,
            image_size: [
                settings.image_size.width(),
                settings.image_size.height(),
                settings.image_size.depth(),
            ],
            _padding3: 0u32,
        };
        self.queue
            .write_buffer_with(
                &self.config,
                0,
                (std::mem::size_of_val(&config) as u64).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice(config.as_bytes());

        // Build the dense tile array of 64^3 tiles, and active tile list
        let nx = render_size.width() / 64;
        let ny = render_size.height() / 64;
        let nz = render_size.depth() / 64;
        let mut active_tiles: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = x + y * nx + z * nx * ny;
                    active_tiles.entry(z).or_default().push(i);
                }
            }
        }

        self.buffers
            .reset(&self.device, &self.queue, settings.image_size);
        let active_tile_offsets = self.interval_tile_ctx.reset(
            &self.device,
            &self.queue,
            active_tiles,
            settings.image_size,
        );

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        let ctx = CommonCtx {
            device: &self.device,
            config: &self.config,
            buf: &self.buffers,
            pipelines,
            render_size,
            image_size: settings.image_size,
        };

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        // Evaluate tiles in reverse-Z order by strata (64 voxels deep)
        for (offset, count) in active_tile_offsets {
            self.interval_tile_ctx
                .run(offset, count, &ctx, &mut compute_pass);
            self.voxel_ctx.run(
                &ctx,
                &self.interval_tile_ctx.tile4_buffers, // dispatch list
                &self.interval_tile_ctx.tile16_buffers.tiles, // to clear
                &mut compute_pass,
            );
            self.backfill_ctx.run(
                &self.interval_tile_ctx.tile4_buffers.zmin,
                &self.interval_tile_ctx.tile16_buffers.zmin,
                &self.interval_tile_ctx.tile64_buffers.zmin,
                &ctx,
                &mut compute_pass,
            );
        }
        self.merge_ctx.run(
            &self.interval_tile_ctx.tile4_buffers.zmin,
            &self.interval_tile_ctx.tile16_buffers.zmin,
            &self.interval_tile_ctx.tile64_buffers.zmin,
            &ctx,
            &mut compute_pass,
        );
        self.normals_ctx.run(&ctx, &mut compute_pass);
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &ctx.buf.geom,
            0,
            &ctx.buf.image,
            0,
            ctx.buf.geom.size(),
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

        GeometryBuffer::build(result, settings.image_size).unwrap()
    }
}

struct BackfillContext {
    bind_group_layout: wgpu::BindGroupLayout,

    pipeline4: wgpu::ComputePipeline,
    pipeline16: wgpu::ComputePipeline,
    pipeline64: wgpu::ComputePipeline,
}

impl BackfillContext {
    fn new(device: &wgpu::Device) -> Self {
        let shader_code = BACKFILL_SHADER.to_owned() + COMMON_SHADER;
        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // subtile_zmin
                    buffer_rw(2),  // count_clear
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
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
        &mut self,
        tile4_zmin: &wgpu::Buffer,
        tile16_zmin: &wgpu::Buffer,
        tile64_zmin: &wgpu::Buffer,
        ctx: &CommonCtx,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let nx = ctx.render_size.width().div_ceil(64) as usize;
        let ny = ctx.render_size.width().div_ceil(64) as usize;

        let bind_group4 =
            self.create_bind_group(ctx, &ctx.buf.result, tile4_zmin);
        compute_pass.set_pipeline(&self.pipeline4);
        compute_pass.set_bind_group(0, &bind_group4, &[]);
        compute_pass.dispatch_workgroups((nx * ny * 16) as u32, 1, 1);

        let bind_group16 = self.create_bind_group(ctx, tile4_zmin, tile16_zmin);
        compute_pass.set_pipeline(&self.pipeline16);
        compute_pass.set_bind_group(0, &bind_group16, &[]);
        compute_pass.dispatch_workgroups((ny * ny * 4) as u32, 1, 1);

        let bind_group64 =
            self.create_bind_group(ctx, tile16_zmin, tile64_zmin);
        compute_pass.set_pipeline(&self.pipeline64);
        compute_pass.set_bind_group(0, &bind_group64, &[]);
        compute_pass.dispatch_workgroups((ny * ny) as u32, 1, 1);
    }

    fn create_bind_group(
        &self,
        ctx: &CommonCtx,
        subtile_zmin: &wgpu::Buffer,
        tile_zmin: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: subtile_zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_zmin.as_entire_binding(),
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
    fn new(device: &wgpu::Device) -> Self {
        let shader_code = MERGE_SHADER.to_owned() + COMMON_SHADER;

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // tile64_zmin
                    buffer_ro(2),  // tile16_zmin
                    buffer_ro(3),  // tile4_zmin
                    buffer_ro(4),  // result
                    buffer_rw(5),  // merged
                ],
            });

        // Create the compute pipeline
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
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
        &mut self,
        tile4_zmin: &wgpu::Buffer,
        tile16_zmin: &wgpu::Buffer,
        tile64_zmin: &wgpu::Buffer,
        ctx: &CommonCtx,
        compute_pass: &mut wgpu::ComputePass,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile64_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile16_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tile4_zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.buf.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.buf.merged.as_entire_binding(),
                    },
                ],
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            ctx.image_size.width().div_ceil(8),
            ctx.image_size.width().div_ceil(8),
            1,
        );
    }
}
