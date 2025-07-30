use crate::{
    Error,
    eval::{Function, MathFunction},
    render::{
        RenderHandle, RenderWorker, Tile as RenderTile, TileSizesRef,
        VoxelRenderConfig, VoxelSize,
    },
    shape::{Shape, ShapeTracingEval, ShapeVars},
    types::Interval,
    wgpu::{
        interval_subtiles_shader, interval_tiles_shader, resize_buffer,
        resize_buffer_with, voxel_ray_shader, write_storage_buffer,
    },
};

use nalgebra::{Point3, Vector3};
use std::collections::HashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes};

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct IntervalTileConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    /// Number of tiles in the active tile list
    active_tile_count: u32,

    /// Total window size
    image_size: [u32; 3],
    _padding: u32,
}

struct IntervalTileContext {
    /// Configuration data (fixed size)
    config_buf: wgpu::Buffer,

    /// Active tile list (flexibly sized, with one tile per workgroup)
    active_tile64: wgpu::Buffer,

    /// Output from this stage
    active_tile16_count: wgpu::Buffer,
    active_tile16: wgpu::Buffer,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
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
    fn new(device: &wgpu::Device) -> Result<Self, Error> {
        let shader_code = interval_tiles_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // tape_data
                    buffer_ro(2),  // tile64_tapes
                    buffer_ro(3),  // active_tile64
                    buffer_rw(4),  // tile64_next
                    buffer_rw(5),  // tile64_occupancy
                    buffer_rw(6),  // active_tile16_count
                    buffer_rw(7),  // active_tile16
                    buffer_rw(8),  // tile16_occupancy
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
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile16_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<IntervalTileConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let active_tile64 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch active_tile64"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let active_tile16 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch active_tile16"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // These buffers never change size
        let active_tile16_count =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("active_tile16_count"),
                size: 3 * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });

        Ok(Self {
            pipeline,
            bind_group_layout,
            config_buf,
            active_tile16_count,
            active_tile64,
            active_tile16,
        })
    }

    fn run<F: Function + MathFunction>(
        &mut self,
        active_tiles: &[u32],
        ctx: &CommonCtx<F>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let vars = ctx.shape.inner().vars();
        let config = IntervalTileConfig {
            mat: ctx.mat.data.as_slice().try_into().unwrap(),
            axes: ctx
                .shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            image_size: [
                ctx.settings.image_size.width(),
                ctx.settings.image_size.height(),
                ctx.settings.image_size.depth(),
            ],
            active_tile_count: active_tiles.len().try_into().unwrap(),
            _padding: 0u32,
        };
        println!("{config:?}");
        println!("{}", ctx.mat);

        // Load the config into our buffer
        ctx.queue
            .write_buffer_with(
                &self.config_buf,
                0,
                (std::mem::size_of_val(&config) as u64).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice(config.as_bytes());

        // Clear the `active_tile16_count` global counter
        ctx.queue
            .write_buffer_with(
                &self.active_tile16_count,
                0,
                (std::mem::size_of::<[u32; 3]>() as u64).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice([0u32, 1, 1].as_bytes());

        // Write active tiles to our buffer
        write_storage_buffer(
            ctx.device,
            ctx.queue,
            &mut self.active_tile64,
            "active tiles",
            active_tiles,
        );

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.config_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buf.tape_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.buf.tile64_tapes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.active_tile64.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.buf.tile64_next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.buf.tile64_occupancy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.active_tile16_count.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.active_tile16.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: ctx.buf.tile16_occupancy.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(
            (active_tiles.len() % 65536) as u32,
            (active_tiles.len() / 65536).max(1).try_into().unwrap(),
            1,
        );
        drop(compute_pass);
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct IntervalSubtileConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],
    _padding1: u32,

    /// Total window size
    image_size: [u32; 3],
    _padding2: u32,
}

struct IntervalSubtileContext {
    /// Configuration data (fixed size)
    config_buf: wgpu::Buffer,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IntervalSubtileContext {
    fn new(device: &wgpu::Device) -> Result<Self, Error> {
        let shader_code = interval_subtiles_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // tape_data
                    buffer_ro(2),  // tile64_tapes
                    buffer_ro(3),  // active_tile16_count
                    buffer_ro(4),  // active_tile16
                    buffer_rw(5),  // tile16_occupancy
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
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile4_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<IntervalTileConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            config_buf,
        })
    }

    fn run<F: Function + MathFunction>(
        &mut self,
        ctx: &CommonCtx<F>,
        active_tile16_count: &wgpu::Buffer,
        active_tile16: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let vars = ctx.shape.inner().vars();
        let config = IntervalSubtileConfig {
            mat: ctx.mat.data.as_slice().try_into().unwrap(),
            axes: ctx
                .shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            image_size: [
                ctx.settings.image_size.width(),
                ctx.settings.image_size.height(),
                ctx.settings.image_size.depth(),
            ],
            _padding1: 0,
            _padding2: 0,
        };

        // Load the config into our buffer
        ctx.queue
            .write_buffer_with(
                &self.config_buf,
                0,
                (std::mem::size_of_val(&config) as u64).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice(config.as_bytes());

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.config_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buf.tape_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.buf.tile64_tapes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: active_tile16_count.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: active_tile16.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.buf.tile16_occupancy.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups_indirect(active_tile16_count, 0);
        drop(compute_pass);
    }
}
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct VoxelConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    _padding1: u32,

    /// Total window size
    image_size: [u32; 3],

    _padding2: u32,
}

struct VoxelContext {
    /// Configuration data (fixed size)
    config_buf: wgpu::Buffer,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl VoxelContext {
    fn new(device: &wgpu::Device) -> Result<Self, Error> {
        let shader_code = voxel_ray_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    buffer_cfg(0), // config
                    buffer_ro(1),  // tape_data
                    buffer_ro(2),  // tile64_tapes
                    buffer_ro(3),  // tile64_occupancy
                    buffer_ro(4),  // tile64_next
                    buffer_ro(5),  // tile16_occupancy
                    buffer_rw(6),  // result
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
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("voxel_ray_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<VoxelConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            config_buf,
            bind_group_layout,
        })
    }

    fn run<F: Function>(
        &mut self,
        ctx: &CommonCtx<F>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let vars = ctx.shape.inner().vars();
        let config = VoxelConfig {
            mat: ctx.mat.data.as_slice().try_into().unwrap(),
            axes: ctx
                .shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            _padding1: 0,
            image_size: [
                ctx.settings.image_size.width(),
                ctx.settings.image_size.height(),
                ctx.settings.image_size.depth(),
            ],
            _padding2: 0,
        };

        // Load the config into our buffer
        ctx.queue
            .write_buffer_with(
                &self.config_buf,
                0,
                (std::mem::size_of_val(&config) as u64).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice(config.as_bytes());

        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.config_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buf.tape_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.buf.tile64_tapes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.buf.tile64_occupancy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.buf.tile64_next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.buf.tile16_occupancy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: ctx.buf.result.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Each workgroup is 4x4x4, i.e. covering a 4x4 splat of pixels with 4x
        // workers in the Z direction.
        compute_pass.dispatch_workgroups(
            ctx.settings.image_size.width().div_ceil(4),
            ctx.settings.image_size.height().div_ceil(4),
            1,
        );
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &ctx.buf.result,
            0,
            &ctx.buf.image,
            0,
            ctx.buf.result.size(),
        );
    }
}

fn get_buffer<T>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
) -> Vec<T>
where
    T: FromBytes + Immutable + Clone,
{
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch"),
        size: buf.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
    encoder.copy_buffer_to_buffer(buf, 0, &scratch, 0, buf.size());

    queue.submit(Some(encoder.finish()));

    let buffer_slice = scratch.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let out = <[T]>::ref_from_bytes(&buffer_slice.get_mapped_range())
        .unwrap()
        .to_vec();
    scratch.unmap();
    out
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 3D (voxel) rendering
pub struct VoxelRayContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: DynamicBuffers,

    interval_tile_ctx: IntervalTileContext,
    interval_subtile_ctx: IntervalSubtileContext,
    voxel_ctx: VoxelContext,
}

/// Buffers which must be dynamically sized
pub struct DynamicBuffers {
    tape_data: wgpu::Buffer,

    /// Densely-packed tape indices (or special values to indicate empty / full)
    tile64_tapes: wgpu::Buffer,

    /// Densely-packed occupancy arrays
    tile64_occupancy: wgpu::Buffer,

    /// Densely-packed 'next tile' pointers
    tile64_next: wgpu::Buffer,

    /// Sparsely-packed occupancy arrays
    tile16_occupancy: wgpu::Buffer,

    /// Result buffer written by the voxel compute shader
    ///
    /// Note that this buffer can't be read by the host; it must be copied to
    /// 'image_buf`
    ///
    /// (dynamic size, implicit from image size in config)
    result: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    ///
    /// (dynamic size, implicit from image size in config)
    image: wgpu::Buffer,
}

impl DynamicBuffers {
    fn new(device: &wgpu::Device) -> Self {
        let scratch = |name| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: 1,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            tape_data: scratch("dummy tape data"),
            tile64_tapes: scratch("dummy tile64 tapes"),
            tile64_occupancy: scratch("dummy tile64 occupancy"),
            tile64_next: scratch("dummy tile64 next"),
            tile16_occupancy: scratch("dummy tile16 occupancy"),
            result: scratch("dummy result"),
            image: scratch("dummy image"),
        }
    }
    fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active_tile16: &mut wgpu::Buffer,
        tape_data: &[u32],
        tile64_tapes: &[u32],
        image_size: VoxelSize,
    ) {
        write_storage_buffer(
            device,
            queue,
            &mut self.tape_data,
            "tape_data",
            tape_data,
        );
        write_storage_buffer(
            device,
            queue,
            &mut self.tile64_tapes,
            "tile64_tapes",
            tile64_tapes,
        );
        let tile64_count = u64::from(image_size.width().div_ceil(64))
            * u64::from(image_size.height().div_ceil(64))
            * u64::from(image_size.depth().div_ceil(64));
        resize_buffer::<[u32; 4]>(
            device,
            &mut self.tile64_occupancy,
            "tile64_occupancy",
            tile64_count as usize,
        );
        resize_buffer::<u32>(
            device,
            &mut self.tile64_next,
            "tile64_next",
            tile64_count as usize,
        );

        let tile16_count = tile64_count * 4u64.pow(3);
        resize_buffer::<u32>(
            device,
            active_tile16,
            "active_tile16",
            tile16_count as usize,
        );
        resize_buffer::<[u32; 4]>(
            device,
            &mut self.tile16_occupancy,
            "tile16_occupancy",
            tile16_count as usize,
        );

        let image_pixels =
            image_size.width() as usize * image_size.height() as usize;
        resize_buffer_with::<u32>(
            device,
            &mut self.result,
            "result",
            image_pixels,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        resize_buffer_with::<u32>(
            device,
            &mut self.image,
            "image",
            image_pixels,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
    }
}

#[derive(Copy, Clone)]
pub struct CommonCtx<'a, 'b, F> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,

    buf: &'a DynamicBuffers,
    settings: &'a VoxelRenderConfig<'b>,

    shape: &'a Shape<F>,
    mat: nalgebra::Matrix4<f32>,
}

impl VoxelRayContext {
    /// Build a new 2D (pixel) rendering context
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
                .ok_or(Error::NoAdapter)?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .map_err(Error::NoDevice)
        })?;

        let interval_tile_ctx = IntervalTileContext::new(&device)?;
        let interval_subtile_ctx = IntervalSubtileContext::new(&device)?;
        let voxel_ctx = VoxelContext::new(&device)?;
        let buffers = DynamicBuffers::new(&device);

        Ok(Self {
            device,
            buffers,
            queue,
            interval_tile_ctx,
            interval_subtile_ctx,
            voxel_ctx,
        })
    }

    /// Renders a single image using GPU acceleration
    ///
    /// Returns a heightmap
    pub fn run_3d<F: Function + MathFunction>(
        &mut self,
        shape: Shape<F>, // XXX add ShapeVars here
        settings: VoxelRenderConfig,
    ) -> Option<Result<Vec<u32>, Error>> {
        if settings.tile_sizes.iter().last().unwrap() != &64 {
            panic!("Voxel raycast only uses 64x64x64 tile rendering");
        };

        // Convert to a 4x4 matrix and apply to the shape
        let shape = shape.apply_transform(settings.mat());
        let mat = shape.transform().unwrap();

        let max_size = settings
            .image_size
            .width()
            .max(settings.image_size.height()) as usize;
        let tile_sizes = TileSizesRef::new(&settings.tile_sizes, max_size);

        let start = std::time::Instant::now();
        let rs = crate::render::render_tiles::<F, Worker3D<F>>(
            shape.clone(),
            &ShapeVars::new(),
            &settings,
        )?;
        println!("got tiles in {:?}", start.elapsed());

        let bytecode = shape.inner().to_bytecode();
        let mut max_reg = bytecode.reg_count;
        let mut max_mem = bytecode.mem_count;

        let mut pos = HashMap::new(); // map from shape to bytecode start
        let mut tape_data = bytecode.data; // address 0 is the original tape
        pos.insert(shape.inner().id(), 0);

        // Build the dense tile array of 64^3 tiles, and active tile list
        let tile_size = tile_sizes.last() as u32;
        let nx = settings.image_size.width().div_ceil(64);
        let ny = settings.image_size.height().div_ceil(64);
        let nz = settings.image_size.depth().div_ceil(64);
        let mut tile64_occupancy =
            vec![0u32; nx as usize * ny as usize * nz as usize];

        // Iterate over 64^3 tile results
        let mut active_tiles = vec![];
        let mut full = 0;
        let mut empty = 0;
        for r in rs.iter().flat_map(|(_t, r)| r.iter()) {
            let id = r.shape.inner().id();
            let start = pos.entry(id).or_insert_with(|| {
                let prev = tape_data.len();
                let bytecode = r.shape.inner().to_bytecode();
                max_reg = max_reg.max(bytecode.reg_count);
                max_mem = max_mem.max(bytecode.mem_count);
                tape_data.extend(bytecode.data.into_iter());
                prev
            });
            let start = *start as u32;
            assert_eq!(r.corner.x % 64, 0);
            assert_eq!(r.corner.y % 64, 0);
            assert_eq!(r.corner.z % 64, 0);

            for x in 0..r.tile_size / tile_size {
                for y in 0..r.tile_size / tile_size {
                    for z in 0..r.tile_size / tile_size {
                        let i = r.corner.x / tile_size
                            + x
                            + (r.corner.y / tile_size + y) * nx
                            + (r.corner.z / tile_size + z) * nx * ny;
                        tile64_occupancy[i as usize] = match r.mode {
                            TileMode::Empty => {
                                empty += 1;
                                u32::MAX
                            }
                            TileMode::Full => {
                                full += 1;
                                start | (1 << 31)
                            }
                            TileMode::Voxels => {
                                active_tiles.push(i);
                                start
                            }
                        };
                    }
                }
            }
        }

        if tape_data.len() < 1024 {
            println!("bytecode len: {} B", tape_data.len() * 4);
        } else {
            println!("bytecode len: {} KiB", tape_data.len() * 4 / 1024);
        }
        println!("max reg: {max_reg}");
        println!(
            "full: {full}\nempty: {empty}\nvoxels: {}",
            active_tiles.len()
        );
        assert_eq!(max_mem, 0, "external memory is not yet supported");

        self.buffers.resize(
            &self.device,
            &self.queue,
            &mut self.interval_tile_ctx.active_tile16,
            &tape_data,
            &tile64_occupancy,
            settings.image_size,
        );

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        let ctx = CommonCtx {
            device: &self.device,
            queue: &self.queue,
            buf: &self.buffers,
            mat,
            shape: &shape,
            settings: &settings,
        };

        self.interval_tile_ctx
            .run(&active_tiles, &ctx, &mut encoder);
        self.interval_subtile_ctx.run(
            &ctx,
            &self.interval_tile_ctx.active_tile16_count,
            &self.interval_tile_ctx.active_tile16,
            &mut encoder,
        );
        self.voxel_ctx.run(&ctx, &mut encoder);

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.buffers.image.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        // Get the pixel-populated image
        let mut result =
            <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        self.buffers.image.unmap();

        // Convert from absolute heightmap to greyscale
        // XXX normals?
        let m = result.iter().max().cloned().unwrap_or(1).max(1);
        for r in &mut result {
            if *r > 0 {
                let i = ((*r as u64) * 255 / m as u64) as u8 as u32;
                *r = 0xFF << 24 | i << 8 | i << 16 | i;
            }
        }

        if false {
            let size = get_buffer::<u32>(
                &self.device,
                &self.queue,
                &self.interval_tile_ctx.active_tile16_count,
            );
            println!("got dispatch size {size:?}");
            let active = get_buffer::<u32>(
                &self.device,
                &self.queue,
                &self.interval_tile_ctx.active_tile16,
            );
            println!(
                "got active tile16 {:?}",
                &active[..size[0] as usize + 65536 * (size[1] as usize - 1)]
            );
            println!(
                "got active tile16 {:?}",
                active[..size[0] as usize + 65536 * (size[1] as usize - 1)]
                    .iter()
                    .map(|t| (t % 4, (t / 4) % 4, (t / 16)))
                    .collect::<Vec<_>>()
            );

            let occupancy64 = get_buffer::<[u32; 4]>(
                &self.device,
                &self.queue,
                &self.buffers.tile64_occupancy,
            );
            print_occupancy_map(settings.image_size, 64, &occupancy64);
            println!("----------------------------------------\n");
            let occupancy16 = get_buffer::<[u32; 4]>(
                &self.device,
                &self.queue,
                &self.buffers.tile16_occupancy,
            );
            print_occupancy_map(settings.image_size, 16, &occupancy16);
        }

        Some(Ok(result))
    }
}

fn print_occupancy_map(
    size: VoxelSize,
    tile_size: u32,
    occupancy: &[[u32; 4]],
) {
    let nx = size.width().div_ceil(tile_size);
    let ny = size.height().div_ceil(tile_size);
    let nz = size.depth().div_ceil(tile_size);
    for z in 0..nz * 4 {
        println!("Z = {}", z * tile_size / 4);
        for y in 0..ny * 4 {
            for x in 0..nx * 4 {
                let tile = x / 4 + y / 4 * nx + z / 4 * nx * ny;
                let b = 2 * ((z % 4) + (y % 4) * 4 + (x % 4) * 16) as usize;
                let c = match (occupancy[tile as usize][b / 32] >> (b % 32))
                    & 0b11
                {
                    0 => '?',
                    1 => 'X',
                    2 => '.',
                    3 => '-',
                    _ => panic!(),
                };
                print!("{c}{c}");
            }
            println!();
        }
        println!();
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
enum TileMode {
    Full,
    Empty,
    Voxels,
}

struct WorkResult<F> {
    /// Tile corner (in image space)
    corner: Point3<u32>,
    /// Tile size, in pixels
    tile_size: u32,
    /// Shape at this tile
    shape: Shape<F>,
    /// Fill or render individual pixels
    mode: TileMode,
}

/// Per-thread worker
struct Worker3D<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,
    depth: u32,

    eval_interval: ShapeTracingEval<F::IntervalEval>,

    /// Spare tape storage for reuse
    tape_storage: Vec<F::TapeStorage>,

    /// Spare shape storage for reuse
    shape_storage: Vec<F::Storage>,

    /// Workspace for shape simplification
    workspace: F::Workspace,

    /// Tiles to render using the compute shader
    result: Vec<WorkResult<F>>,
}

impl<'a, F: Function> RenderWorker<'a, F> for Worker3D<'a, F> {
    type Config = VoxelRenderConfig<'a>;
    type Output = Vec<WorkResult<F>>;
    fn new(cfg: &'a Self::Config) -> Self {
        let max_size =
            cfg.image_size.width().max(cfg.image_size.height()) as usize;
        let tile_sizes = TileSizesRef::new(&cfg.tile_sizes, max_size);
        Worker3D::<F> {
            tile_sizes,
            depth: cfg.image_size.depth(),
            eval_interval: Default::default(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
            result: vec![],
        }
    }

    fn render_tile(
        &mut self,
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        tile: RenderTile<2>,
    ) -> Self::Output {
        self.result = vec![];
        let root_tile_size = self.tile_sizes[0];
        for k in (0..self.depth.div_ceil(root_tile_size as u32)).rev() {
            let tile = RenderTile::new(Point3::new(
                tile.corner.x,
                tile.corner.y,
                k as usize * root_tile_size,
            ));
            if !self.render_tile_recurse(shape, vars, 0, tile) {
                break;
            }
        }
        std::mem::take(&mut self.result)
    }
}

impl<F: Function> Worker3D<'_, F> {
    /// Returns `true` if we should keep going, `false` otherwise
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: RenderTile<3>,
    ) -> bool {
        let tile_size = self.tile_sizes[depth];

        // Find the interval bounds of the region, in screen coordinates
        let base = Point3::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(base.z, base.z + tile_size as f32);

        // The shape applies the screen-to-model transform
        let (i, simplify) = self
            .eval_interval
            .eval_v(shape.i_tape(&mut self.tape_storage), x, y, z, vars)
            .unwrap();

        if i.upper() < 0.0 {
            self.result.push(WorkResult {
                corner: tile.corner.map(|i| i as u32),
                tile_size: tile_size as u32,
                shape: shape.inner().clone(),
                mode: TileMode::Full,
            });
            return false; // completely full
        } else if i.lower() > 0.0 {
            self.result.push(WorkResult {
                corner: tile.corner.map(|i| i as u32),
                tile_size: tile_size as u32,
                shape: shape.inner().clone(),
                mode: TileMode::Empty,
            });
            return true; // ambiguous, keep going
        }

        // Only simplify at root tiles, to keep buffer sizes down
        let sub_tape = if let Some(trace) = simplify.as_ref() {
            shape.simplify(
                trace,
                &mut self.workspace,
                &mut self.shape_storage,
                &mut self.tape_storage,
            )
        } else {
            shape
        };

        if let Some(next_tile_size) = self.tile_sizes.get(depth + 1) {
            let n = tile_size / next_tile_size;
            let mut all_done = true;
            for j in 0..n {
                for i in 0..n {
                    for k in (0..n).rev() {
                        all_done &= !self.render_tile_recurse(
                            sub_tape,
                            vars,
                            depth + 1,
                            RenderTile::new(
                                tile.corner
                                    + Vector3::new(i, j, k) * next_tile_size,
                            ),
                        );
                    }
                }
            }
            !all_done
        } else {
            // Pixel-level rendering is done on the GPU
            self.result.push(WorkResult {
                corner: tile.corner.map(|p| p as u32),
                tile_size: tile_size as u32,
                shape: sub_tape.inner().clone(),
                mode: TileMode::Voxels,
            });
            // TODO recycle things here?
            true // keep going down the stack
        }
    }
}
