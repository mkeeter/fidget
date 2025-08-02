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
        interval_tiles_shader, resize_buffer_with, voxel_ray_shader,
        write_storage_buffer,
    },
};

use nalgebra::{Point3, Vector3};
use std::collections::HashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes};

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

    /// Total window size
    image_size: [u32; 3],
    _padding2: u32,
}

struct IntervalTileContext {
    /// Input tiles (64^3)
    tile64_buffers: TileBuffers<64>,

    /// First stage output tiles (16^3)
    tile16_buffers: TileBuffers<16>,

    /// First stage output tiles (4^3)
    tile4_buffers: TileBuffers<4>,

    /// Compute pipeline for 16^3 tiles
    pipeline16: wgpu::ComputePipeline,

    /// Compute pipeline for 4^3 tiles
    pipeline4: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
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
                    buffer_ro(0), // config + tape data
                    buffer_ro(1), // tile64_tapes
                    buffer_ro(2), // tiles_in
                    buffer_ro(3), // tile_zmin
                    buffer_rw(4), // subtiles_out
                    buffer_rw(5), // subtile_zmin
                    buffer_rw(6), // count_clear
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

        let pipeline16 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 64.0)],
                    ..Default::default()
                },
                cache: None,
            });

        let pipeline4 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("interval_tile_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TILE_SIZE", 16.0)],
                    ..Default::default()
                },
                cache: None,
            });

        let tile64_buffers = TileBuffers::new(device);
        let tile16_buffers = TileBuffers::new(device);
        let tile4_buffers = TileBuffers::new(device);

        Ok(Self {
            bind_group_layout,
            tile64_buffers,
            tile16_buffers,
            tile4_buffers,
            pipeline16,
            pipeline4,
        })
    }

    fn run(
        &mut self,
        active_tiles: &[u32],
        tile64_zmin: &[u32],
        ctx: &CommonCtx,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.tile64_buffers.reset_with_tiles(
            ctx.device,
            ctx.queue,
            active_tiles,
            tile64_zmin,
        );
        self.tile16_buffers.reset(
            ctx.device,
            ctx.queue,
            ctx.settings.image_size,
        );
        self.tile4_buffers.reset(
            ctx.device,
            ctx.queue,
            ctx.settings.image_size,
        );

        let bind_group16 = self.create_bind_group(
            ctx,
            &self.tile64_buffers,
            &self.tile16_buffers,
            &self.tile4_buffers.tiles,
        );
        let mut compute_pass16 =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass16.set_pipeline(&self.pipeline16);
        compute_pass16.set_bind_group(0, &bind_group16, &[]);
        compute_pass16.dispatch_workgroups(active_tiles.len() as u32, 1, 1);
        drop(compute_pass16);

        let bind_group4 = self.create_bind_group(
            ctx,
            &self.tile16_buffers,
            &self.tile4_buffers,
            &self.tile64_buffers.tiles, // clear
        );
        let mut compute_pass4 =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        compute_pass4.set_pipeline(&self.pipeline4);
        compute_pass4.set_bind_group(0, &bind_group4, &[]);
        compute_pass4
            .dispatch_workgroups_indirect(&self.tile16_buffers.tiles, 0);
        drop(compute_pass4);
    }

    fn create_bind_group<const N: usize, const M: usize>(
        &self,
        ctx: &CommonCtx,
        tile_buffers: &TileBuffers<N>,
        subtile_buffers: &TileBuffers<M>,
        clear: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.buf.config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ctx.buf.tile64_tapes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_buffers.tiles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_buffers.zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: subtile_buffers.tiles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: subtile_buffers.zmin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: clear.slice(0..16).into(),
                },
            ],
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

struct VoxelContext {
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
                    buffer_ro(0), // config + tape_data
                    buffer_ro(1), // tile64_tapes
                    buffer_ro(2), // tiles4_in
                    buffer_ro(3), // tile4_zmin
                    buffer_rw(4), // result
                    buffer_rw(5), // count_clear
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

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    fn run(
        &mut self,
        ctx: &CommonCtx,
        tile4_buffers: &TileBuffers<4>,
        clear: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.buf.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.buf.tile64_tapes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile4_buffers.tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tile4_buffers.zmin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.buf.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: clear.slice(0..16).into(),
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
        compute_pass.dispatch_workgroups_indirect(&tile4_buffers.tiles, 0);
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
    device.poll(wgpu::PollType::Wait).unwrap();

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
    voxel_ctx: VoxelContext,
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
        image_size: VoxelSize,
    ) {
        let nx = image_size.width().div_ceil(64) as usize * (64 / N);
        let ny = image_size.height().div_ceil(64) as usize * (64 / N);
        let nz = image_size.depth().div_ceil(64) as usize * (64 / N);

        resize_buffer_with::<u32>(
            device,
            &mut self.tiles,
            &format!("active_tile{N}"),
            4 + nx * ny * nz,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC, // for debug
        );

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

    fn reset_with_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active_tiles: &[u32],
        tile_zmin: &[u32],
    ) {
        resize_buffer_with::<u32>(
            device,
            &mut self.tiles,
            &format!("active_tile{N}"),
            4 + active_tiles.len(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, // for debug
        );
        let mut buf = queue
            .write_buffer_with(
                &self.tiles,
                0,
                ((4 + active_tiles.len() as u64) * 4)
                    .try_into()
                    .expect("buffer size must be > 0"),
            )
            .unwrap();
        buf[0..16].copy_from_slice(
            [0u32, 0, 0, active_tiles.len() as u32].as_bytes(),
        );
        buf[16..].copy_from_slice(active_tiles.as_bytes());

        write_storage_buffer(
            device,
            queue,
            &mut self.zmin,
            &format!("tile{N}_zmin"),
            tile_zmin,
        );
    }
}

/// Buffers which must be dynamically sized
pub struct DynamicBuffers {
    config: wgpu::Buffer,

    /// Densely-packed tape indices (or special values to indicate empty / full)
    tile64_tapes: wgpu::Buffer,

    /// Result buffer written by the voxel compute shader
    ///
    /// Note that this buffer can't be read by the host; it must be copied to
    /// 'image_buf`
    ///
    /// (dynamic size, implicit from image size in config)
    result: wgpu::Buffer,
    result_scratch: Vec<u32>,

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
            config: scratch_buffer(device),
            tile64_tapes: scratch_buffer(device),
            result: scratch_buffer(device),
            image: scratch_buffer(device),
            result_scratch: vec![],
        }
    }
    fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config_data: &[u8],
        tile64_tapes: &[u32],
        image_size: VoxelSize,
    ) {
        write_storage_buffer(
            device,
            queue,
            &mut self.config,
            "config",
            config_data,
        );
        write_storage_buffer(
            device,
            queue,
            &mut self.tile64_tapes,
            "tile64_tapes",
            tile64_tapes,
        );

        let image_pixels =
            image_size.width() as usize * image_size.height() as usize;
        resize_buffer_with::<u32>(
            device,
            &mut self.result,
            "result",
            image_pixels,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        if image_pixels != self.result_scratch.len() {
            self.result_scratch.resize(image_pixels, 0u32);
        }
        queue
            .write_buffer_with(
                &self.result,
                0,
                (image_pixels as u64 * 4).try_into().unwrap(),
            )
            .unwrap()
            .copy_from_slice(self.result_scratch.as_bytes());

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
pub struct CommonCtx<'a, 'b> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,

    buf: &'a DynamicBuffers,
    settings: &'a VoxelRenderConfig<'b>,
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
                .map_err(|_| Error::NoAdapter)?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .map_err(Error::NoDevice)
        })?;

        let interval_tile_ctx = IntervalTileContext::new(&device)?;
        let voxel_ctx = VoxelContext::new(&device)?;
        let buffers = DynamicBuffers::new(&device);

        Ok(Self {
            device,
            buffers,
            queue,
            interval_tile_ctx,
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

        // CPU work
        let start = std::time::Instant::now();
        let rs = crate::render::render_tiles::<F, Worker3D<F>>(
            shape.clone(),
            &ShapeVars::new(),
            &settings,
        )?;
        println!("got tiles in {:?}", start.elapsed());

        let vars = shape.inner().vars();
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            _padding1: 0u32,
            image_size: [
                settings.image_size.width(),
                settings.image_size.height(),
                settings.image_size.depth(),
            ],
            _padding2: 0u32,
        };
        println!("{config:?}");
        let mut cfg_buf = config.as_bytes().to_vec();
        let bytecode = shape.inner().to_bytecode();
        let mut max_reg = bytecode.reg_count;
        let mut max_mem = bytecode.mem_count;

        let mut pos = HashMap::new(); // map from shape to bytecode start
        let tape_start = cfg_buf.len();
        println!("tape start at {tape_start}");
        cfg_buf.extend(bytecode.as_bytes()); // address 0 is the original tape
        pos.insert(shape.inner().id(), 0);

        // Build the dense tile array of 64^3 tiles, and active tile list
        let tile_size = tile_sizes.last() as u32;
        let nx = settings.image_size.width().div_ceil(64);
        let ny = settings.image_size.height().div_ceil(64);
        let nz = settings.image_size.depth().div_ceil(64);
        let mut tile64_tapes =
            vec![0u32; nx as usize * ny as usize * nz as usize];
        let mut tile64_zmin = vec![0u32; nx as usize * ny as usize];

        // Iterate over 64^3 tile results
        let mut active_tiles = vec![];
        let mut full = 0;
        let mut empty = 0;
        for r in rs.iter().flat_map(|(_t, r)| r.iter()) {
            let id = r.shape.inner().id();
            let start = pos.entry(id).or_insert_with(|| {
                let prev = cfg_buf.len() - tape_start;
                assert_eq!(prev % 4, 0);
                let bytecode = r.shape.inner().to_bytecode();
                max_reg = max_reg.max(bytecode.reg_count);
                max_mem = max_mem.max(bytecode.mem_count);
                cfg_buf.extend(bytecode.as_bytes());
                prev / 4
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
                        tile64_tapes[i as usize] = match r.mode {
                            TileMode::Empty => {
                                empty += 1;
                                u32::MAX
                            }
                            TileMode::Full => {
                                let i = r.corner.x / tile_size
                                    + x
                                    + (r.corner.y / tile_size + y) * nx;
                                let t = &mut tile64_zmin[i as usize];
                                *t = (*t).max(r.corner.z + z * 64 + 64);
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

        let tape_data_len = cfg_buf.len() - tape_start;
        if tape_data_len < 1024 {
            println!("bytecode len: {tape_data_len} B");
        } else {
            println!("bytecode len: {} KiB", tape_data_len / 1024);
        }
        println!("max reg: {max_reg}");
        println!(
            "full: {full}\nempty: {empty}\nvoxels: {}",
            active_tiles.len()
        );
        assert_eq!(max_mem, 0, "external memory is not yet supported");

        self.buffers.reset(
            &self.device,
            &self.queue,
            &cfg_buf,
            &tile64_tapes,
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
            settings: &settings,
        };

        self.interval_tile_ctx.run(
            &active_tiles,
            &tile64_zmin,
            &ctx,
            &mut encoder,
        );
        self.voxel_ctx.run(
            &ctx,
            &self.interval_tile_ctx.tile4_buffers,
            &self.interval_tile_ctx.tile16_buffers.tiles,
            &mut encoder,
        );

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));

        if false {
            let tile64_data = get_buffer::<u32>(
                ctx.device,
                ctx.queue,
                &self.interval_tile_ctx.tile64_buffers.tiles,
            );
            println!("tile64 size: {:?}", &tile64_data[..4]);
            let tile16_data = get_buffer::<u32>(
                ctx.device,
                ctx.queue,
                &self.interval_tile_ctx.tile16_buffers.tiles,
            );
            println!("tile16 size: {:?}", &tile16_data[..4]);
            let tile4_data = get_buffer::<u32>(
                ctx.device,
                ctx.queue,
                &self.interval_tile_ctx.tile4_buffers.tiles,
            );
            println!("tile4 size: {:?}", &tile4_data[0..4]);
            //println!("{:?}", &tile4_data[..tile4_size[0] as usize]);
        }

        // Map result buffer and read back the data
        let buffer_slice = self.buffers.image.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).unwrap();

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

        Some(Ok(result))
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
