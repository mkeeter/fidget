use crate::{
    Error,
    eval::{Function, MathFunction},
    render::{
        ImageSizeLike, RenderHandle, RenderWorker, Tile as RenderTile,
        TileSizesRef, VoxelRenderConfig,
    },
    shape::{Shape, ShapeTracingEval, ShapeVars},
    types::Interval,
    wgpu::{interval_tiles_shader, voxel_ray_shader, write_storage_buffer},
};

use nalgebra::{Point3, Vector3};
use std::collections::HashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes};

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct IntervalConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    /// Number of tiles in `active_tiles`
    tile_count: u32,

    /// Total window size
    image_size_tiles: [u32; 3],

    /// Alignment to 16-byte boundary
    _padding: u32,
}

struct IntervalTileContext {
    /// Configuration data (fixed size)
    config_buf: wgpu::Buffer,

    /// Active tile list (flexibly sized, with size in config)
    active_tiles: wgpu::Buffer,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IntervalTileContext {
    fn new(device: &wgpu::Device) -> Result<Self, Error> {
        let shader_code = interval_tiles_shader();

        // Create bind group layout and bind group
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<VoxelConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let active_tiles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("active_tiles"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            config_buf,
            active_tiles,
        })
    }

    fn run<F: Function + MathFunction>(
        &mut self,
        active_tiles: &[u32],
        ctx: &CommonCtx<F>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let vars = ctx.shape.inner().vars();
        let config = IntervalConfig {
            mat: ctx.mat.data.as_slice().try_into().unwrap(),
            axes: ctx
                .shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            tile_count: active_tiles.len().try_into().unwrap(),
            image_size_tiles: [
                ctx.settings.image_size.width().div_ceil(64),
                ctx.settings.image_size.height().div_ceil(64),
                ctx.settings.image_size.depth().div_ceil(64),
            ],
            _padding: 0,
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

        // Write active tiles to our buffer
        write_storage_buffer(
            ctx.device,
            ctx.queue,
            &mut self.active_tiles,
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
                        resource: ctx.dense_tile64.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.active_tiles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.dense_tile8_out.as_entire_binding(),
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
            8,
        );
        println!(
            "dispatching with {}, {}, {}",
            (active_tiles.len() % 65536) as u32,
            (active_tiles.len() / 65536).max(1),
            8
        );
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

    /// Result buffer written by the compute shader
    ///
    /// Note that this buffer can't be read by the host; it must be copied to
    /// 'image_buf`
    ///
    /// (dynamic size, implicit from image size in config)
    result_buf: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    ///
    /// (dynamic size, implicit from image size in config)
    image_buf: wgpu::Buffer,

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
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Dummy buffers
        let (result_buf, image_buf) = Self::new_result_buffers(device, 4);

        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<VoxelConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            config_buf,
            result_buf,
            image_buf,
            bind_group_layout,
        })
    }

    /// Resizes `result_buf` to fit the given image (if necessary)
    fn resize_image_buf<I: ImageSizeLike>(
        &mut self,
        device: &wgpu::Device,
        image_size: I,
    ) {
        let pixels = image_size.width() as u64 * image_size.height() as u64;
        let required_size = pixels * std::mem::size_of::<u32>() as u64;
        if required_size != self.image_buf.size() {
            let (r, i) = Self::new_result_buffers(device, required_size);
            self.result_buf = r;
            self.image_buf = i;
        }
    }

    fn new_result_buffers(
        device: &wgpu::Device,
        size: u64,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (r, i)
    }

    fn run<F: Function>(
        &mut self,
        ctx: &CommonCtx<F>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Resize internal buffers
        self.resize_image_buf(ctx.device, ctx.settings.image_size);

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
                        resource: ctx.dense_tile64.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.dense_tile8_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.result_buf.as_entire_binding(),
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
        let image_pixels =
            ctx.settings.image_size.width() * ctx.settings.image_size.height();
        encoder.copy_buffer_to_buffer(
            &self.result_buf,
            0,
            &self.image_buf,
            0,
            image_pixels as u64 * std::mem::size_of::<f32>() as u64,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Context for 3D (voxel) rendering
pub struct VoxelRayContext {
    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Tape buffer
    tape_buf: wgpu::Buffer,

    /// 64x64x64 tiles data (densely packed), used by both pipelines
    dense_tile64: wgpu::Buffer,

    /// 8x8x8 tiles output (densely packed)
    ///
    /// This output is written by the interval pass and used by the voxel pass
    dense_tile8_out: wgpu::Buffer,

    interval_ctx: IntervalTileContext,
    voxel_ctx: VoxelContext,
}

#[derive(Copy, Clone)]
pub struct CommonCtx<'a, 'b, F> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    tape_buf: &'a wgpu::Buffer,
    dense_tile64: &'a wgpu::Buffer,
    dense_tile8_out: &'a wgpu::Buffer,
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

        let interval_ctx = IntervalTileContext::new(&device)?;
        let voxel_ctx = VoxelContext::new(&device)?;

        // Dummy buffers
        let tape_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy tape buf"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dense_tile64 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy dense tile64"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dense_tile8_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy dense tile8 out"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            interval_ctx,
            voxel_ctx,
            tape_buf,
            dense_tile64,
            dense_tile8_out,
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
        if settings.tile_sizes.iter().collect::<Vec<_>>() != [&64] {
            panic!("Voxel raycast only uses 64x64x64 tile rendering");
        };

        // Convert to a 4x4 matrix and apply to the shape
        let shape = shape.apply_transform(settings.mat());
        let mat = shape.transform().unwrap();

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
        let mut data = bytecode.data; // address 0 is the original tape
        pos.insert(shape.inner().id(), 0);

        // Build the dense tile array of 64^3 tiles, and active tile list
        let nx = settings.image_size.width().div_ceil(64);
        let ny = settings.image_size.height().div_ceil(64);
        let nz = settings.image_size.depth().div_ceil(64);
        let mut ts = vec![0u32; nx as usize * ny as usize * nz as usize];

        // Iterate over 64^3 tile results
        let mut active_tiles = vec![];
        let mut full = 0;
        let mut empty = 0;
        for r in rs.iter().flat_map(|(_t, r)| r.iter()) {
            let id = r.shape.inner().id();
            let start = pos.entry(id).or_insert_with(|| {
                let prev = data.len();
                let bytecode = r.shape.inner().to_bytecode();
                max_reg = max_reg.max(bytecode.reg_count);
                max_mem = max_mem.max(bytecode.mem_count);
                data.extend(bytecode.data.into_iter());
                prev
            });
            let start = *start as u32;
            assert_eq!(r.corner.x % 64, 0);
            assert_eq!(r.corner.y % 64, 0);
            assert_eq!(r.corner.z % 64, 0);
            let i = r.corner.x / 64
                + r.corner.y / 64 * nx
                + r.corner.z / 64 * nx * ny;
            ts[i as usize] = match r.mode {
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

        if data.len() < 1024 {
            println!("bytecode len: {} B", data.len() * 4);
        } else {
            println!("bytecode len: {} KiB", data.len() * 4 / 1024);
        }
        println!("max reg: {max_reg}");
        println!(
            "full: {full}\nempty: {empty}\nvoxels: {}",
            active_tiles.len()
        );
        assert_eq!(max_mem, 0, "external memory is not yet supported");

        // Prepare our common buffers with tape and tile data
        write_storage_buffer(
            &self.device,
            &self.queue,
            &mut self.tape_buf,
            "tape",
            data.as_slice(),
        );
        write_storage_buffer(
            &self.device,
            &self.queue,
            &mut self.dense_tile64,
            "dense tile64",
            ts.as_slice(),
        );

        // Allocate a buffer for the densely-packed 8^3 subtiles
        let dense_tile8_count =
            nx as usize * ny as usize * ny as usize * 8usize.pow(3);
        let dense_tile8_buf_size =
            (dense_tile8_count * std::mem::size_of::<u32>()) as u64;
        if self.dense_tile8_out.size() != dense_tile8_buf_size {
            self.dense_tile8_out =
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dense til8"),
                    size: dense_tile8_buf_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
        }

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

        let ctx = CommonCtx {
            device: &self.device,
            queue: &self.queue,
            tape_buf: &self.tape_buf,
            dense_tile64: &self.dense_tile64,
            dense_tile8_out: &self.dense_tile8_out,
            mat,
            shape: &shape,
            settings: &settings,
        };

        self.interval_ctx.run(&active_tiles, &ctx, &mut encoder);
        self.voxel_ctx.run(&ctx, &mut encoder);

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.voxel_ctx.image_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        // Get the pixel-populated image
        let mut result =
            <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        self.voxel_ctx.image_buf.unmap();

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
                shape: shape.inner().clone(),
                mode: TileMode::Full,
            });
            return false; // completely full
        } else if i.lower() > 0.0 {
            self.result.push(WorkResult {
                corner: tile.corner.map(|i| i as u32),
                shape: shape.inner().clone(),
                mode: TileMode::Empty,
            });
            return true; // ambiguous, keep going
        }

        // Only simplify at root tiles, to keep buffer sizes down
        let sub_tape = if depth == 0
            && let Some(trace) = simplify.as_ref()
        {
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
                shape: sub_tape.inner().clone(),
                mode: TileMode::Voxels,
            });
            // TODO recycle things here?
            true // keep going down the stack
        }
    }
}
