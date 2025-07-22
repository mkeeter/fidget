use crate::{
    Error,
    eval::{Function, MathFunction},
    render::{
        ImageSizeLike, RenderHandle, RenderWorker, Tile as RenderTile,
        TileSizesRef, VoxelRenderConfig,
    },
    shape::{Shape, ShapeTracingEval, ShapeVars},
    types::Interval,
    wgpu::{TileContext, voxel_ray_shader},
};

use nalgebra::{Point3, Vector3};
use std::collections::HashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes};

#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct Config {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    /// Tile size to use when rendering
    tile_size: u32,

    /// Total window size
    image_size: [u32; 3],

    /// Alignment to 16-byte boundary
    _padding: u32,
}

/// Context for 3D (voxel) rendering
pub struct VoxelContext {
    /// Common context
    ctx: TileContext,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    /// Result buffer written by the compute shader
    ///
    /// Note that this buffer can't be read by the host; it must be copied to a
    /// separate buffer (typically [`TileContext::out_buf`])
    ///
    /// (dynamic size, implicit from image size in config)
    result_buf: wgpu::Buffer,
}

impl VoxelContext {
    /// Build a new 2D (pixel) rendering context
    pub fn new() -> Result<Self, Error> {
        let ctx = TileContext::new()?;

        let shader_code = voxel_ray_shader();

        // Create bind group layout and bind group
        let bind_group_layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
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
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        );

        // Create the compute pipeline
        let pipeline_layout = ctx.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            },
        );

        // Compile the shader
        let shader_module =
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });

        let pipeline = ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let result_buf = Self::new_result_buffer(&ctx.device, 512);

        Ok(Self {
            ctx,
            pipeline,
            result_buf,
            bind_group_layout,
        })
    }

    /// Builds a `result` buffer that can be written by a compute shader
    fn new_result_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
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
        let mut data = bytecode.data; // address 0 is the original tape
        pos.insert(shape.inner().id(), 0);

        let mut full = 0;
        let mut empty = 0;
        let mut voxels = 0;
        for (_, rs) in &rs {
            for r in rs.iter() {
                match r.mode {
                    TileMode::Full => full += 1,
                    TileMode::Empty => empty += 1,
                    TileMode::Voxels => voxels += 1,
                }
            }
        }
        println!("full: {full}\nempty: {empty}\nvoxels: {voxels}");

        let tile_size = tile_sizes.last() as u32;
        let nx = settings.image_size.width().div_ceil(tile_size);
        let ny = settings.image_size.height().div_ceil(tile_size);
        let nz = settings.image_size.depth().div_ceil(tile_size);

        let mut ts = vec![0u32; nx as usize * ny as usize * nz as usize];

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
            for x in 0..r.tile_size / tile_size {
                for y in 0..r.tile_size / tile_size {
                    for z in 0..r.tile_size / tile_size {
                        let i = r.corner.x / tile_size
                            + x
                            + (r.corner.y / tile_size + y) * nx
                            + (r.corner.z / tile_size + z) * nx * ny;
                        ts[i as usize] = match r.mode {
                            TileMode::Empty => u32::MAX,
                            TileMode::Full => start | (1 << 31),
                            TileMode::Voxels => start,
                        };
                    }
                }
            }
        }
        println!("bytecode len: {} KiB", data.len() * 4 / 1024);
        println!("max reg: {max_reg}");
        assert_eq!(max_mem, 0, "external memory is not yet supported");

        let vars = shape.inner().vars();
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            tile_size,
            image_size: [
                settings.image_size.width(),
                settings.image_size.height(),
                settings.image_size.depth(),
            ],
            _padding: 0,
        };
        self.ctx.load_config(&config);
        self.ctx.load_raw_tiles(&ts);
        self.ctx.resize_out_buf(settings.image_size);
        self.resize_result_buf(settings.image_size);

        self.ctx.load_tape(data.as_bytes());

        // TODO should we cache this?
        let bind_group = self.create_bind_group();

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );

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
            settings.image_size.width().div_ceil(4),
            settings.image_size.height().div_ceil(4),
            1,
        );
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        let image_pixels =
            settings.image_size.width() * settings.image_size.height();
        encoder.copy_buffer_to_buffer(
            &self.result_buf,
            0,
            &self.ctx.out_buf,
            0,
            image_pixels as u64 * std::mem::size_of::<f32>() as u64,
        );

        // Submit the commands and wait for the GPU to complete
        self.ctx.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.ctx.out_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);

        // Get the pixel-populated image
        let mut result =
            <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        self.ctx.out_buf.unmap();

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

    /// Builds a new bind group with the current buffers
    fn create_bind_group(&self) -> wgpu::BindGroup {
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ctx.config_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.ctx.tile_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.ctx.tape_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.result_buf.as_entire_binding(),
                    },
                ],
            })
    }

    /// Resizes `result_buf` to fit the given image (if necessary)
    fn resize_result_buf<I: ImageSizeLike>(&mut self, image_size: I) {
        let pixels = image_size.width() as u64 * image_size.height() as u64;
        let required_size = pixels * std::mem::size_of::<u32>() as u64;
        if required_size > self.result_buf.size() {
            self.result_buf =
                Self::new_result_buffer(&self.ctx.device, required_size);
        }
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
                tile_size: tile_size as u32,
                shape: sub_tape.inner().clone(),
                mode: TileMode::Voxels,
            });
            // TODO recycle things here?
            true // keep going down the stack
        }
    }
}
