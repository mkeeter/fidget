use crate::{
    Error,
    eval::{Function, MathFunction},
    render::{
        ImageRenderConfig, ImageSize, RenderHandle, RenderWorker,
        Tile as RenderTile, TileSizesRef, View2,
    },
    shape::{Shape, ShapeTracingEval, ShapeVars},
    types::Interval,
    wgpu::{Config, Tile, TileContext, pixel_tiles_shader},
};

use nalgebra::{Matrix4, Point2, Vector2};
use std::collections::HashMap;
use zerocopy::{FromBytes, IntoBytes};

/// Context for 2D (pixel) rendering
pub struct PixelContext {
    /// Common context
    ctx: TileContext,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    /// Result buffer written by the compute shader
    ///
    /// (dynamic size, implicit from image size in config)
    result_buf: wgpu::Buffer,
}

impl PixelContext {
    /// Build a new 2D (pixel) rendering context
    pub fn new() -> Result<Self, Error> {
        let ctx = TileContext::new()?;

        let shader_code = pixel_tiles_shader();

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
    pub fn run_2d<F: Function + MathFunction>(
        &mut self,
        shape: Shape<F>, // XXX add ShapeVars here
        settings: ImageRenderConfig,
    ) -> Option<Result<Vec<u32>, Error>> {
        // Convert to a 4x4 matrix and apply to the shape
        let mat = settings.mat();
        let mat = mat.insert_row(2, 0.0);
        let mat = mat.insert_column(2, 0.0);
        let shape = shape.apply_transform(mat);

        let max_size = settings
            .image_size
            .width()
            .max(settings.image_size.height()) as usize;
        let tile_sizes = TileSizesRef::new(&settings.tile_sizes, max_size);

        let rs = crate::render::render_tiles::<F, Worker2D<F>>(
            shape.clone(),
            &ShapeVars::new(),
            &settings,
        )?;

        let bytecode = shape.inner().to_bytecode();
        let mut max_reg = bytecode.reg_count;
        let mut max_mem = bytecode.mem_count;

        let mut pos = HashMap::new(); // map from shape to bytecode start
        let mut data = bytecode.data; // address 0 is the original tape
        pos.insert(shape.inner().id(), 0);

        let mut tiles = vec![];
        for r in rs
            .iter()
            .flat_map(|(_t, r)| r.iter())
            .filter(|r| matches!(r.mode, TileMode::Pixels))
        {
            let id = r.shape.inner().id();
            let start = pos.entry(id).or_insert_with(|| {
                let prev = data.len();
                let bytecode = r.shape.inner().to_bytecode();
                max_reg = max_reg.max(bytecode.reg_count);
                max_mem = max_mem.max(bytecode.mem_count);
                data.extend(bytecode.data.into_iter());
                prev
            });
            assert_eq!(r.tile_size as usize, tile_sizes.last());
            tiles.push(Tile {
                corner: [r.corner.x, r.corner.y, 0],
                start: u32::try_from(*start).unwrap_or(0),
            })
        }
        assert_eq!(max_mem, 0, "external memory is not yet supported");

        let tile_size = tile_sizes.last() as u32;

        let vars = shape.inner().vars();
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes().map(|a| vars.get(&a).unwrap_or(0) as u32),
            tile_size,
            window_size: [
                settings.image_size.width(),
                settings.image_size.height(),
            ],
            tile_count: tiles.len() as u32,
            _padding: 0,
        };
        self.ctx.load_config(&config);
        self.ctx.load_tiles(&tiles);
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

        // total work = tile pixels / (workgroup size * SIMD width)
        let dispatch_size =
            ((tile_size as usize).pow(2) * tiles.len()).div_ceil(64 * 4) as u32;
        compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
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
        self.ctx.device.poll(wgpu::PollType::Wait).unwrap();

        let mut result =
            <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
                .unwrap()
                .to_owned();
        self.ctx.out_buf.unmap();

        // Fill in our filled types
        for r in rs.iter().flat_map(|(_t, r)| r.iter()) {
            let fill = match r.mode {
                TileMode::Full => 0xFFAAAAAA,
                TileMode::Empty => 0xFF222222,
                TileMode::Pixels => continue,
            };
            for j in 0..r.tile_size {
                if r.corner.y + j < settings.image_size.height() {
                    let o = (r.corner.y + j) * settings.image_size.width();
                    for i in 0..r.tile_size {
                        if r.corner.x + i < settings.image_size.width() {
                            result[(o + r.corner.x + i) as usize] = fill;
                        }
                    }
                }
            }
        }

        Some(Ok(result))
    }

    /// Renders a single image using GPU acceleration
    pub fn run_2d_brute<F: Function + MathFunction>(
        &mut self,
        shape: Shape<F>, // XXX add ShapeVars here
        image_size: ImageSize,
        view: View2,
    ) -> Result<Vec<u32>, Error> {
        let world_to_model = view
            .world_to_model()
            .insert_row(2, 0.0)
            .insert_column(2, 0.0);

        let screen_to_world = image_size
            .screen_to_world()
            .insert_row(2, 0.0)
            .insert_column(2, 0.0);

        // Build the combined transform matrix
        let mat = shape.transform().unwrap_or_else(Matrix4::identity)
            * world_to_model
            * screen_to_world;

        const TILE_SIZE: u32 = 64;
        let mut tiles = vec![]; // TODO
        for x in 0..image_size.width().div_ceil(TILE_SIZE) {
            for y in 0..image_size.height().div_ceil(TILE_SIZE) {
                tiles.push(Tile {
                    corner: [x * TILE_SIZE, y * TILE_SIZE, 0],
                    start: 0,
                });
            }
        }
        let vars = shape.inner().vars();
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape
                .axes()
                .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX)),
            tile_size: TILE_SIZE,
            window_size: [image_size.width(), image_size.height()],
            tile_count: tiles.len() as u32,
            _padding: 0,
        };
        self.ctx.load_config(&config);
        self.ctx.load_tiles(&tiles);
        self.ctx.resize_out_buf(image_size);
        self.resize_result_buf(image_size);

        let bytecode = shape.inner().to_bytecode();
        self.ctx.load_tape(bytecode.as_bytes());

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

        // total work = tile pixels / (workgroup size * SIMD width)
        let pixel_count = (TILE_SIZE as usize).pow(2) * tiles.len();
        let dispatch_size = pixel_count.div_ceil(64 * 4) as u32;
        compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        drop(compute_pass);

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        let image_pixels = image_size.width() * image_size.height();
        encoder.copy_buffer_to_buffer(
            &self.result_buf,
            0,
            &self.ctx.out_buf,
            0,
            (image_pixels as usize * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands and wait for the GPU to complete
        self.ctx.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.ctx.out_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::Wait).unwrap();

        let result = <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
            .unwrap()
            .to_owned();
        self.ctx.out_buf.unmap();

        Ok(result)
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

    /// Resizes the `out` buffer to fit the given image (if necessary)
    fn resize_result_buf(&mut self, image_size: ImageSize) {
        let pixels = image_size.width() as u64 * image_size.height() as u64;
        let required_size = pixels * std::mem::size_of::<u32>() as u64;
        if required_size > self.result_buf.size() {
            self.result_buf =
                Self::new_result_buffer(&self.ctx.device, required_size);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

enum TileMode {
    Full,
    Empty,
    Pixels,
}

struct WorkResult<F> {
    /// Tile corner (in image space)
    corner: Point2<u32>,
    /// Tile size, in pixels
    tile_size: u32,
    /// Shape at this tile
    shape: Shape<F>,
    /// Fill or render individual pixels
    mode: TileMode,
}

/// Per-thread worker
struct Worker2D<'a, F: Function> {
    tile_sizes: TileSizesRef<'a>,

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

impl<'a, F: Function> RenderWorker<'a, F> for Worker2D<'a, F> {
    type Config = ImageRenderConfig<'a>;
    type Output = Vec<WorkResult<F>>;
    fn new(cfg: &'a Self::Config) -> Self {
        let max_size =
            cfg.image_size.width().max(cfg.image_size.height()) as usize;
        let tile_sizes = TileSizesRef::new(&cfg.tile_sizes, max_size);
        Worker2D::<F> {
            tile_sizes,
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
        self.render_tile_recurse(shape, vars, 0, tile);
        std::mem::take(&mut self.result)
    }
}

impl<F: Function> Worker2D<'_, F> {
    fn render_tile_recurse(
        &mut self,
        shape: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        depth: usize,
        tile: RenderTile<2>,
    ) {
        let tile_size = self.tile_sizes[depth];

        // Find the interval bounds of the region, in screen coordinates
        let base = Point2::from(tile.corner).cast::<f32>();
        let x = Interval::new(base.x, base.x + tile_size as f32);
        let y = Interval::new(base.y, base.y + tile_size as f32);
        let z = Interval::new(0.0, 0.0);

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
            return;
        } else if i.lower() > 0.0 {
            self.result.push(WorkResult {
                corner: tile.corner.map(|i| i as u32),
                tile_size: tile_size as u32,
                shape: shape.inner().clone(),
                mode: TileMode::Empty,
            });
            return;
        }

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
            for j in 0..n {
                for i in 0..n {
                    self.render_tile_recurse(
                        sub_tape,
                        vars,
                        depth + 1,
                        RenderTile::new(
                            tile.corner + Vector2::new(i, j) * next_tile_size,
                        ),
                    );
                }
            }
        } else {
            self.result.push(WorkResult {
                corner: tile.corner.map(|p| p as u32),
                tile_size: tile_size as u32,
                shape: sub_tape.inner().clone(),
                mode: TileMode::Pixels,
            })
            // TODO recycle things here?
        }
    }
}
