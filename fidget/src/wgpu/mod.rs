//! Shader generation and WGPU-based image rendering
use crate::{
    bytecode,
    eval::{Function, MathFunction},
    render::{
        ImageRenderConfig, ImageSize, RenderHandle, RenderWorker,
        Tile as RenderTile, TileSizes, View2,
    },
    shape::{Shape, ShapeTracingEval, ShapeVars},
    types::Interval,
    Error,
};

use heck::ToShoutySnakeCase;
use nalgebra::{Matrix4, Point2, Vector2};
use std::collections::HashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes};

// Square tiles
#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct Tile {
    /// Corner of this tile, in pixel units
    corner: [u32; 3],

    /// Start of this tile's tape, as an offset into the global tape
    start: u32,
}

#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct Config {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    axes: [u32; 3],

    /// Tile size to use when rendering
    tile_size: u32,

    /// Total window size
    window_size: [u32; 2],

    /// Number of tiles to render
    tile_count: u32,

    /// Alignment to 16-byte boundary
    _padding: u32,
}

/// Context for rendering a set of 2D tiles
struct TileContext {
    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Configuration data (fixed size)
    config_buf: wgpu::Buffer,

    /// Tiles (flexible size, active length is specified in config)
    tile_buf: wgpu::Buffer,

    /// Tape (flexible size)
    tape_buf: wgpu::Buffer,

    /// Result buffer that can be read back from the host
    ///
    /// (dynamic size, implicit from image size in config)
    out_buf: wgpu::Buffer,
}

impl TileContext {
    /// Builds a new context, creating the WGPU device, queue, and shaders
    pub fn new() -> Result<Self, Error> {
        // Initialize wgpu
        let instance = wgpu::Instance::default();
        let (device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok_or(Error::NoAdapter)?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .map_err(Error::NoDevice)
        })?;

        // Create buffers for the input and output data
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<Config>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tape_buf = Self::new_tape_buffer(&device, 512);
        let tile_buf = Self::new_tile_buffer(&device, 512);
        let out_buf = Self::new_out_buffer(&device, 512);

        Ok(Self {
            device,
            queue,
            config_buf,
            tile_buf,
            tape_buf,
            out_buf,
        })
    }

    /// Helper function to build a tape buffer
    fn new_tape_buffer(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tape"),
            size: (len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Helper function to build a tile buffer
    fn new_tile_buffer(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile"),
            size: (len * std::mem::size_of::<Tile>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Builds an `out` buffer that can be read back by the host
    fn new_out_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Loads a config into `config_buf`
    fn load_config(&self, config: &Config) {
        self.queue
            .write_buffer(&self.config_buf, 0, config.as_bytes());
    }

    /// Loads a list of tiles into `tile_buf`, resizing as needed
    fn load_tiles(&mut self, tiles: &[Tile]) {
        let required_size = std::mem::size_of_val(tiles);
        if required_size > self.tile_buf.size() as usize {
            self.tile_buf = Self::new_tile_buffer(&self.device, required_size);
        }
        self.queue.write_buffer(&self.tile_buf, 0, tiles.as_bytes());
    }

    /// Loads a tape into `tape_buf`, resizing as needed
    fn load_tape(&mut self, bytecode: &[u8]) {
        let required_size = std::mem::size_of_val(bytecode);
        if required_size > self.tape_buf.size() as usize {
            self.tape_buf = Self::new_tape_buffer(&self.device, required_size);
        }
        self.queue.write_buffer(&self.tape_buf, 0, bytecode);
    }

    /// Resizes the `out` buffer to fit the given image (if necessary)
    fn resize_out_buf(&mut self, image_size: ImageSize) {
        let pixels = image_size.width() as u64 * image_size.height() as u64;
        let required_size = pixels * std::mem::size_of::<u32>() as u64;
        if required_size > self.out_buf.size() {
            self.out_buf = Self::new_out_buffer(&self.device, required_size);
        }
    }
}

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
    ) -> Result<Vec<u32>, Error> {
        // Convert to a 4x4 matrix and apply to the shape
        let mat = settings.mat();
        let mat = mat.insert_row(2, 0.0);
        let mat = mat.insert_column(2, 0.0);
        let shape = shape.apply_transform(mat);

        let rs = crate::render::render_tiles::<F, Worker2D<F>>(
            shape.clone(),
            &ShapeVars::new(),
            &settings,
        );

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
            assert_eq!(r.tile_size as usize, settings.tile_sizes.last());
            tiles.push(Tile {
                corner: [r.corner.x, r.corner.y, 0],
                start: u32::try_from(*start).unwrap_or(0),
            })
        }
        assert_eq!(max_mem, 0, "external memory is not yet supported");

        let tile_size = settings.tile_sizes.last() as u32;

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
        self.ctx.device.poll(wgpu::Maintain::Wait);

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
                let o =
                    r.corner.x + (r.corner.y + j) * settings.image_size.width();
                result[o as usize..][..r.tile_size as usize].fill(fill);
            }
        }

        Ok(result)
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
        let mat = shape.transform().cloned().unwrap_or_else(Matrix4::identity)
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
            axes: shape.axes().map(|a| vars.get(&a).unwrap_or(0) as u32),
            tile_size: TILE_SIZE,
            window_size: [image_size.width(), image_size.height()],
            tile_count: tiles.len() as u32,
            _padding: 0,
        };
        self.ctx.load_config(&config);
        self.ctx.load_tiles(&tiles);
        self.ctx.resize_out_buf(image_size);

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
        encoder.copy_buffer_to_buffer(
            &self.result_buf,
            0,
            &self.ctx.out_buf,
            0,
            (pixel_count * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands and wait for the GPU to complete
        self.ctx.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.ctx.out_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);

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
    tile_sizes: &'a TileSizes,

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
        Worker2D::<F> {
            tile_sizes: &cfg.tile_sizes,
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
                shape: shape.shape(),
                mode: TileMode::Full,
            });
            return;
        } else if i.lower() > 0.0 {
            self.result.push(WorkResult {
                corner: tile.corner.map(|i| i as u32),
                tile_size: tile_size as u32,
                shape: shape.shape(),
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
                shape: sub_tape.shape(),
                mode: TileMode::Pixels,
            })
            // TODO recycle things here?
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Returns a set of constant definitions for each opcode
fn opcode_constants() -> String {
    let mut out = String::new();
    for (op, i) in bytecode::iter_ops() {
        out += &format!("const OP_{}: u32 = {i};\n", op.to_shouty_snake_case());
    }
    out
}

/// Returns a shader string to evaluate pixel tiles (2D)
pub fn pixel_tiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERPRETER_4F;
    shader_code += COMMON_SHADER;
    shader_code += PIXEL_TILES_SHADER;
    shader_code
}

/// Returns a shader string to evaluate voxel tiles (3D)
pub fn voxel_tiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERPRETER_4F;
    shader_code += COMMON_SHADER;
    shader_code += VOXEL_TILES_SHADER;
    shader_code
}

/// Shader fragment to run a f32x4 interpreter
const INTERPRETER_4F: &str = include_str!("interpreter_4f.wgsl");

/// `main` shader function for pixel tile evaluation
const PIXEL_TILES_SHADER: &str = include_str!("pixel_tiles.wgsl");

/// `main` shader function for pixel tile evaluation
const VOXEL_TILES_SHADER: &str = include_str!("voxel_tiles.wgsl");

/// Common data types for shaders
const COMMON_SHADER: &str = include_str!("common.wgsl");

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn interpreter_4f_has_all_ops() {
        for (op, _) in bytecode::iter_ops() {
            let op = format!("OP_{}", op.to_shouty_snake_case());
            assert!(
                INTERPRETER_4F.contains(&op),
                "interpreter is missing {op}"
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
            (pixel_tiles_shader(), "pixel tiles"),
            (voxel_tiles_shader(), "voxel tiles"),
        ] {
            // This isn't the best formatting, but it will at least include the
            // relevant text.
            let m = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                let i = e.location(&src).unwrap();
                let pos = i.offset as usize..(i.offset + i.length) as usize;
                panic!("{}", e.emit_to_string_with_path(&src[pos], desc));
            });
            if let Err(e) = v.validate(&m) {
                let (pos, desc) = e.spans().next().unwrap();
                panic!(
                    "{}",
                    e.emit_to_string_with_path(
                        &src[pos.to_range().unwrap()],
                        desc
                    )
                );
            }
        }
    }
}
