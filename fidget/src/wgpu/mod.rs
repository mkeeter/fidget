//! Shader generation and WGPU-based image rendering
use crate::{
    bytecode,
    eval::{Function, MathFunction, TracingEvaluator},
    render::{ImageRenderConfig, ImageSize, View2},
    shape::{Shape, ShapeTape, ShapeTracingEval},
    types::Interval,
    Error,
};

use heck::ToShoutySnakeCase;
use nalgebra::{Matrix4, Point2, Vector2};
use zerocopy::{FromBytes, Immutable, IntoBytes};

// Square tiles
#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct Tile {
    /// Corner of this tile, in pixel units
    corner: [u32; 2],

    /// Start of this tile's tape, as an offset into the global tape
    start: u32,

    /// Alignment to 16-byte boundary
    _padding: u32,
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
pub struct TileContext {
    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    /// Configuration data (fixed size)
    config_buf: wgpu::Buffer,

    /// Tiles (flexible size, active length is specified in config)
    tile_buf: wgpu::Buffer,

    /// Tape (flexible size)
    tape_buf: wgpu::Buffer,

    /// Result buffer written by the compute shader
    ///
    /// (dynamic size, implicit from image size in config)
    result_buf: wgpu::Buffer,

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

        let shader_code = pixel_tiles_shader();

        // Compile the shader
        let shader_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        // Create buffers for the input and output data
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: std::mem::size_of::<Config>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tape_buf = Self::new_tape_buffer(&device, 512);
        let tile_buf = Self::new_tile_buffer(&device, 512);
        let (result_buf, out_buf) = Self::new_result_buffers(&device, 512);

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

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            queue,
            pipeline,
            config_buf,
            tile_buf,
            tape_buf,
            result_buf,
            out_buf,
            bind_group_layout,
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

    /// Builds a tuple of `(result, out)` buffers with a given size (in bytes)
    ///
    /// The `result` buffer can be written by a compute shader; the `out` buffer
    /// can be read back by the host.
    fn new_result_buffers(
        device: &wgpu::Device,
        size: u64,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (result_buffer, out_buffer)
    }

    /// Renders a single image using GPU acceleration
    pub fn run<F: Function + MathFunction>(
        &mut self,
        shape: Shape<F>, // XXX add ShapeVars here
        settings: ImageRenderConfig,
    ) -> Result<Vec<u32>, Error> {
        let world_to_model = settings
            .view
            .world_to_model()
            .insert_row(2, 0.0)
            .insert_column(2, 0.0);

        let screen_to_world = settings
            .image_size
            .screen_to_world()
            .insert_row(2, 0.0)
            .insert_column(2, 0.0);

        // Build the combined transform matrix
        let mat = shape.transform().cloned().unwrap_or_else(Matrix4::identity)
            * world_to_model
            * screen_to_world;

        // Clear out the previous transform
        let shape = shape.set_transform(mat);

        const TILE_SIZE: u32 = 64;

        let tiles = build_tiles_in_parallel(&shape, &settings);

        let vars = shape.inner().vars();
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes().map(|a| vars.get(&a).unwrap_or(0) as u32),
            tile_size: TILE_SIZE,
            window_size: [
                settings.image_size.width(),
                settings.image_size.height(),
            ],
            tile_count: tiles.len() as u32,
            _padding: 0,
        };
        self.load_config(&config);
        self.load_tiles(&tiles);
        self.resize_result_buf(settings.image_size);

        let bytecode = shape.inner().to_bytecode();
        self.load_tape(bytecode.as_bytes());

        // TODO should we cache this?
        let bind_group = self.create_bind_group();

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
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
            &self.out_buf,
            0,
            (pixel_count * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.out_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result = <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
            .unwrap()
            .to_owned();
        self.out_buf.unmap();

        Ok(result)
    }

    /// Renders a single image using GPU acceleration
    pub fn run_brute<F: Function + MathFunction>(
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
                    corner: [x * TILE_SIZE, y * TILE_SIZE],
                    start: 0,
                    _padding: 0,
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
        self.load_config(&config);
        self.load_tiles(&tiles);
        self.resize_result_buf(image_size);

        let bytecode = shape.inner().to_bytecode();
        self.load_tape(bytecode.as_bytes());

        // TODO should we cache this?
        let bind_group = self.create_bind_group();

        // Create a command encoder and dispatch the compute work
        let mut encoder = self.device.create_command_encoder(
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
            &self.out_buf,
            0,
            (pixel_count * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands and wait for the GPU to complete
        self.queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = self.out_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result = <[u32]>::ref_from_bytes(&buffer_slice.get_mapped_range())
            .unwrap()
            .to_owned();
        self.out_buf.unmap();

        Ok(result)
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

    /// Resizes the result buffer to fit the given image (if necessary)
    fn resize_result_buf(&mut self, image_size: ImageSize) {
        let pixels = image_size.width() as u64 * image_size.height() as u64;
        let required_size = pixels * std::mem::size_of::<u32>() as u64;
        if required_size > self.result_buf.size() {
            let (result_buf, out_buf) =
                Self::new_result_buffers(&self.device, required_size);
            self.result_buf = result_buf;
            self.out_buf = out_buf;
        }
    }

    /// Builds a new bind group with the current buffers
    fn create_bind_group(&self) -> wgpu::BindGroup {
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
                    resource: self.tile_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.result_buf.as_entire_binding(),
                },
            ],
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Unit of work passed to worker threads
struct Task<F: Function> {
    corner: Point2<u32>,

    /// Index into the worker's tile size array
    depth: usize,

    /// Shape to render
    shape: Shape<F>,

    /// Precomputed interval tape (if available)
    tape: Option<ShapeTape<<F::IntervalEval as TracingEvaluator>::Tape>>,
}

enum TaskOrStop<F: Function> {
    Task(Task<F>),
    Stop,
}

enum WorkResult<F> {
    Done {
        corner: Point2<u32>,
        depth: usize,
        full: bool,
    },
    Pixels {
        corner: Point2<u32>,
        shape: Shape<F>,
    },
}

struct Worker<F: Function> {
    shape_storage: Vec<F::Storage>,
    tape_storage: Vec<F::TapeStorage>,
    workspace: F::Workspace,
}

impl<F: Function> Default for Worker<F> {
    fn default() -> Self {
        Self {
            shape_storage: Default::default(),
            tape_storage: Default::default(),
            workspace: Default::default(),
        }
    }
}

fn build_tiles_in_parallel<F: Function>(
    s: &Shape<F>,
    cfg: &ImageRenderConfig,
) -> Vec<Tile> {
    let tile_size = cfg.tile_sizes.get(0).unwrap() as u32;

    use thread_local::ThreadLocal;
    let tls: ThreadLocal<std::cell::RefCell<Worker<F>>> = ThreadLocal::new();
    let eval: ThreadLocal<
        std::cell::RefCell<ShapeTracingEval<F::IntervalEval>>,
    > = ThreadLocal::new();

    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .expect("failed to build thread pool");
    let (tx, rx) = std::sync::mpsc::channel();

    let mut tiles = vec![]; // TODO
    for x in 0..cfg.image_size.width().div_ceil(tile_size) {
        for y in 0..cfg.image_size.height().div_ceil(tile_size) {
            tiles.push(Task {
                corner: Point2::new(x * tile_size, y * tile_size),
                depth: 0,
                shape: s.clone(),
                tape: None,
            });
        }
    }

    let run_one = |mut t: Task<F>| -> Vec<Task<F>> {
        let tile_size = cfg.tile_sizes[t.depth] as u32;
        let x =
            Interval::new(t.corner.x as f32, (t.corner.x + tile_size) as f32);
        let y =
            Interval::new(t.corner.y as f32, (t.corner.y + tile_size) as f32);
        let z = Interval::new(0.0, 0.0);

        let mut worker = tls.get_or_default().borrow_mut();
        let mut eval = eval.get_or_default().borrow_mut();

        let tape = t.tape.get_or_insert_with(|| {
            t.shape
                .interval_tape(worker.tape_storage.pop().unwrap_or_default())
        });
        let (r, trace) = eval.eval(tape, x, y, z).unwrap();
        let has_trace = trace.is_some();

        // Early return if this region is empty / full
        if r.lower() > 0.0 || r.upper() < 0.0 {
            worker.tape_storage.extend(t.tape.take().unwrap().recycle());
            worker.shape_storage.extend(t.shape.recycle());
            tx.send(WorkResult::Done {
                corner: t.corner,
                depth: t.depth,
                full: r.upper() < 0.0,
            })
            .unwrap();
            return vec![];
        }

        let next = if let Some(trace) = trace {
            t.shape
                .simplify(
                    trace,
                    worker.shape_storage.pop().unwrap_or_default(),
                    &mut worker.workspace,
                )
                .unwrap()
        } else {
            t.shape
        };

        if let Some(next_tile_size) =
            cfg.tile_sizes.get(t.depth + 1).map(|t| t as u32)
        {
            let n = tile_size / next_tile_size;
            let mut out = Vec::with_capacity((n as usize).pow(2));
            for j in 0..n {
                for i in 0..n {
                    out.push(Task {
                        corner: t.corner + Vector2::new(i, j) * next_tile_size,
                        depth: t.depth + 1,
                        shape: next.clone(),
                        tape: if has_trace { t.tape.clone() } else { None },
                    });
                }
            }
            out
        } else {
            // We're done recursing, push the pixel work to the GPU
            worker.tape_storage.extend(t.tape.take().unwrap().recycle());
            tx.send(WorkResult::Pixels {
                corner: t.corner,
                shape: next,
            })
            .unwrap();
            vec![]
        }
    };

    pool.install(|| {
        for t in tiles {
            for out in run_one(t) {
                pool.install(|| run_one(out)); // XXX more recursion?
            }
        }
    });

    todo!()
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
    shader_code += PIXEL_TILES_SHADER;
    shader_code
}

/// Shader fragment to run a f32x4 interpreter
const INTERPRETER_4F: &str = include_str!("interpreter_4f.wgsl");

/// `main` shader function for pixel tile evaluation
const PIXEL_TILES_SHADER: &str = include_str!("pixel_tiles.wgsl");

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
}