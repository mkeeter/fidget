//! Shader generation and WGPU-based image rendering
use crate::{
    bytecode,
    eval::{Function, MathFunction},
    render::{ImageSize, View2},
    shape::{Shape, ShapeVars},
    Error,
};

use heck::ToShoutySnakeCase;
use nalgebra::Matrix4;
use zerocopy::{FromBytes, Immutable, IntoBytes};

// Square tiles
#[derive(Debug, IntoBytes, Immutable)]
#[repr(C)]
struct Tile {
    /// Corner of this tile, in pixel units
    corner: [u32; 2],

    /// Start of this tile's tape, as an offset into the global tape
    start: u32,
    // XXX do we need padding here?
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
        shape: Shape<F>,
        _vars: &ShapeVars<f32>, // XXX TODO
        image_size: ImageSize,
        view: &View2,
    ) -> Result<Vec<u32>, Error> {
        let mat = shape.transform().cloned().unwrap_or_else(Matrix4::identity);

        let world_to_model = view
            .world_to_model()
            .insert_row(2, 0.0)
            .insert_column(2, 0.0);

        let screen_to_world = image_size
            .screen_to_world()
            .insert_row(2, 0.0)
            .insert_column(2, 0.0);

        // Build the combined transform matrix
        let mat = mat * world_to_model * screen_to_world;

        let vars = shape.inner().vars();

        const TILE_SIZE: u32 = 64;
        let mut tiles = vec![]; // TODO
        for x in 0..image_size.width().div_ceil(TILE_SIZE) {
            for y in 0..image_size.height().div_ceil(TILE_SIZE) {
                tiles.push(Tile {
                    corner: [x * TILE_SIZE, y * TILE_SIZE],
                    start: 0,
                });
            }
        }
        let config = Config {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes().map(|a| vars.get(&a).unwrap_or(0) as u32),
            tile_size: TILE_SIZE,
            window_size: [image_size.width(), image_size.height()],
            tile_count: tiles.len() as u32,
        };
        self.load_config(&config);
        self.load_tiles(&tiles);
        self.resize_result_buf(image_size);

        let bytecode = shape.inner().to_bytecode();
        self.load_tape(bytecode.as_bytes());

        // TODO should we cache this?
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
            });

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

        let data = buffer_slice.get_mapped_range();
        let result = <[u32]>::ref_from_bytes(&data).unwrap();
        Ok(result.to_owned())
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
}

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
