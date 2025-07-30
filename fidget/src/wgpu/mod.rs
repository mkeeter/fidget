//! Shader generation and WGPU-based image rendering
use crate::{Error, bytecode, render::ImageSizeLike};
use zerocopy::{Immutable, IntoBytes};

use heck::ToShoutySnakeCase;

mod pixel;
pub use pixel::PixelContext;
mod voxel;
pub use voxel::VoxelContext;
mod voxel_ray;
pub use voxel_ray::VoxelRayContext;

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
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
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
    fn load_config<C: IntoBytes + Immutable>(&self, config: &C) {
        self.write_buffer_with(&self.config_buf, config.as_bytes());
    }

    /// Loads a list of tiles into `tile_buf`, resizing as needed
    fn load_tiles(&mut self, tiles: &[Tile]) {
        let required_size = std::mem::size_of_val(tiles);
        if required_size > self.tile_buf.size() as usize {
            self.tile_buf = Self::new_tile_buffer(&self.device, required_size);
        }
        self.write_buffer_with(&self.tile_buf, tiles.as_bytes());
    }

    /// Loads a tape into `tape_buf`, resizing as needed
    fn load_tape(&mut self, bytecode: &[u8]) {
        let required_size = std::mem::size_of_val(bytecode);
        if required_size > self.tape_buf.size() as usize {
            self.tape_buf = Self::new_tape_buffer(&self.device, required_size);
        }
        self.write_buffer_with(&self.tape_buf, bytecode);
    }

    fn write_buffer_with(&self, buf: &wgpu::Buffer, data: &[u8]) {
        let Ok(len) = (data.len() as u64).try_into() else {
            return;
        };
        self.queue
            .write_buffer_with(buf, 0, len)
            .unwrap()
            .copy_from_slice(data)
    }

    /// Resizes the `out` buffer to fit the given image (if necessary)
    fn resize_out_buf<I: ImageSizeLike>(&mut self, image_size: I) {
        let pixels = image_size.width() as u64 * image_size.height() as u64;
        let required_size = pixels * std::mem::size_of::<u32>() as u64;
        if required_size > self.out_buf.size() {
            self.out_buf = Self::new_out_buffer(&self.device, required_size);
        }
    }
}

pub(crate) fn write_storage_buffer<T: IntoBytes + Immutable>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &mut wgpu::Buffer,
    name: &'static str,
    data: &[T],
) {
    let size = std::mem::size_of_val(data) as u64;
    if size != buf.size() {
        println!("making new buffer with label {name} and size {size}");
        *buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    queue
        .write_buffer_with(
            buf,
            0,
            size.try_into().expect("buffer size must be > 0"),
        )
        .unwrap()
        .copy_from_slice(data.as_bytes())
}

pub(crate) fn resize_buffer<T>(
    device: &wgpu::Device,
    buf: &mut wgpu::Buffer,
    name: &'static str,
    count: usize,
) {
    resize_buffer_with::<T>(
        device,
        buf,
        name,
        count,
        // XXX remove COPY_SRC once debug is done
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )
}

pub(crate) fn resize_buffer_with<T>(
    device: &wgpu::Device,
    buf: &mut wgpu::Buffer,
    name: &'static str,
    count: usize,
    usages: wgpu::BufferUsages,
) {
    let size = (std::mem::size_of::<T>() * count) as u64;
    if size != buf.size() {
        println!("making new buffer with label {name} and size {size}");
        *buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: usages,
            mapped_at_creation: false,
        });
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

/// Returns a shader string to evaluate voxel tiles (3D)
pub fn voxel_ray_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERPRETER_1F;
    shader_code += OCCUPANCY_SHADER;
    shader_code += VOXEL_RAY_SHADER;
    shader_code
}

/// Returns a shader string to perform the interval evaluation step
pub fn interval_tiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERPRETER_I;
    shader_code += INTERVAL_TILES_SHADER;
    shader_code += OCCUPANCY_SHADER;
    shader_code
}

/// Returns a shader string to perform the interval evaluation step
pub fn interval_subtiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += INTERPRETER_I;
    shader_code += INTERVAL_SUBTILES_SHADER;
    shader_code += OCCUPANCY_SHADER;
    shader_code
}

/// Shader fragment to run a f32x4 interpreter
const INTERPRETER_4F: &str = include_str!("interpreter_4f.wgsl");

/// Shader fragment to run a f32 interpreter
const INTERPRETER_1F: &str = include_str!("interpreter_1f.wgsl");

/// Shader fragment to run a interval arithmetic interpreter
const INTERPRETER_I: &str = include_str!("interpreter_i.wgsl");

/// `main` shader function for pixel tile evaluation
const PIXEL_TILES_SHADER: &str = include_str!("pixel_tiles.wgsl");

/// `main` shader function for voxel tile evaluation
const VOXEL_TILES_SHADER: &str = include_str!("voxel_tiles.wgsl");

/// `main` shader function for voxel tile evaluation with raymarching
const VOXEL_RAY_SHADER: &str = include_str!("voxel_ray.wgsl");

/// `main` shader function for 16x16x16 tile interval evaluation
const INTERVAL_TILES_SHADER: &str = include_str!("interval_tiles.wgsl");

/// `main` shader function for 4x4x4 tile interval evaluation
const INTERVAL_SUBTILES_SHADER: &str = include_str!("interval_subtiles.wgsl");

/// `Occupancy` data type for shaders
const OCCUPANCY_SHADER: &str = include_str!("occupancy.wgsl");

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
            (voxel_ray_shader(), "voxel raymarching"),
            (interval_tiles_shader(), "interval tiles"),
            (interval_subtiles_shader(), "interval subtiles"),
        ] {
            // This isn't the best formatting, but it will at least include the
            // relevant text.
            let m = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                if let Some(i) = e.location(&src) {
                    let pos = i.offset as usize..(i.offset + i.length) as usize;
                    panic!("{}", e.emit_to_string_with_path(&src[pos], desc));
                } else {
                    panic!("{}", e.emit_to_string(desc));
                }
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
