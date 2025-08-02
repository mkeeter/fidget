//! Shader generation and WGPU-based image rendering
use crate::bytecode;
use zerocopy::{Immutable, IntoBytes};

use heck::ToShoutySnakeCase;

mod voxel_ray;
pub use voxel_ray::VoxelRayContext;

pub(crate) fn write_storage_buffer<T: IntoBytes + Immutable>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &mut wgpu::Buffer,
    name: &str,
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

pub(crate) fn resize_buffer_with<T>(
    device: &wgpu::Device,
    buf: &mut wgpu::Buffer,
    name: &str,
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

/// Returns a shader string to evaluate voxel tiles (3D)
pub fn voxel_ray_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += COMMON_RAY;
    shader_code += VOXEL_RAY_SHADER;
    shader_code
}

/// Returns a shader string to perform the interval evaluation step
pub fn interval_tiles_shader() -> String {
    let mut shader_code = opcode_constants();
    shader_code += COMMON_RAY;
    shader_code += INTERVAL_TILES_SHADER;
    shader_code
}

/// `main` shader function for voxel tile evaluation with raymarching
const VOXEL_RAY_SHADER: &str = include_str!("voxel_ray.wgsl");

/// `main` shader function for 16x16x16 tile interval evaluation
const INTERVAL_TILES_SHADER: &str = include_str!("interval_tiles.wgsl");

/// Common data types for shaders
const COMMON_RAY: &str = include_str!("common_ray.wgsl");

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn shader_has_all_ops() {
        for (op, _) in bytecode::iter_ops() {
            let op = format!("OP_{}", op.to_shouty_snake_case());
            assert!(
                VOXEL_RAY_SHADER.contains(&op),
                "interpreter is missing {op}"
            );
            assert!(
                INTERVAL_TILES_SHADER.contains(&op),
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
            (interval_tiles_shader(), "interval tiles"),
            (voxel_ray_shader(), "voxel raymarching"),
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
