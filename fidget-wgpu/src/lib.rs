//! Shader generation and WGPU-based image rendering
#![warn(missing_docs)]

use heck::ToShoutySnakeCase;

pub mod buf;
pub mod effects;
pub mod voxel;

/// Re-export the `wgpu` module
pub use wgpu;

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");

////////////////////////////////////////////////////////////////////////////////

/// Returns a set of constant definitions for each opcode
fn opcode_constants() -> String {
    let mut out = String::new();
    for (op, i) in fidget_bytecode::iter_ops() {
        out += &format!("const OP_{}: u32 = {i};\n", op.to_shouty_snake_case());
    }
    out
}

////////////////////////////////////////////////////////////////////////////////

/// Error type for [`init`]
#[derive(Debug, thiserror::Error)]
pub enum InitError {
    /// Error when requesting an adapter
    #[error(transparent)]
    Adapter(#[from] wgpu::RequestAdapterError),

    /// Error when requesting a device
    #[error(transparent)]
    Device(#[from] wgpu::RequestDeviceError),
}

/// Returns a WebGPU device and queue with appropriate settings
///
/// Non-default settings are as follows:
/// - We request a [`wgpu::PowerPreference::HighPerformance`] adapter
/// - We enable the [`wgpu::Features::TIMESTAMP_QUERY`] feature
///
/// This is a helper function for simplicity; more sophisticated systems will
/// likely construct the adapter, device, and queue themselves.
pub async fn init() -> Result<(wgpu::Device, wgpu::Queue), InitError> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..wgpu::RequestAdapterOptions::default()
        })
        .await?;
    let out = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..wgpu::DeviceDescriptor::default()
        })
        .await?;
    Ok(out)
}

////////////////////////////////////////////////////////////////////////////////

/// Helper function for use in unit tests
#[cfg(test)]
fn compile_shader(src: &str, desc: &str) {
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    // This isn't the best formatting, but it will at least include the
    // relevant text.
    let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
        if let Some(i) = e.location(src) {
            let pos = i.offset as usize..(i.offset + i.length) as usize;
            panic!(
                "shader compilation failed\n{src}\n{}",
                e.emit_to_string_with_path(&src[pos], desc)
            );
        } else {
            panic!(
                "shader compilation failed\n{src}\n{}",
                e.emit_to_string(desc)
            );
        }
    });
    if let Err(e) = v.validate(&m) {
        let (pos, desc) = e.spans().next().unwrap();
        panic!(
            "shader compilation failed\n{src}\n{}",
            e.emit_to_string_with_path(&src[pos.to_range().unwrap()], desc)
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{context::Tree, vm::VmShape};
    use fidget_raster::voxel::RenderSize;

    #[test]
    fn render_and_merge() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        let instance = wgpu::Instance::default();
        let (device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap()
        });

        let voxel_ctx = voxel::Context::new(device.clone(), queue.clone());
        let effects_ctx = effects::Context::new(device.clone(), queue.clone());

        let size = 32;
        let image_size = RenderSize::from(size);
        let mut buf = voxel_ctx.buffers(image_size).unwrap();
        let mut merge_buf = effects_ctx.merge_buffers(size.into()).unwrap();
        let mut shade_buf = effects_ctx.shade_buffers(size.into()).unwrap();
        let mut shade_out = effects_ctx.shaded_read_buffer(&shade_buf);

        let (x, y, z) = Tree::axes();
        let sphere =
            (x.square() + y.square() + z.square()).sqrt() - Tree::constant(0.5);
        let shape = voxel_ctx.shape(&VmShape::from(sphere)).unwrap();

        voxel_ctx
            .submit(
                &shape,
                &mut buf,
                None,
                &voxel::RenderConfig {
                    world_to_model: nalgebra::Matrix4::identity(),
                },
            )
            .unwrap();
        effects_ctx
            .submit_merge(&[buf.image_storage_buffer()], true, &mut merge_buf)
            .unwrap();
        effects_ctx
            .submit_shade(&merge_buf, &mut shade_buf, Some(&mut shade_out))
            .unwrap();
        let img = effects_ctx.map_shaded_image(&mut shade_out);
        let (_out, size) = img.image().take();
        assert_eq!(size.width(), 32);
        assert_eq!(size.height(), 32);
    }
}
