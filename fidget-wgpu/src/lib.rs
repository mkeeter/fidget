//! Shader generation and WGPU-based image rendering
#![warn(missing_docs)]

use heck::ToShoutySnakeCase;
use zerocopy::{FromBytes, Immutable};

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

/// Error type for GPU initialization
#[derive(Debug, thiserror::Error)]
pub enum InitError {
    /// Error when requesting an adapter
    #[error(transparent)]
    Adapter(#[from] wgpu::RequestAdapterError),

    /// Error when requesting a device
    #[error(transparent)]
    Device(#[from] wgpu::RequestDeviceError),
}

/// Handle to a GPU device
#[derive(Clone)]
pub struct Gpu {
    /// GPU device
    pub device: wgpu::Device,
    /// GPU queue
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Returns a [`Gpu`] object with customized settings
    ///
    /// Non-default settings are as follows:
    /// - We request a [`wgpu::PowerPreference::HighPerformance`] adapter
    /// - We enable the [`wgpu::Features::TIMESTAMP_QUERY`] feature
    ///
    /// This is a helper function for simplicity; more sophisticated systems
    /// will likely construct the adapter, device, and queue themselves.
    pub async fn init() -> Result<Gpu, InitError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..wgpu::RequestAdapterOptions::default()
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                ..wgpu::DeviceDescriptor::default()
            })
            .await?;
        Ok(Gpu { device, queue })
    }

    /// Returns a [`Gpu`] object with default settings
    ///
    /// This is useful for CI, where `TIMESTAMP_QUERY` is unsupported
    #[doc(hidden)]
    pub async fn init_basic() -> Result<Gpu, InitError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        Ok(Gpu { device, queue })
    }

    /// Returns a readable buffer for the given image buffer
    pub fn read_buffer_for<T: buf::BufferTag>(
        &self,
        buf: &buf::ImageBuffer<T>,
    ) -> buf::ImageReadBuffer<T> {
        buf::ImageBuffer::new(
            &self.device,
            format!("{} (read)", buf.name()),
            buf.size(),
        )
        .expect("buf.size should always be a valid size for ImageBuffer::new")
    }

    /// Maps a readable image buffer, returning a mapped image
    pub fn map<'a, T: buf::BufferTag>(
        &self,
        buf: &'a mut buf::ImageReadBuffer<T>,
    ) -> buf::MappedImage<'a, T> {
        buf::MappedImage::map(&self.device, buf)
    }

    /// Debug function to read from a buffer to a `Vec<T>`
    pub fn read_vec<T: FromBytes + Immutable + Clone + Copy>(
        &self,
        buf: &wgpu::Buffer,
    ) -> Vec<T> {
        let scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buf.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &scratch, 0, buf.size());
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = scratch.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let result = <[T]>::ref_from_bytes(&buffer_slice.get_mapped_range())
            .unwrap()
            .to_vec();
        scratch.unmap();
        result
    }
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

        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let voxel_ctx = voxel::Context::new(&gpu);
        let effects_ctx = effects::Context::new(&gpu);

        let size = 128;
        let image_size = RenderSize::from(size);
        let mut buf = voxel_ctx.buffers();
        let mut merge_buf = effects_ctx.merge_buffers(size.into()).unwrap();
        let mut shade_buf = effects_ctx.shade_buffers(size.into()).unwrap();
        let mut shade_out = gpu.read_buffer_for(shade_buf.output());

        let (x, y, z) = Tree::axes();
        let x_ = x.clone() - 0.2;
        let sphere1 = (x_.square() + y.square() + z.square()).sqrt()
            - Tree::constant(0.5);
        let x_ = x + 0.2;
        let sphere2 = (x_.square() + y.square() + z.square()).sqrt()
            - Tree::constant(0.5);
        let spheres = sphere1.min(sphere2);
        let shape = voxel_ctx.shape(&VmShape::from(spheres)).unwrap();

        voxel_ctx
            .submit(
                &shape,
                &mut buf,
                None,
                &voxel::RenderConfig {
                    image_size,
                    world_to_model: nalgebra::Matrix4::identity(),
                },
            )
            .unwrap();
        effects_ctx
            .submit_merge(&[buf.image_storage_buffer()], true, &mut merge_buf)
            .unwrap();
        let mut ssao_buf = effects_ctx.ssao_buffers(size.into()).unwrap();
        effects_ctx.submit_ssao(&merge_buf, &mut ssao_buf).unwrap();
        effects_ctx
            .submit_shade(
                &merge_buf,
                Some(&ssao_buf),
                &mut shade_buf,
                Some(&mut shade_out),
            )
            .unwrap();
        let img = gpu.map(&mut shade_out);
        let (_out, img_size) = img.image().take();
        assert_eq!(img_size.width(), size);
        assert_eq!(img_size.height(), size);
    }
}
