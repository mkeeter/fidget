//! Shader generation and WGPU-based image rendering
#![warn(missing_docs)]

use fidget_core::render::ImageSize;
use heck::ToShoutySnakeCase;
use zerocopy::{FromBytes, Immutable};

pub mod effects;
pub mod voxel;

/// Re-export the [`wgpu`] module
pub use wgpu;

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

/// Handle around a growable GPU buffer
///
/// The buffer keeps track of both its current size and capacity (which may be
/// larger).  It is used to prevent GPU buffer allocation churn.
struct GenericFlexBuffer<T, B, const U: u32> {
    /// Current size, which may be smaller than the buffer's capacity
    size: B,
    /// Actual GPU buffer
    data: wgpu::Buffer,
    /// Buffer label (to be used when reallocating)
    name: String,
    /// Marker for buffer data type
    _t: std::marker::PhantomData<T>,
    /// Marker for buffer size type
    _b: std::marker::PhantomData<B>,
}

/// Flexible buffer which can be resized with a single item count
type ArrayBuffer<T, const U: u32> = GenericFlexBuffer<T, usize, U>;

/// Flexible buffer which can be resized to fit an image size
type ImageBuffer<T, const U: u32> = GenericFlexBuffer<T, ImageSize, U>;

/// Module containing usage constants
mod usage {
    /// Helper macro to generate usage constants
    macro_rules! u {
        ($($flag:ident),+ $(,)?) => {
            $( wgpu::BufferUsages::$flag.bits() )|+
        };
    }
    pub const STORAGE_COPY_DST: u32 = u!(STORAGE, COPY_DST);
    pub const STORAGE_COPY_SRC: u32 = u!(STORAGE, COPY_SRC);
    pub const STORAGE_INDIRECT: u32 = u!(STORAGE, INDIRECT);
    pub const STORAGE_INDIRECT_COPY_DST: u32 = u!(STORAGE, INDIRECT, COPY_DST);
    pub const STORAGE_COPY_SRC_DST: u32 = u!(STORAGE, COPY_SRC, COPY_DST);
    pub const COPY_DST_MAP_READ: u32 = u!(COPY_DST, MAP_READ);
}

trait BufferItemCount {
    fn item_count(&self) -> usize;
}

impl BufferItemCount for usize {
    fn item_count(&self) -> usize {
        *self
    }
}

impl BufferItemCount for ImageSize {
    fn item_count(&self) -> usize {
        usize::try_from(self.width())
            .unwrap()
            .checked_mul(usize::try_from(self.height()).unwrap())
            .unwrap()
    }
}

impl<T, B: BufferItemCount + Copy, const U: u32> GenericFlexBuffer<T, B, U> {
    fn new(
        device: &wgpu::Device,
        name: String,
        size: B,
    ) -> Result<Self, BufferSizeError> {
        Self::check_size(size)?;
        let size_bytes = Self::calculate_buffer_size(size);
        let usage = wgpu::BufferUsages::from_bits(U).unwrap();
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name.as_str()),
            size: size_bytes,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self {
            data,
            size,
            name,
            _t: std::marker::PhantomData,
            _b: std::marker::PhantomData,
        })
    }

    /// Calculate size from buffer item count
    ///
    /// Size is rounded up to the nearest multiple of 4 for alignment
    fn calculate_buffer_size(item_count: B) -> u64 {
        let out = u64::try_from(item_count.item_count())
            .unwrap()
            .checked_mul(u64::try_from(std::mem::size_of::<T>()).unwrap())
            .unwrap();
        out.next_multiple_of(4)
    }

    /// Returns the active buffer size (in bytes)
    fn size_bytes(&self) -> u64 {
        Self::calculate_buffer_size(self.size)
    }

    fn check_size(size: B) -> Result<(), BufferSizeError> {
        let size = Self::calculate_buffer_size(size);
        let usage = wgpu::BufferUsages::from_bits(U).unwrap();

        let buf_ty = if usage.contains(wgpu::BufferUsages::STORAGE) {
            BufferType::Storage
        } else if usage.contains(wgpu::BufferUsages::UNIFORM) {
            BufferType::Uniform
        } else {
            BufferType::Generic
        };
        buf_ty.check(size)
    }

    /// Grows the buffer to fit a particular size in bytes
    ///
    /// If the buffer already fits that size, then no allocation is performed,
    /// but we always update the internal `item_count` (e.g. so that
    /// [`bind_active`](Self::bind_active) returns the correct subset of the
    /// buffer).
    fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        size: B,
    ) -> Result<(), BufferSizeError> {
        Self::check_size(size)?;
        let new_size = Self::calculate_buffer_size(size);
        if new_size > self.size_bytes() {
            let usage = self.data.usage();
            self.data = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.name.as_str()),
                size: new_size,
                usage,
                mapped_at_creation: false,
            });
        }
        self.size = size;
        Ok(())
    }

    /// Returns a binding resource for the active slice of the buffer
    fn bind_active(&self) -> wgpu::BindingResource<'_> {
        self.data.slice(0..self.size_bytes()).into()
    }

    /// Returns the total buffer capacity (in bytes)
    fn capacity(&self) -> u64 {
        self.data.size()
    }

    /// Maps the active portion of the buffer for reading
    fn map_async(
        &self,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>)
        + wgpu::WasmNotSend
        + 'static,
    ) -> wgpu::BufferSlice<'_> {
        let slice = self.data.slice(0..self.size_bytes());
        slice.map_async(wgpu::MapMode::Read, callback);
        slice
    }

    /// Clears the active portion of the buffer
    fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.data, 0, Some(self.size_bytes()));
    }
}

////////////////////////////////////////////////////////////////////////////////
// Error handling zone!  This is perhaps a bit overengineered, but it meets the
// desired behavior of function error types only containing errors that they can
// actually return.

/// Error type when resizing a buffer beyond its limit
///
/// We check against maximum buffer sizes (from the WebGPU spec) and return an
/// error immediately, instead of deferring the error to the point where the
/// buffer is used.
#[derive(Debug, thiserror::Error)]
pub enum BufferSizeError {
    /// Buffer size is too large for the requested buffer usage
    #[error(
        "requested size {requested_size} exceeds maximum {} for \
        {buffer_type} buffer",
        buffer_type.max_size()
    )]
    TooLarge {
        /// Size requested (in bytes)
        requested_size: u64,
        /// Buffer type (which determines the [max size](BufferType::max_size))
        buffer_type: BufferType,
    },
}

/// Buffer type for error reporting
#[derive(Copy, Clone, Debug)]
pub enum BufferType {
    /// Uniform buffer ([`wgpu::BufferUsages::UNIFORM`])
    Uniform,
    /// Storage buffer ([`wgpu::BufferUsages::STORAGE`])
    Storage,
    /// Other buffer type (e.g. [`wgpu::BufferUsages::MAP_READ`])
    Generic,
}

impl std::fmt::Display for BufferType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BufferType::Uniform => "uniform",
            BufferType::Storage => "storage",
            BufferType::Generic => "generic",
        };
        s.fmt(f)
    }
}

impl BufferType {
    /// Maximum size of this buffer type, per the WebGPU spec
    pub const fn max_size(&self) -> u64 {
        // These are copied from the spec, since we don't ask for anything extra
        match self {
            // maxUniformBufferBindingSize
            BufferType::Uniform => 64 * 1024,
            // maxStorageBufferBindingSize
            BufferType::Storage => 128 * 1024 * 1024,
            // maxBufferSize
            BufferType::Generic => 256 * 1024 * 1024,
        }
    }

    fn check(&self, requested_size: u64) -> Result<(), BufferSizeError> {
        if requested_size > self.max_size() {
            Err(BufferSizeError::TooLarge {
                requested_size,
                buffer_type: *self,
            })
        } else {
            Ok(())
        }
    }
}

/// Helper function to make a uniform buffer binding
fn buffer_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper function to make a read-only buffer binding
fn buffer_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper function to make a read-only buffer binding with dynamic offset
fn buffer_ro_dyn(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper function to make a read-write buffer binding
fn buffer_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
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

/// Debug function to read a buffer to a `Vec<T>`
#[allow(unused)]
fn read_buffer<T: FromBytes + Immutable + Clone + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
) -> Vec<T> {
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buf.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_buffer"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &scratch, 0, buf.size());
    queue.submit(Some(encoder.finish()));

    let buffer_slice = scratch.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let result = <[T]>::ref_from_bytes(&buffer_slice.get_mapped_range())
        .unwrap()
        .to_vec();
    scratch.unmap();
    result
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::{context::Tree, vm::VmShape};
    use fidget_raster::voxel::RenderSize;
    use zerocopy::IntoBytes;

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
            .submit_merge(&[buf.image_storage_buffer()], &mut merge_buf)
            .unwrap();
        effects_ctx
            .submit_shade(&merge_buf, &mut shade_buf, Some(&mut shade_out))
            .unwrap();
        let img = effects_ctx.map_shaded_image(&mut shade_out);
        let (out, size) = img.image().take();
        let mut iter = out.iter();
        for y in 0..32 {
            for x in 0..32 {
                print!("{:08x} ", iter.next().unwrap());
            }
            println!();
        }
        assert!(iter.next().is_none());

        image::save_buffer(
            "shaded.png",
            out.as_bytes(),
            size.width(),
            size.height(),
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
}
