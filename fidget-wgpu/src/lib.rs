//! Shader generation and WGPU-based image rendering
#![warn(missing_docs)]
pub mod voxel;

use fidget_core::render::ImageSize;
use heck::ToShoutySnakeCase;

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

/// Handle around a growable GPU buffer which pretends to be smaller
struct GenericFlexBuffer<T, B, const U: u32> {
    /// Current item count, which may be smaller than the buffer's capacity
    item_count: usize,
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
        item_count: B,
    ) -> Result<Self, BufferSizeError> {
        Self::check_size(item_count)?;
        let item_count = item_count.item_count();
        let size = Self::calculate_buffer_size(item_count);
        let usage = wgpu::BufferUsages::from_bits(U).unwrap();
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name.as_str()),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self {
            data,
            item_count,
            name,
            _t: std::marker::PhantomData,
            _b: std::marker::PhantomData,
        })
    }

    /// Calculate size from buffer item count
    ///
    /// Size is rounded up to the nearest multiple of 4 for alignment
    fn calculate_buffer_size(item_count: usize) -> u64 {
        let out = u64::try_from(item_count)
            .unwrap()
            .checked_mul(u64::try_from(std::mem::size_of::<T>()).unwrap())
            .unwrap();
        out.next_multiple_of(4)
    }

    /// Returns the active buffer size (in bytes)
    fn size(&self) -> u64 {
        Self::calculate_buffer_size(self.item_count)
    }

    fn check_size(item_count: B) -> Result<(), BufferSizeError> {
        let item_count = item_count.item_count();
        let size = Self::calculate_buffer_size(item_count);
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
        item_count: B,
    ) -> Result<(), BufferSizeError> {
        Self::check_size(item_count)?;
        let item_count = item_count.item_count();
        if item_count > self.item_capacity() {
            let size = Self::calculate_buffer_size(item_count);
            let usage = self.data.usage();
            self.data = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.name.as_str()),
                size,
                usage,
                mapped_at_creation: false,
            });
        }
        self.item_count = item_count;
        Ok(())
    }

    /// Returns a binding resource for the active slice of the buffer
    fn bind_active(&self) -> wgpu::BindingResource<'_> {
        self.data.slice(0..self.size()).into()
    }

    /// Returns the buffer's total capacity (in items)
    ///
    /// This may be larger than [`self.item_count`](Self::item_count)
    fn item_capacity(&self) -> usize {
        let c = usize::try_from(self.capacity()).unwrap();
        assert_eq!(c % std::mem::size_of::<T>(), 0);
        c / std::mem::size_of::<T>()
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
        let slice = self.data.slice(0..self.size());
        slice.map_async(wgpu::MapMode::Read, callback);
        slice
    }

    /// Clears the active portion of the buffer
    fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.data, 0, Some(self.size()));
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
