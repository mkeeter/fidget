//! Strongly-typed buffers
//!
//! This module is mostly internal to the crate, but is public because its types
//! appear as return values and arguments.
use fidget_core::render::{ImageSize, VoxelSize};
use zerocopy::FromBytes;

/// Handle around a growable GPU buffer
///
/// The buffer keeps track of both its current size and capacity (which may be
/// larger).  It is used to prevent GPU buffer allocation churn.
pub struct GenericFlexBuffer<T, B> {
    /// Current size, which may be smaller than the buffer's capacity
    size: B,
    /// Actual GPU buffer
    data: wgpu::Buffer,
    /// Buffer label (to be used when reallocating)
    name: String,
    /// Marker for buffer tag type
    _t: std::marker::PhantomData<T>,
}

/// Resizable array buffer
pub type ArrayBuffer<T> = GenericFlexBuffer<T, usize>;

/// Resizable image buffer
pub type ImageBuffer<T> = GenericFlexBuffer<T, ImageSize>;

/// Resizable image buffer
pub type DepthImageBuffer<T> = GenericFlexBuffer<T, VoxelSize>;

/// Tag associated with a particular [`GenericFlexBuffer`]
///
/// The tag type serves two purposes:
///
/// - It declares the storage type and usage bits for the buffer
/// - It makes buffers strongly typed, so that two buffers with equivalent
///   storage and usage bits can be distinct types.
pub trait BufferTag {
    /// Data type stored in the buffer
    type T;
    /// Usage bits for the buffer
    ///
    /// This must be a union of [`wgpu::BufferUsages`] values
    fn usage() -> u32;
}

/// Helper `struct` to make a mapped version of a storage buffer
pub struct MappedBufferTag<T: BufferTag> {
    _t: std::marker::PhantomData<T>,
}
impl<T: BufferTag> BufferTag for MappedBufferTag<T> {
    type T = T::T;
    fn usage() -> u32 {
        wgpu::BufferUsages::COPY_DST.bits()
            | wgpu::BufferUsages::MAP_READ.bits()
    }
}

/// Helper macro to declare a buffer tag
#[macro_export]
macro_rules! tag {
    ($vis:vis $name:ident,  $t:ty, $($flag:ident)|+ $(,$doc:expr)?) => {
        $(#[doc = $doc])?
        $vis struct $name;
        impl $crate::buf::BufferTag for $name {
            type T = $t;
            fn usage() -> u32 {
                $( wgpu::BufferUsages::$flag.bits() )|+
            }
        }
    }
}

/// Trait for types which have a certain number of items
pub trait BufferItemCount {
    /// The number of items
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

impl BufferItemCount for VoxelSize {
    fn item_count(&self) -> usize {
        ImageSize::from(*self).item_count()
    }
}

impl<T: BufferTag, B: BufferItemCount + Copy> GenericFlexBuffer<T, B> {
    pub(crate) fn new(
        device: &wgpu::Device,
        name: String,
        size: B,
    ) -> Result<Self, BufferSizeError> {
        Self::check_size(size)?;
        let size_bytes = Self::calculate_buffer_size(size);
        let usage = wgpu::BufferUsages::from_bits(T::usage()).unwrap();
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
        })
    }

    /// Calculate size from buffer item count
    ///
    /// Size is rounded up to the nearest multiple of 4 for alignment
    fn calculate_buffer_size(item_count: B) -> u64 {
        let out = u64::try_from(item_count.item_count())
            .unwrap()
            .checked_mul(u64::try_from(std::mem::size_of::<T::T>()).unwrap())
            .unwrap();
        out.next_multiple_of(4)
    }

    /// Returns the active buffer size (in bytes)
    pub fn size_bytes(&self) -> u64 {
        Self::calculate_buffer_size(self.size)
    }

    pub(crate) fn check_size(size: B) -> Result<(), BufferSizeError> {
        let size = Self::calculate_buffer_size(size);
        let usage = wgpu::BufferUsages::from_bits(T::usage()).unwrap();

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
    pub(crate) fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        size: B,
    ) -> Result<(), BufferSizeError> {
        Self::check_size(size)?;
        let new_size = Self::calculate_buffer_size(size);
        if new_size > self.capacity() {
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
    pub fn bind_active(&self) -> wgpu::BindingResource<'_> {
        self.data.slice(0..self.size_bytes()).into()
    }

    /// Returns the total buffer capacity (in bytes)
    pub(crate) fn capacity(&self) -> u64 {
        self.data.size()
    }

    /// Returns the buffer name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Maps the active portion of the buffer for reading
    pub(crate) fn map_async(
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
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.data, 0, Some(self.size_bytes()));
    }

    /// Returns a reference to the inner WGPU buffer
    ///
    /// Note that the whole buffer may not be active, since we allow for
    /// oversized buffers!  Use [`size_bytes`](Self::size_bytes) to get the
    /// active size, or [`bind_active`](Self::bind_active) to get a GPU binding.
    pub fn data(&self) -> &wgpu::Buffer {
        &self.data
    }

    /// Returns the size of the buffer (which is generic)
    pub fn size(&self) -> B {
        self.size
    }
}

/// Buffer for reading data back from the GPU
///
/// Once mapped, this is wrapped by a [`MappedImage`]
pub type ImageReadBuffer<T> = ImageBuffer<MappedBufferTag<T>>;

/// Handle to a mapped [`ImageReadBuffer`], which unmaps the image when dropped
pub struct MappedImage<'a, T: BufferTag> {
    buf: &'a ImageReadBuffer<T>,
    slice: wgpu::BufferSlice<'a>,
}

impl<T: BufferTag> Drop for MappedImage<'_, T> {
    fn drop(&mut self) {
        self.buf.data().unmap();
    }
}

impl<'a, T: BufferTag> MappedImage<'a, T> {
    /// Blocking function to build a new mapped image
    pub fn map(
        device: &wgpu::Device,
        image: &'a mut ImageReadBuffer<T>,
    ) -> Self {
        let slice = image.map_async(|_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        MappedImage { buf: image, slice }
    }

    /// Returns the image's data
    pub fn image(&self) -> fidget_raster::Image<u32, ImageSize> {
        let result = <[u32]>::ref_from_bytes(&self.slice.get_mapped_range())
            .unwrap()
            .to_owned();
        fidget_raster::Image::build(result, self.buf.size()).unwrap()
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
pub(crate) fn buffer_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
pub(crate) fn buffer_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
pub(crate) fn buffer_ro_dyn(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
pub(crate) fn buffer_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
