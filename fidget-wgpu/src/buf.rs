//! Strongly-typed buffers
//!
//! This module is mostly internal to the crate, but is public because its types
//! appear as return values and arguments.
use fidget_core::render::{ImageSize, VoxelSize};
use fidget_raster::RenderSize;
use zerocopy::FromBytes;

/// Handle around a growable GPU buffer
///
/// The buffer keeps track of both its current size and capacity (which may be
/// larger).  It is used to prevent GPU buffer allocation churn.
pub struct FlexBuffer<T: BufferTag> {
    /// Current size, which may be smaller than the buffer's capacity
    size: T::S,
    /// Actual GPU buffer
    data: wgpu::Buffer,
    /// Buffer label (to be used when reallocating)
    name: String,
    /// Marker for buffer tag type
    _t: std::marker::PhantomData<T>,
}

/// Tag associated with a particular [`FlexBuffer`]
///
/// The tag type serves two purposes:
///
/// - It declares the storage type, usage bits, and size type for the buffer
/// - It makes buffers strongly typed, so that two buffers with equivalent
///   storage and usage bits can be distinct types.
pub trait BufferTag {
    /// Data type stored in the buffer
    type T;
    /// Size type
    type S: BufferItemCount + Copy;
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
    type S = T::S;
    fn usage() -> u32 {
        wgpu::BufferUsages::COPY_DST.bits()
            | wgpu::BufferUsages::MAP_READ.bits()
    }
}

/// Helper macro to declare a buffer tag
#[macro_export]
macro_rules! tag {
    ($vis:vis $name:ident, $t:ty, $s:ty, $($flag:ident)|+ $(,$doc:expr)?) => {
        $(#[doc = $doc])?
        $vis struct $name;
        impl $crate::buf::BufferTag for $name {
            type T = $t;
            type S = $s;
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

/// Tag for a flex buffer which contains a config `C` followed by many `T`
///
/// Item count is reported in bytes
pub(crate) struct FlexConfigSize<C, T> {
    count: usize,
    _c: std::marker::PhantomData<C>,
    _t: std::marker::PhantomData<T>,
}

impl<C, T> From<usize> for FlexConfigSize<C, T> {
    fn from(value: usize) -> Self {
        Self {
            count: value,
            _c: std::marker::PhantomData,
            _t: std::marker::PhantomData,
        }
    }
}

impl<C, T> Clone for FlexConfigSize<C, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C, T> Copy for FlexConfigSize<C, T> {}

impl<C, T> BufferItemCount for FlexConfigSize<C, T> {
    fn item_count(&self) -> usize {
        self.count * std::mem::size_of::<T>() + std::mem::size_of::<C>()
    }
}

pub(crate) struct FlexConfigTag<C, T> {
    _c: std::marker::PhantomData<C>,
    _t: std::marker::PhantomData<T>,
}

impl<C, T> BufferTag for FlexConfigTag<C, T>
where
    T: Copy,
{
    type T = u8;
    type S = FlexConfigSize<C, T>;
    fn usage() -> u32 {
        wgpu::BufferUsages::COPY_DST.bits() | wgpu::BufferUsages::STORAGE.bits()
    }
}

/// Buffer which contains one config `C` followed by many `T`
pub(crate) type FlexConfigBuffer<C, T> = FlexBuffer<FlexConfigTag<C, T>>;

impl<C, T> FlexConfigBuffer<C, T>
where
    C: zerocopy::IntoBytes + zerocopy::Immutable + Copy,
    T: Copy,
{
    /// Writes a config value to the buffer
    pub(crate) fn write_config(&self, c: &C, queue: &wgpu::Queue) {
        let config_len = std::mem::size_of::<C>();
        let mut writer = queue
            .write_buffer_with(
                self.data(),
                0,
                (config_len as u64).try_into().unwrap(),
            )
            .unwrap();
        writer.copy_from_slice(c.as_bytes());
    }
}

impl<T: BufferTag> FlexBuffer<T> {
    pub(crate) fn new(
        device: &wgpu::Device,
        name: impl AsRef<str>,
        size: T::S,
    ) -> Result<Self, BufferSizeError> {
        Self::check_size(size)?;
        let size_bytes = Self::calculate_buffer_size(size);
        let usage = wgpu::BufferUsages::from_bits(T::usage()).unwrap();
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name.as_ref()),
            size: size_bytes,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self {
            data,
            size,
            name: name.as_ref().to_owned(),
            _t: std::marker::PhantomData,
        })
    }

    /// Calculate size from buffer item count
    ///
    /// Size is rounded up to the nearest multiple of 4 for alignment
    pub fn calculate_buffer_size(item_count: T::S) -> u64 {
        let out = u64::try_from(item_count.item_count())
            .unwrap()
            .checked_mul(u64::try_from(std::mem::size_of::<T::T>()).unwrap())
            .unwrap();
        out.next_multiple_of(4)
    }

    /// Returns the active buffer size (in bytes)
    ///
    /// Note that this is rounded up to the nearest multiple of 4, per
    /// [`calculate_buffer_size`](Self::calculate_buffer_size)
    pub fn size_bytes(&self) -> u64 {
        Self::calculate_buffer_size(self.size)
    }

    pub(crate) fn check_size(size: T::S) -> Result<(), BufferSizeError> {
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
    ///
    /// Returns a comparison between the previous item count and the new item
    /// count (set by `size`).
    pub(crate) fn grow_to_fit(
        &mut self,
        device: &wgpu::Device,
        size: T::S,
    ) -> Result<std::cmp::Ordering, BufferSizeError> {
        Self::check_size(size)?;
        let r = self.size.item_count().cmp(&size.item_count());
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
        Ok(r)
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
    pub fn size(&self) -> T::S {
        self.size
    }
}

/// Buffer for reading data back from the GPU
///
/// Once mapped, this can be wrapped by a [`MappedImage`] (if the tag's size
/// parameter is an image size).
pub type ReadBuffer<T> = FlexBuffer<MappedBufferTag<T>>;

/// Handle to a mapped [`ReadBuffer`]
///
/// The buffer is automatically unmapped when this handle is dropped
pub struct MappedBuffer<'a, T: BufferTag> {
    buf: &'a ReadBuffer<T>,
    slice: wgpu::BufferSlice<'a>,
}

impl<T: BufferTag> Drop for MappedBuffer<'_, T> {
    fn drop(&mut self) {
        self.buf.data().unmap();
    }
}

impl<'a, T> MappedBuffer<'a, T>
where
    T: BufferTag,
    T::T: zerocopy::FromBytes + zerocopy::Immutable + Copy,
{
    /// Basic constructor
    pub(crate) fn new(
        buf: &'a ReadBuffer<T>,
        slice: wgpu::BufferSlice<'a>,
    ) -> Self {
        Self { buf, slice }
    }

    /// Copies the buffer data into a `Vec`
    pub fn to_vec(&self) -> Vec<T::T> {
        // We don't want to use the buffer's size, because it's rounded up to a
        // multiple of 4; use the raw item size (in bytes) instead.
        let slice = self.slice.get_mapped_range();
        let n = self.buf.size().item_count() * std::mem::size_of::<T::T>();
        <[T::T]>::ref_from_bytes(&slice[..n]).unwrap().to_owned()
    }
}

/// Handle to a mapped [`ReadBuffer`] containing an image
///
/// The buffer is automatically unmapped when this handle is dropped
pub struct MappedImage<'a, T: BufferTag>(MappedBuffer<'a, T>)
where
    T::S: RenderSize;

impl<'a, T> From<MappedBuffer<'a, T>> for MappedImage<'a, T>
where
    T: BufferTag,
    T::S: RenderSize,
{
    fn from(value: MappedBuffer<'a, T>) -> Self {
        Self(value)
    }
}

impl<'a, T> MappedImage<'a, T>
where
    T: BufferTag,
    T::T: zerocopy::FromBytes + zerocopy::Immutable + Copy,
    T::S: RenderSize,
{
    /// Returns the image's data
    pub fn image(&self) -> fidget_raster::Image<T::T, T::S> {
        fidget_raster::Image::build(self.0.to_vec(), self.0.buf.size()).unwrap()
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
