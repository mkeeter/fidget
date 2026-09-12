//! Shader generation and WGPU-based image rendering
//!
//! # API design
//! Using GPUs is complicated<sup>[citation needed]</sup>.  The APIs in this
//! crate try to strike a balance between ease of use and efficiency.  As
//! always, feel free to open an issue or discussion if the APIs don't work for
//! you; they were codesigned along with
//! [Halfspace](https://github.com/mkeeter/halfspace), and may not yet be
//! suitable for every use case.
//!
//! ## Object types
//! All of the modules use similar patterns of objects:
//!
//! - The [`Gpu`] object is passed around to provide device and queues
//! - A `Context` object contains pipelines
//! - A `Workspace` object contains buffers used when rendering
//! - An output buffer can be read back to the CPU or passed to a subsequent
//!   render pipeline
//!
//! ### Context objects
//! A context object contains GPU pipelines and allow users to dispatch work to
//! the GPU.  Voxel and pixel rendering are managed by [`voxel::Context`] and
//! [`pixel::Context`] respectively.  Post-processing is done by
//! [`voxel::effects::Context`] and [`pixel::effects::Context`].
//!
//! Users are expected to create one (of each) context object per thread or
//! worker, since GPU resources can't be shared.
//!
//! Context objects have two flavors of functions.  At the highest level, `run`
//! and `run_async` functions perform rendering and copy data back to the CPU
//! (e.g. [`voxel::Context::run`] and [`run_async`](voxel::Context::run_async)).
//! To simply submit work to the GPU, use a `submit` function (e.g.
//! [`voxel::Context::submit`]).
//!
//! ### Workspace objects
//! Workspace objects contain all of the buffers that are used when dispatching
//! work to the GPU.  They are also per-thread (or per-worker).  You may have
//! more than one per thread if you want to dispatch multiple jobs
//! simultaneously; it's your computer.
//!
//! Workspaces resize themselves automatically when used in rendering.  They
//! typically have [`size()`](voxel::Workspace::size) (active bytes) and
//! [`capacity()`](voxel::Workspace::capacity) (total allocated bytes)
//! functions; users may want to check for overly large ratios and recreate
//! workspaces.
//!
//! Workspaces are stateful; after they are used in a `submit` function, they
//! will contain data in a GPU buffer.  The output buffer is typically accessed
//! with the `output()` function, e.g. [`voxel::Workspace::output`].
//!
//! ### Reading data from buffers
//! The output of a workspace is a [`FlexBuffer`](crate::buf::FlexBuffer)
//! (indeed, they are used pervasively throughout this crate).  Output buffers
//! are created with `STORAGE | COPY_SRC`.  Reading data back to the CPU is a
//! three-part process:
//!
//! - Create a CPU-readable buffer with [`Gpu::read_buffer_for`]
//! - Copy data with [`Gpu::copy`]
//! - Map the readable buffer with [`Gpu::map`] or [`Gpu::map_async`]
//!
//! For quick debugging, [`Gpu::read_vec`] does all of these steps.  You
//! wouldn't want to use in a tight loop, since it allocates a GPU buffer on
//! each call.
#![warn(missing_docs)]

use fidget_bytecode::{Bytecode, ReservedRegister};
use fidget_core::{
    eval::Function,
    shape::{MissingVar, ShapeVars},
    var::{Var, VarMap},
    vm::VmShape,
};
use fidget_raster::RenderSize;

use heck::ToShoutySnakeCase;
use zerocopy::{FromBytes, Immutable, IntoBytes};

pub mod buf;
pub mod color;
pub mod pixel;
pub mod voxel;

/// Re-export the `wgpu` module
pub use wgpu;

pub(crate) mod shaders {
    use super::*;

    pub const COMMON: &str = include_str!("shaders/common.wgsl");
    pub const DUMMY_STACK: &str = include_str!("shaders/dummy_stack.wgsl");
    pub const FLOAT_OPS: &str = include_str!("shaders/float_ops.wgsl");
    pub const GRAD_OPS: &str = include_str!("shaders/grad_ops.wgsl");
    pub const INTERVAL_OPS: &str = include_str!("shaders/interval_ops.wgsl");
    pub const STACK: &str = include_str!("shaders/stack.wgsl");
    pub const TAPE_INTERPRETER: &str =
        include_str!("shaders/tape_interpreter.wgsl");
    pub const TAPE_SIMPLIFY: &str = include_str!("shaders/tape_simplify.wgsl");

    /// Returns a set of constant definitions for each opcode
    pub fn opcode_constants() -> String {
        let mut out = String::new();
        for (op, i) in fidget_bytecode::iter_ops() {
            out += &format!(
                "const OP_{}: u32 = {i};\n",
                op.to_shouty_snake_case()
            );
        }
        out
    }
}

/// Number of [`TapeWord`] words in the tape data flexible array
pub(crate) const TAPE_DATA_CAPACITY: usize = 8 * 1024 * 1024; // 8M words, 64 MiB

#[repr(C)]
pub(crate) struct TapeWord {
    op: u32,
    imm: u32,
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

    /// Returns a readable buffer for the given buffer
    ///
    /// See [`copy`](Self::copy) and [`map`](Self::map) for how to use the
    /// resulting read buffer.
    pub fn read_buffer_for<T>(
        &self,
        buf: &buf::FlexBuffer<T>,
    ) -> buf::ReadBuffer<T>
    where
        T: buf::BufferTag,
    {
        buf::ReadBuffer::new(
            &self.device,
            format!("{} (read)", buf.name()),
            buf.size(),
        )
        .expect("buf.size should always be a valid size for ReadBuffer::new")
    }

    /// Builds a new readable buffer
    ///
    /// The buffer has an arbitrary starting size and will be resized when used
    /// in [`copy`](Self::copy).
    ///
    /// If the target buffer is known, use
    /// [`read_buffer_for`](Self::read_buffer_for) to avoid this reallocation.
    pub fn read_buffer<T>(&self, name: &str) -> buf::ReadBuffer<T>
    where
        T: buf::BufferTag,
        T::S: TryFrom<u32>,
    {
        let Ok(size) = 64u32.try_into() else {
            panic!("could not build size");
        };
        buf::ReadBuffer::new(&self.device, name, size)
            .expect("64 should always be a valid size for ReadBuffer::new")
    }

    /// Helper function to read from a buffer to a `Vec`
    ///
    /// Under the hood, this function simply calls
    /// [`read_buffer_for`](Self::read_buffer_for),
    /// [`copy`](Self::copy), [`map`](Self::map), and
    /// [`to_vec`](buf::MappedBuffer::to_vec).
    ///
    /// Note that this function allocates a GPU buffer; if the same source
    /// buffer is being read repeatedly, it's recommended to build the read
    /// buffer *once* and then call `copy`, `map`, `to_vec` repeatedly.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_vec<T: buf::BufferTag>(
        &self,
        buf: &buf::FlexBuffer<T>,
    ) -> Vec<T::T>
    where
        T::T: FromBytes + Immutable + Clone + Copy,
    {
        let mut scratch = self.read_buffer_for(buf);
        self.copy(buf, &mut scratch);
        let data = self.map(&mut scratch);
        data.to_vec()
    }

    /// Copies from a GPU-resident buffer to a host-mappable buffer
    ///
    /// The host-mappable destination buffer is resized to fit the data.
    pub fn copy<A>(
        &self,
        src: &buf::FlexBuffer<A>,
        dst: &mut buf::ReadBuffer<A>,
    ) where
        A: buf::BufferTag,
    {
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            },
        );
        self.encode_copy(src, dst, &mut encoder);
        self.queue.submit(Some(encoder.finish()));
    }

    /// Low-level command to submit a buffer copy to a command encoder
    ///
    /// See [`copy`](Self::copy) for details
    pub fn encode_copy<A>(
        &self,
        src: &buf::FlexBuffer<A>,
        dst: &mut buf::ReadBuffer<A>,
        encoder: &mut wgpu::CommandEncoder,
    ) where
        A: buf::BufferTag,
    {
        dst.grow_to_fit(&self.device, src.size())
            .expect("dst buffer should be resizable to match src buffer");
        encoder.copy_buffer_to_buffer(
            src.data(),
            0,
            dst.data(),
            0,
            src.size_bytes(),
        );
    }

    /// Blocking function to build a new mapped buffer
    #[cfg(any(not(target_arch = "wasm32"), doc))]
    pub fn map<'a, T>(
        &self,
        image: &'a mut buf::ReadBuffer<T>,
    ) -> buf::MappedBuffer<'a, T>
    where
        T: buf::BufferTag,
        T::T: Immutable + FromBytes + Copy,
    {
        pollster::block_on(self.map_async(image))
    }

    /// Async function to build a new mapped buffer
    ///
    /// This can be called on either native or web platforms.  On native
    /// platforms, the single `await` is trivial (guaranteed to always be
    /// ready); on the web, the mapping sends us back to the event loop until
    /// it's ready.
    pub async fn map_async<'a, T>(
        &self,
        data: &'a mut buf::ReadBuffer<T>,
    ) -> buf::MappedBuffer<'a, T>
    where
        T: buf::BufferTag,
        T::T: Immutable + FromBytes + Copy,
    {
        let (tx, rx) = flume::bounded(1);
        let slice = data.map_async(move |_| tx.send(()).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv_async().await.unwrap();
        buf::MappedBuffer::new(data, slice)
    }

    /// Blocking function to build a new mapped image
    #[cfg(any(not(target_arch = "wasm32"), doc))]
    pub fn map_image<'a, T>(
        &self,
        data: &'a mut buf::ReadBuffer<T>,
    ) -> buf::MappedImage<'a, T>
    where
        T: buf::BufferTag,
        T::T: Immutable + FromBytes + Copy,
        T::S: RenderSize,
    {
        pollster::block_on(self.map_image_async(data))
    }

    /// Async function to build a new mapped image
    ///
    /// This can be called on either native or web platforms.  On native
    /// platforms, the single `await` is trivial (guaranteed to always be
    /// ready); on the web, the mapping sends us back to the event loop until
    /// it's ready.
    pub async fn map_image_async<'a, T>(
        &self,
        data: &'a mut buf::ReadBuffer<T>,
    ) -> buf::MappedImage<'a, T>
    where
        T: buf::BufferTag,
        T::T: Immutable + FromBytes + Copy,
        T::S: RenderSize,
    {
        self.map_async(data).await.into()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Container of multiple pipelines, parameterized by register count
pub(crate) struct RegPipeline {
    thunk: Box<dyn Fn(u8) -> wgpu::ComputePipeline>,
    cache:
        [std::cell::OnceCell<wgpu::ComputePipeline>; REG_PIPELINE_SIZES.len()],
}

const REG_PIPELINE_SIZES: &[u8] = &[8, 16, 32, 64, 128, 192, 255];

impl RegPipeline {
    pub fn build(builder: Box<dyn Fn(u8) -> wgpu::ComputePipeline>) -> Self {
        Self {
            thunk: builder,
            cache: std::array::from_fn(|_| Default::default()),
        }
    }

    /// Returns the pipeline with sufficient registers to render `reg_count`
    ///
    /// # Panics
    /// If `reg_count` is 256 (which is not allowed in bytecode tapes)
    pub fn get(&self, reg_count: u8) -> &wgpu::ComputePipeline {
        let (i, r) = REG_PIPELINE_SIZES
            .iter()
            .enumerate()
            .find(|(_i, r)| reg_count <= **r)
            .expect("bytecode tape cannot use more than 255 registers");
        self.cache[i].get_or_init(|| (*self.thunk)(*r))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Shape for rendering on the GPU
///
/// Note that this object does not allocate any memory on the GPU itself; it
/// stores a bytecode-serialized version of the shape, which is copied to the
/// GPU during rendering.
pub struct RenderShape {
    /// Copy of our shape (kept around for access to the variable map)
    shape: VmShape,
    /// Serialized bytecode for the shape
    bytecode: Bytecode,
}

/// Error type when constructing a [`RenderShape`]
#[derive(Debug, thiserror::Error)]
pub enum RenderShapeError {
    /// The shape doesn't fit in the GPU tape buffer
    #[error(
        "shape bytecode is {0} tape words (8 bytes each), which exceeds \
        buffer capacity of {TAPE_DATA_CAPACITY} tape words"
    )]
    TooLong(usize),
    /// The shape uses a reserved register
    #[error(transparent)]
    RegisterError(#[from] ReservedRegister),
}

impl RenderShape {
    /// Builds a new render shape
    pub fn new(shape: &VmShape) -> Result<Self, RenderShapeError> {
        // Generate bytecode for the root tape
        let bytecode = Bytecode::new(shape.inner().data())?;
        if bytecode.len() / 2 > TAPE_DATA_CAPACITY {
            return Err(RenderShapeError::TooLong(bytecode.len() / 2));
        }

        Ok(Self {
            shape: shape.clone(),
            bytecode,
        })
    }

    /// Helper function to return XYZ variable indices
    fn axes(&self) -> [u32; 3] {
        let vars = self.shape.inner().vars();
        [Var::X, Var::Y, Var::Z]
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
    }

    /// Copies variables into a variables buffer
    ///
    /// Returns `true` if the buffer size changed, which could invalidate cached
    /// bind groups.
    fn copy_vars(
        &self,
        gpu: &Gpu,
        vars: &ShapeVars<f32>,
        buf: &mut buf::FlexBuffer<voxel::VarsBufferTag>,
    ) -> Result<CopyVarsChanged, CopyVarsError> {
        copy_vars(gpu, self.shape.inner().vars(), vars, buf)
    }
}

pub(crate) fn copy_vars(
    gpu: &Gpu,
    vs: &VarMap,
    vars: &ShapeVars<f32>,
    buf: &mut buf::FlexBuffer<voxel::VarsBufferTag>,
) -> Result<CopyVarsChanged, CopyVarsError> {
    let mut changed = CopyVarsChanged::BufferUnchanged;
    if vs.has_free_vars() {
        // Do an initial pass to check for errors before resizing the buffer
        for (v, _i) in vs.iter() {
            match v {
                Var::X | Var::Y | Var::Z => (),
                Var::V(vi) => {
                    if vars.get(vi).is_none() {
                        return Err(MissingVar { var: vi }.into());
                    };
                }
            }
        }
        // If we have to change the vars buffer size, then we'll return
        // `true` indicating that things have changed and bind groups should
        // be invalidated.  TODO: only do this if we grow the buffer, since
        // binding an overly-large buffer is fine?
        let r = buf.grow_to_fit(&gpu.device, vs.len())?;
        if !matches!(r, std::cmp::Ordering::Equal) {
            changed = CopyVarsChanged::BufferChanged;
        }
        let mut writer = gpu
            .queue
            .write_buffer_with(
                buf.data(),
                0,
                ((vs.len() * std::mem::size_of::<f32>()) as u64)
                    .try_into()
                    .unwrap(),
            )
            .unwrap();
        for (v, i) in vs.iter() {
            match v {
                Var::X | Var::Y | Var::Z => (),
                Var::V(vi) => {
                    let value = vars.get(vi).unwrap(); // checked above
                    let offset = i * std::mem::size_of::<f32>();
                    writer
                        .slice(offset..offset + 4)
                        .copy_from_slice(value.as_bytes());
                }
            }
        }
    }
    Ok(changed)
}

#[derive(thiserror::Error, Debug)]
enum CopyVarsError {
    #[error(transparent)]
    BufferSize(#[from] buf::BufferSizeError),
    #[error(transparent)]
    MissingVar(#[from] MissingVar),
}

#[must_use]
#[derive(Copy, Clone, Debug)]
enum CopyVarsChanged {
    BufferChanged,
    BufferUnchanged,
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
    use std::collections::HashSet;

    #[test]
    fn shader_has_all_ops() {
        for (op, _) in fidget_bytecode::iter_ops() {
            let op = format!("OP_{}", op.to_shouty_snake_case());
            assert!(
                shaders::TAPE_INTERPRETER.contains(&op),
                "tape interpreter is missing {op}"
            );
            assert!(
                shaders::TAPE_SIMPLIFY.contains(&op),
                "tape simplification is missing {op}"
            );
        }
    }

    pub(crate) fn compare_struct_layout<T: facet::Facet<'static>>(
        shader: &str,
        struct_name: &str,
    ) {
        let module = naga::front::wgsl::parse_str(shader).expect("valid WGSL");
        let (members, span) = module
            .types
            .iter()
            .find_map(|(_, ty)| {
                if ty.name.as_deref() == Some(struct_name)
                    && let naga::TypeInner::Struct { members, span } = &ty.inner
                {
                    Some((members, *span))
                } else {
                    None
                }
            })
            .expect("could not find struct");

        // If the last member of the struct is a dynamically sized array, we'll
        // treat the beginning offset of the array as our struct size.
        let dynamic_array_offset = members.last().and_then(|m| {
            let ty = &module.types[m.ty];
            let naga::TypeInner::Array {
                base: _,
                size: naga::ir::ArraySize::Dynamic,
                stride: _,
            } = &ty.inner
            else {
                return None;
            };
            Some(m.offset)
        });
        if let Some(dynamic_array_offset) = dynamic_array_offset {
            assert_eq!(
                dynamic_array_offset as usize,
                std::mem::size_of::<T>(),
                "dynamic array offset is incorrect; invalid last member size?"
            );
        } else {
            assert_eq!(
                span as usize,
                std::mem::size_of::<T>(),
                "overall size is incorrect"
            );
        }

        let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty
        else {
            panic!("must build a struct");
        };

        // Check field sizes and offset between Rust and WGSL
        let mut shape_field_names = HashSet::new();
        for field in shape.fields {
            let field_name = field.name;
            shape_field_names.insert(field_name);
            let wgsl_member = members
                .iter()
                .find(|m| m.name.as_deref() == Some(field_name))
                .unwrap_or_else(|| {
                    panic!("field `{field_name}` missing in WGSL struct")
                });
            assert_eq!(
                wgsl_member.offset as usize, field.offset,
                "offset mismatch for field `{field_name}`"
            );
            assert_eq!(
                module.types[wgsl_member.ty].inner.size(module.to_ctx())
                    as usize,
                field.shape().layout.sized_layout().unwrap().size(),
                "size mismatch for field `{field_name}`"
            );
        }
        let slice_len = if dynamic_array_offset.is_some() {
            members.len() - 1
        } else {
            members.len()
        };
        for m in &members[..slice_len] {
            let field_name =
                m.name.as_ref().expect("cannot check unnamed WGSL fields");
            assert!(
                shape_field_names.contains(field_name.as_str()),
                "field `{field_name}` missing in Rust struct"
            );
        }
    }
}
