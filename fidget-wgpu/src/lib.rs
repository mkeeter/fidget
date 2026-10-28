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
//! The [`Gpu`] object provides device and queue access, as well as functions
//! to create, map, and read from buffers.  In addition, it lets users create
//! two kinds of render objects:
//!
//! - Render shape objects ([`RenderShape`]) are used for inside / outside
//!   evaluation, generating heightmaps + normals (in 3D), and distance fields
//!   (in 2D).  This mimicks functionality in `fidget::render`.
//! - Render color objects ([`RenderColor`](color::RenderColor)) are used for
//!   per-pixel color evaluation.
//!
//! Render objects store data in GPU buffers and should only be regenerated when
//! shapes actually change (i.e. not once per frame).
//!
//! To actually perform GPU work, each module contains a `Context` object, which
//! implements the [`Context`] trait.  This is where the bulk of the API design
//! lives; see its docs for details!
//!
//! ## Reading data from buffers
//! The output of a rendering pass is a [`FlexBuffer`](crate::buf::FlexBuffer),
//! wrapping a WGPU buffer created with `STORAGE | COPY_SRC`.  Reading data back
//! to the CPU is a three-part process:
//!
//! - Create a CPU-readable buffer with [`Gpu::read_buffer_for`]
//! - Copy data with [`Gpu::copy`]
//! - Map the readable buffer with [`Gpu::map`] or [`Gpu::map_async`]
//!
//! For quick debugging, [`Gpu::read_vec`] does all of these steps.  You
//! wouldn't want to use in a tight loop, since it allocates a GPU buffer on
//! each call.
//!
//! ## Three kinds of rendering functions
//! Context objects have three flavors of functions, with increasing level of
//! control (and required boilerplate).
//!
//! ### `run` functions
//! At the highest level, `run` and `run_async` functions perform rendering
//! and copy data back to the CPU (e.g. [`voxel::render::Context::run`] and
//! [`run_async`](voxel::render::Context::run_async)).  These are not present on
//! every `Context` object; only ones where it seemed useful.
//!
//! ```
//! # fidget_wgpu::require_gpu!();
//! use fidget_wgpu::{Context as _, Gpu, voxel::render::Context};
//!
//! // GPU setup (this only needs to happen once)
//! let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
//! let ctx = Context::new(&gpu);
//! let mut workspace = ctx.workspace();
//!
//! // Per-shape setup
//! let x = fidget_core::context::Tree::x();
//! let shape = fidget_core::vm::VmShape::from(x);
//! let render_shape = gpu.build_render_shape(&shape).unwrap();
//!
//! // Select our render size
//! let mut buffers = ctx.buffers(128.into()).unwrap();
//! let mut out = gpu.read_buffer_for(buffers.output());
//!
//! // Use defaults for other render configuration
//! let render_config = fidget_wgpu::voxel::render::Settings::default();
//!
//! // Do the actual work!
//! let image = ctx.run(
//!     &render_shape,
//!     render_config,
//!     &mut workspace,
//!     &mut buffers,
//!     &mut out,
//! ).unwrap();
//!
//! // Check that the image is correct
//! assert_eq!(image.size().width(), 128);
//! assert_eq!(image.size().height(), 128);
//! for y in 0..128 {
//!     for x in 0..128 {
//!         let expected = if x < 64 {
//!             127
//!         } else {
//!             0
//!         };
//!         assert_eq!(image[(y, x)].depth, expected, "bad pixel at {x}, {y}");
//!     }
//! }
//! ```
//!
//! ### `submit` functions
//! To simply submit work to the GPU, call its [`submit`](Context::submit)
//! function.
//!
//! ```
//! # fidget_wgpu::require_gpu!();
//! # use fidget_wgpu::{Context as _, Gpu, voxel::render::Context};
//! # let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
//! # let ctx = Context::new(&gpu);
//! # let mut workspace = ctx.workspace();
//! # let x = fidget_core::context::Tree::x();
//! # let shape = fidget_core::vm::VmShape::from(x);
//! # let render_shape = gpu.build_render_shape(&shape).unwrap();
//! # let mut buffers = ctx.buffers(128.into()).unwrap();
//! # let mut out = gpu.read_buffer_for(buffers.output());
//! # let render_config = fidget_wgpu::voxel::render::Settings::default();
//! // Setup is done per the previous example
//!
//! // Submit the work to the queue
//! ctx.submit(
//!     &render_shape,
//!     render_config,
//!     &mut workspace,
//!     &mut buffers,
//! ).unwrap();
//!
//! // We'll also enqueue a copy to our output buffer
//! gpu.copy(&buffers.output(), &mut out);
//!
//! // Mapping the image is a blocking step which waits for the GPU
//! let image = gpu.map_image(&mut out).image();
//!
//! // Then we can do the same image tests as above
//! # assert_eq!(image.size().width(), 128);
//! # assert_eq!(image.size().height(), 128);
//! # for y in 0..128 {
//! #     for x in 0..128 {
//! #         let expected = if x < 64 {
//! #             127
//! #         } else {
//! #             0
//! #         };
//! #         assert_eq!(image[(y, x)].depth, expected, "bad pixel at {x}, {y}");
//! #     }
//! # }
//! ```
//!
//! ### `encode` functions
//! At the lowest level, to encode a rendering operation into a WebGPU
//! `CommandEncoder`, use an [`encode`](Context::encode) function.  `encode`
//! functions are expected to copy small amounts of data to GPU memory (e.g.
//! transform matrices in a configuration object), so `encode` takes a
//! [`wgpu::util::StagingBelt`] object.
//! ```
//! # fidget_wgpu::require_gpu!();
//! # use fidget_wgpu::{Context as _, Gpu, voxel::render::Context};
//! # let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
//! # let ctx = Context::new(&gpu);
//! # let mut workspace = ctx.workspace();
//! # let x = fidget_core::context::Tree::x();
//! # let shape = fidget_core::vm::VmShape::from(x);
//! # let render_shape = gpu.build_render_shape(&shape).unwrap();
//! # let mut buffers = ctx.buffers(128.into()).unwrap();
//! # let mut out = gpu.read_buffer_for(buffers.output());
//! # let render_config = fidget_wgpu::voxel::render::Settings::default();
//! // Setup is done per the previous example
//!
//! // We need to provide the encoder and staging buffer ourselves
//! let mut encoder = gpu.device.create_command_encoder(
//!     &wgpu::CommandEncoderDescriptor { label: None },
//! );
//! let mut staging = wgpu::util::StagingBelt::new(gpu.device.clone(), 1024);
//!
//! // Send the work to our encoder
//! ctx.encode(
//!     &render_shape,
//!     render_config,
//!     &mut workspace,
//!     &mut buffers,
//!     &mut encoder,
//!     &mut staging,
//! ).unwrap();
//!
//! // Note the use of `encode_copy`, which sends the copy operation to an
//! // encoder (rather than submitting work to the GPU queue itself)
//! gpu.encode_copy(&buffers.output(), &mut out, &mut encoder);
//!
//! // Submit the `StagingBelt` copies
//! staging.finish();
//!
//! // ...and we're done
//! gpu.queue.submit(Some(encoder.finish()));
//! staging.recall();
//!
//! // Everything else proceeds as above (using `gpu.map_image(&mut out)` to get
//! // the image data back to the host and test that it's correct).
//! # let image = gpu.map_image(&mut out).image();
//! # assert_eq!(image.size().width(), 128);
//! # assert_eq!(image.size().height(), 128);
//! # for y in 0..128 {
//! #     for x in 0..128 {
//! #         let expected = if x < 64 {
//! #             127
//! #         } else {
//! #             0
//! #         };
//! #         assert_eq!(image[(y, x)].depth, expected, "bad pixel at {x}, {y}");
//! #     }
//! # }
//! ```
//!
#![warn(missing_docs)]

use fidget_bytecode::{Bytecode, ReservedRegister};
use fidget_core::{
    shape::{MissingVar, ShapeVars},
    var::{Var, VarMap},
    vm::VmShape,
};
use fidget_raster::RenderSize;

use heck::ToShoutySnakeCase;
use std::sync::Arc;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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

static EMPTY_VARS: ShapeVars<f32> = ShapeVars::new();

////////////////////////////////////////////////////////////////////////////////

/// Type indicating an image size mismatch in a 2D image
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {}, got {} × {}",
    expected.width(), expected.height(),
    actual.width(),   actual.height(),
)]
pub struct ImageSizeMismatch {
    expected: fidget_core::render::ImageSize,
    actual: fidget_core::render::ImageSize,
}

/// Type indicating an image size mismatch in a 3D image
#[derive(Debug, thiserror::Error)]
#[error(
    "image size mismatch: expected {} × {} × {}, got {} × {} × {}",
    expected.width(), expected.height(), expected.depth(),
    actual.width(),   actual.height(),   actual.depth()
)]
pub struct VoxelSizeMismatch {
    expected: fidget_core::render::VoxelSize,
    actual: fidget_core::render::VoxelSize,
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

/// Error returned for an invalid buffer copy operation
#[derive(Debug, thiserror::Error)]
#[error(
    "bad copy size: source buffer has {src} bytes \
    but destination has {dst} bytes"
)]
pub struct BadCopySize {
    src: u64,
    dst: u64,
}

/// Handle to a GPU device
#[derive(Clone)]
pub struct Gpu {
    /// GPU device
    pub device: wgpu::Device,
    /// GPU queue
    pub queue: wgpu::Queue,

    /// Layout for a single read-only storage buffer
    input_bind_group_layout: wgpu::BindGroupLayout,

    /// Layout for a single read-write storage buffer
    output_bind_group_layout: wgpu::BindGroupLayout,

    /// Layout for a config bind group (a single read-only uniforms buffer)
    config_bind_group_layout: wgpu::BindGroupLayout,

    /// Layout for a color bind group
    ///
    /// This is used when building [`RenderColor`](crate::color::RenderColor)
    /// handles.
    ///
    /// It binds three buffers:
    /// - 0: `shape_tape`
    /// - 1: `shape_start`
    /// - 2: `var_values`
    color_bind_group_layout: wgpu::BindGroupLayout,
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
        Self::init_inner(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..wgpu::RequestAdapterOptions::default()
            },
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                ..wgpu::DeviceDescriptor::default()
            },
        )
        .await
    }

    /// Returns a [`Gpu`] object with default settings
    ///
    /// This is useful for CI, where `TIMESTAMP_QUERY` is unsupported
    #[doc(hidden)]
    pub async fn init_basic() -> Result<Gpu, InitError> {
        Self::init_inner(&Default::default(), &Default::default()).await
    }

    async fn init_inner(
        adapter_options: &wgpu::RequestAdapterOptions<'_, '_>,
        device_descriptor: &wgpu::DeviceDescriptor<'_>,
    ) -> Result<Self, InitError> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(adapter_options).await?;
        let (device, queue) = adapter.request_device(device_descriptor).await?;
        Ok(Self::new(device, queue))
    }

    /// Builds a new `Gpu` with user-provided GPU handles
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let config_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("config bind group layout"),
                entries: &[buf::buffer_uniform(0)],
            });
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("input bind group layout"),
                entries: &[buf::buffer_ro(0)],
            });
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("output bind group layout"),
                entries: &[buf::buffer_rw(0)],
            });
        let color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("color bind group layout"),
                entries: &[
                    buf::buffer_ro(0),
                    buf::buffer_ro(1),
                    buf::buffer_ro(2),
                ],
            });
        Gpu {
            device,
            queue,
            config_bind_group_layout,
            input_bind_group_layout,
            output_bind_group_layout,
            color_bind_group_layout,
        }
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
        buf::ReadBuffer::new(&self.device, "read buffer", buf.size()).expect(
            "buf.size should always be a valid size for ReadBuffer::new",
        )
    }

    /// Builds a new readable buffer of a particular size
    ///
    /// If the target buffer is known, use
    /// [`read_buffer_for`](Self::read_buffer_for) to use its size instead
    pub fn read_buffer<T>(
        &self,
        name: &str,
        size: T::S,
    ) -> Result<buf::ReadBuffer<T>, buf::BufferSizeError>
    where
        T: buf::BufferTag,
    {
        buf::ReadBuffer::new(&self.device, name, size)
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
        self.copy(buf, &mut scratch).unwrap();
        let data = self.map(&mut scratch);
        data.to_vec()
    }

    /// Copies from a GPU-resident buffer to a host-mappable buffer
    ///
    /// The buffers must have matching sizes (in bytes)
    pub fn copy<A>(
        &self,
        src: &buf::FlexBuffer<A>,
        dst: &mut buf::ReadBuffer<A>,
    ) -> Result<(), BadCopySize>
    where
        A: buf::BufferTag,
    {
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            },
        );
        self.encode_copy(src, dst, &mut encoder)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Low-level command to submit a buffer copy to a command encoder
    ///
    /// See [`copy`](Self::copy) for details
    pub fn encode_copy<A>(
        &self,
        src: &buf::FlexBuffer<A>,
        dst: &mut buf::ReadBuffer<A>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), BadCopySize>
    where
        A: buf::BufferTag,
    {
        if src.size_bytes() != dst.size_bytes() {
            return Err(BadCopySize {
                src: src.size_bytes(),
                dst: dst.size_bytes(),
            });
        }
        encoder.copy_buffer_to_buffer(
            src.data(),
            0,
            dst.data(),
            0,
            src.size_bytes(),
        );
        Ok(())
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

    /// Builds a new render shape
    pub fn build_render_shape(
        &self,
        shape: &VmShape,
    ) -> Result<RenderShape, RenderShapeError> {
        // Generate bytecode for the root tape
        let bytecode = Bytecode::new(shape.inner().data())?;
        let bytecode_len = bytecode.len() / 2; // u32 -> u64 words
        if bytecode_len > TAPE_DATA_CAPACITY {
            return Err(RenderShapeError::TooLong(bytecode.len() / 2));
        }
        let tape = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tape"),
            size: (bytecode.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        tape.slice(..)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(bytecode.as_bytes());
        tape.unmap();

        let vars = shape.inner().data().vars.clone();
        let vars_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vars"),
            size: (vars.len() * std::mem::size_of::<f32>())
                .next_multiple_of(16)
                .max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vars_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vars"),
                layout: &self.input_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vars_buf.as_entire_binding(),
                }],
            });

        Ok(RenderShape {
            tape,
            vars_buf,
            reg_count: bytecode.reg_count(),
            bytecode_len: bytecode_len.try_into().unwrap(),
            vars,
            vars_bind_group,
        })
    }

    /// Builds a new set of render colors
    pub fn build_render_color(
        &self,
        colors: &[color::ShapeColor<VmShape>],
    ) -> Result<color::RenderColor, color::RenderColorError> {
        color::RenderColor::new(
            colors,
            &self.device,
            &self.color_bind_group_layout,
        )
    }

    fn config_workspace<T>(&self, name: &str) -> ConfigWorkspace<T>
    where
        T: FromBytes + Immutable + IntoBytes + KnownLayout,
    {
        let config = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{name} config buffer")),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{name} config bind group")),
                layout: &self.config_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config.as_entire_binding(),
                }],
            });
        ConfigWorkspace {
            config,
            bind_group,
            _t: std::marker::PhantomData,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Generic type to hold a configuration buffer and bind group
pub struct ConfigWorkspace<T> {
    /// Buffer sized to hold a `T`
    config: wgpu::Buffer,
    /// Bind group which just binds `config`
    bind_group: wgpu::BindGroup,
    /// Marker for `T`
    _t: std::marker::PhantomData<T>,
}

impl<T> ConfigWorkspace<T>
where
    T: FromBytes + Immutable + IntoBytes + KnownLayout,
{
    fn encode_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
        data: &T,
    ) {
        let mut writer = staging.write_buffer(
            encoder,
            &self.config,
            0,
            self.config.size().try_into().unwrap(),
        );
        writer.copy_from_slice(data.as_bytes());
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Lazy container of multiple pipelines, parameterized by register count
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

/// Object containing buffers for rendering a shape on the GPU
pub struct RenderShape {
    /// Buffer which stores the bytecode tape on the GPU
    tape: wgpu::Buffer,
    /// Buffer into which we will copy variable values
    vars_buf: wgpu::Buffer,
    /// Bind group which stores just the variable buffer (at index 0)
    vars_bind_group: wgpu::BindGroup,
    /// Number of registers used by the shape bytecode tape
    reg_count: u8,
    /// Length of bytecode (in `u64` words)
    bytecode_len: u64,
    /// Map of variables
    vars: Arc<VarMap>,
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
    /// Helper function to return XYZ variable indices
    fn axes(&self) -> [u32; 3] {
        [Var::X, Var::Y, Var::Z]
            .map(|a| self.vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
    }

    /// Copies variables into a variables buffer
    fn copy_vars(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
        vars: &ShapeVars<f32>,
    ) -> Result<(), MissingVar> {
        encode_copy_vars(encoder, staging, &self.vars, vars, &self.vars_buf)
    }
}

pub(crate) fn encode_copy_vars(
    encoder: &mut wgpu::CommandEncoder,
    staging: &mut wgpu::util::StagingBelt,
    vs: &VarMap,
    vars: &ShapeVars<f32>,
    buf: &wgpu::Buffer,
) -> Result<(), MissingVar> {
    if vs.has_free_vars() {
        // Do an initial pass to check for errors before mapping the buffer
        for (v, _i) in vs.iter() {
            match v {
                Var::X | Var::Y | Var::Z => (),
                Var::V(vi) => {
                    if vars.get(vi).is_none() {
                        return Err(MissingVar { var: vi });
                    };
                }
            }
        }
        if !vs.is_empty() {
            let mut writer = staging.write_buffer(
                encoder,
                buf,
                0,
                ((vs.len() * std::mem::size_of::<f32>()) as u64)
                    .try_into()
                    .unwrap(),
            );
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
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

/// Trait for a compute pipeline context object
///
/// A context object serves three purposes:
///
/// - Stores GPU pipelines for a particular rendering pass
/// - Allows the creation of workspace (size-independent) and buffers
///   (size-dependent) objects
/// - Encodes and submits work to the GPU for evaluation
///
/// Users are expected to create one (of each type of) context object per thread
/// or worker, since GPU resources can't be shared.  These are expensive to
/// create and should not be churned.
pub trait Context {
    /// Fixed-size configuration buffers for evaluation
    ///
    /// One of these is needed for each simultaneous dispatch.  Typically, you
    /// would build one per thread (or per worker) and reuse it.  However, you
    /// can have more than one per thread if you want to dispatch multiple jobs
    /// simultaneously; it's your computer.  Be aware that they may be
    /// be heavy in terms of GPU resources.  In particular, voxel and pixel
    /// workspaces allocate a large arena of GPU memory for bump allocation!
    type Workspace;

    /// Size-dependent buffers for evaluation
    ///
    /// Users will typically create one per simultaneous viewport, and they are
    /// expected to be long-lived (only being deleted if the viewport size
    /// changes).
    ///
    /// These objects are stateful: after they are used in a rendering pass,
    /// they will contain data in a GPU buffer.  The output buffer is typically
    /// accessed with the `output()` function, e.g.
    /// [`voxel::render::Buffers::output`].
    type Buffers;

    /// Error returned by [`Self::buffers`]
    ///
    /// This typically indicates that the buffer size is invalid
    type BuffersError;

    /// Error returned by [`Self::encode`]
    ///
    /// This indicates a mistake in arguments passed to `encode` or `submit`
    /// (e.g. missing variables or mismatched image sizes).
    type EncodeError;

    /// Argument object for [`Self::encode`]
    ///
    /// Arguments must always be present and cannot be default-constructed.
    /// This may either be a single type (e.g. voxel rendering takes a shape as
    /// its argument) or a small `struct` (e.g. shading takes both heightmap and
    /// SSAO buffers).
    type Args<'a>
    where
        Self: 'a;

    /// Settings object for [`Self::encode`]
    ///
    /// Unlike `Args`, settings must be default-constructible
    type Settings<'a>: Default
    where
        Self: 'a;

    /// Size type used in construction of [`Buffers`](Context::Buffers)
    type Size: buf::BufferItemCount;

    /// Builds a new context object
    fn new(gpu: &Gpu) -> Self;

    /// Builds a new workspace object, typically storing configuration buffers
    fn workspace(&self) -> Self::Workspace;

    /// Builds a new size-dependent buffers object, storing viewport buffers
    fn buffers(
        &self,
        size: Self::Size,
    ) -> Result<Self::Buffers, Self::BuffersError>;

    /// Encodes evaluation into a command encoder
    ///
    /// This function **must not** send any work to the GPU queue itself.  It
    /// also **should not** perform any work that requires the `wgpu::Device`
    /// (e.g. creating buffers or bind groups); those should all be created in
    /// advance and passed in through the function arguments.
    fn encode(
        &self,
        args: Self::Args<'_>,
        settings: Self::Settings<'_>,
        workspace: &mut Self::Workspace,
        buffers: &mut Self::Buffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), Self::EncodeError>;

    /// Submits work to a GPU object
    ///
    /// Implementors of the trait should probably use the default `submit`
    /// implementation, which is a thin wrapper around [`Self::encode`].
    fn submit(
        &self,
        args: Self::Args<'_>,
        settings: Self::Settings<'_>,
        workspace: &mut Self::Workspace,
        buffers: &mut Self::Buffers,
    ) -> Result<(), Self::EncodeError> {
        let gpu = self.gpu();
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(gpu.device.clone(), 1024);
        self.encode(
            args,
            settings,
            workspace,
            buffers,
            &mut encoder,
            &mut staging,
        )?;
        staging.finish();
        gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    /// Returns a handle to the `Gpu` object
    ///
    /// This lets `submit` avoid a `Gpu` argument, but is otherwise not expected
    /// to be used.
    #[doc(hidden)]
    fn gpu(&self) -> &Gpu;
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

/// Returns immedately if a `NO_GPU` environment variable is set
#[macro_export]
#[doc(hidden)]
macro_rules! require_gpu {
    () => {
        if std::env::var("NO_GPU").is_ok() {
            return;
        }
    };
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
