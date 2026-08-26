//! Shader generation and WGPU-based image rendering
#![warn(missing_docs)]

use fidget_bytecode::{Bytecode, ReservedRegister};
use fidget_core::{
    eval::Function,
    var::{Var, VarMap},
    vm::VmShape,
};

use heck::ToShoutySnakeCase;
use std::collections::{BTreeMap, HashMap};
use zerocopy::{FromBytes, Immutable, IntoBytes};

pub mod buf;
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

    /// Returns a readable buffer for the given image buffer
    pub fn read_buffer_for<
        T: buf::BufferTag,
        B: buf::BufferItemCount + Copy + Into<fidget_core::render::ImageSize>,
    >(
        &self,
        buf: &buf::GenericFlexBuffer<T, B>,
    ) -> buf::ImageReadBuffer<T> {
        buf::ImageBuffer::new(
            &self.device,
            format!("{} (read)", buf.name()),
            buf.size().into(),
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

    /// Builds a new [`RenderShape`] object for the given shape
    pub fn shape(
        &self,
        shape: &VmShape,
    ) -> Result<RenderShape, RenderShapeError> {
        RenderShape::new(shape, &self.device)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Container of multiple pipelines, parameterized by register count
pub(crate) struct RegPipeline(BTreeMap<u8, wgpu::ComputePipeline>);

impl RegPipeline {
    pub fn build<F: Fn(u8) -> wgpu::ComputePipeline>(builder: F) -> Self {
        let mut out = BTreeMap::new();
        for reg_count in [8, 16, 32, 64, 128, 192, 255] {
            out.insert(reg_count, builder(reg_count));
        }
        Self(out)
    }

    /// Returns the pipeline with sufficient registers to render `reg_count`
    ///
    /// # Panics
    /// If `reg_count` is 256 (which is not allowed in bytecode tapes)
    pub fn get(&self, reg_count: u8) -> &wgpu::ComputePipeline {
        let (r, v) = self
            .0
            .range(reg_count..)
            .next()
            .expect("bytecode tape cannot use more than 255 registers");
        assert!(*r >= reg_count);
        v
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Shape for rendering
///
/// This object is constructed by [`Gpu::shape`] and may only be used with
/// that particular [`Gpu`].
pub struct RenderShape {
    /// Copy of our shape (kept around for access to the variable map)
    shape: VmShape,
    /// Serialized bytecode for the shape
    bytecode: Bytecode,
    /// GPU buffer to contain variables
    ///
    /// This doesn't live in a `Buffers` object because it's dynamically sized
    /// based on the shape; everything in `Buffers` is based on image size.
    vars: wgpu::Buffer,
    /// Lazily-constructed bind group for the vars array
    ///
    /// This is not cached in a buffer-specific `BindGroups` object because it
    /// is shape-specific.
    vars_bind_group: std::cell::OnceCell<wgpu::BindGroup>,
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

/// Error type when constructing a [`ShapeColorBuffers`]
#[derive(Debug, thiserror::Error)]
pub enum ShapeColorError {
    /// The shape uses a reserved register
    #[error(transparent)]
    RegisterError(#[from] ReservedRegister),
}

impl RenderShape {
    fn new(
        shape: &VmShape,
        device: &wgpu::Device,
    ) -> Result<Self, RenderShapeError> {
        // Generate bytecode for the root tape
        let bytecode = Bytecode::new(shape.inner().data())?;
        if bytecode.len() / 2 > TAPE_DATA_CAPACITY {
            return Err(RenderShapeError::TooLong(bytecode.len() / 2));
        }

        let vars = shape.inner().vars();

        // Build a buffer for non-XYZ vars.  This buffer includes slots for XYZ
        // as well, but we special-case them in evaluation.  If the tape has no
        // variables, we'll allocate 4 bytes (because empty buffers are not
        // allowed).
        let vars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vars"),
            size: u64::try_from(std::mem::size_of::<f32>() * vars.len())
                .unwrap()
                .max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            shape: shape.clone(),
            bytecode,
            vars,
            vars_bind_group: Default::default(),
        })
    }

    /// Helper function to return XYZ variable indices
    fn axes(&self) -> [u32; 3] {
        let vars = self.shape.inner().vars();
        [Var::X, Var::Y, Var::Z]
            .map(|a| vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
    }

    fn vars_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> &wgpu::BindGroup {
        self.vars_bind_group.get_or_init(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vars bind group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.vars.as_entire_binding(),
                }],
            })
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Color buffers for rendering a shape's diffuse color
pub struct ShapeColorBuffers<C> {
    /// Unified [`VarMap`] object
    var_map: VarMap,

    /// Number of shapes available
    shape_count: usize,

    /// Maximum number of registers used by any tape
    reg_count: u8,

    /// Config and serialized tapes, all squished together
    ///
    /// The tape data is baked once (at construction); config data is edited
    /// before each evaluation
    config: wgpu::Buffer,

    /// Start of the RGB tapes for each shape
    ///
    /// This is baked once (at construction)
    shape_start: wgpu::Buffer,

    /// Lazily-constructed bind group for the config buffers
    config_bind_group: std::cell::OnceCell<wgpu::BindGroup>,

    /// GPU buffer to contain variables (passed in during evaluation)
    ///
    /// This doesn't live in a `Buffers` object because it's dynamically sized
    /// based on the shape; everything in `Buffers` is based on image size.
    vars: wgpu::Buffer,

    // Marker for the config type
    _config: std::marker::PhantomData<C>,
}

/// Generic shape color generator
pub enum ShapeColor<T> {
    /// Red / green / blue channels
    Rgb {
        /// Red component
        r: T,
        /// Green component
        g: T,
        /// Blue component
        b: T,
    },
}

impl<C> ShapeColorBuffers<C> {
    fn new(
        colors: &[ShapeColor<VmShape>],
        device: &wgpu::Device,
    ) -> Result<Self, ShapeColorError> {
        // Build a single unified variable map, used across all tapes
        let mut var_map = VarMap::new();
        for c in colors {
            let ShapeColor::Rgb { r, g, b } = c;
            for channel in [r, g, b] {
                let vars = channel.inner().vars();
                for (v, _index) in vars.iter() {
                    var_map.insert(v);
                }
            }
        }
        let mut reg_count = 0;
        let mut shape_start = Vec::with_capacity(colors.len());
        let mut bytecode_data: Vec<u32> = Vec::new();
        let mut local_var_map = HashMap::new();
        for c in colors {
            let ShapeColor::Rgb { r, g, b } = c;
            // Divide by 2 to convert from `u32` to `TapeWord`
            shape_start.push(u32::try_from(bytecode_data.len() / 2).unwrap());
            for channel in [r, g, b] {
                // Build a local variable remapping array, reusing allocations
                local_var_map.clear();
                local_var_map.extend(channel.inner().vars().iter().map(
                    |(v, i)| {
                        (
                            u32::try_from(i).unwrap(),
                            u32::try_from(var_map.get(&v).unwrap()).unwrap(),
                        )
                    },
                ));

                // Generate bytecode for the root tape
                let bytecode = Bytecode::new_with_input_map(
                    channel.inner().data(),
                    &local_var_map,
                )?;
                bytecode_data.extend(bytecode.data());
                reg_count = reg_count.max(bytecode.reg_count());
            }
        }

        // Build a buffer for non-XYZ vars.  This buffer includes slots for XYZ
        // as well, but we special-case them in evaluation.  If the tape has no
        // variables, then we'll allocate 4 bytes (because empty buffers aren't
        // allowed).
        let vars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vars"),
            size: u64::try_from(std::mem::size_of::<f32>() * var_map.len())
                .unwrap()
                .max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shape_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape_start"),
            size: u64::try_from(std::mem::size_of::<u32>() * shape_start.len())
                .unwrap(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        shape_start_buf
            .get_mapped_range_mut(0..)
            .copy_from_slice(shape_start.as_bytes());
        shape_start_buf.unmap();

        let config_size = std::mem::size_of::<C>();
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape_start"),
            size: u64::try_from(
                std::mem::size_of::<u32>() * bytecode_data.len() + config_size,
            )
            .unwrap(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        config_buf
            .get_mapped_range_mut(config_size as u64..)
            .copy_from_slice(bytecode_data.as_bytes());
        config_buf.unmap();

        Ok(Self {
            config: config_buf,
            shape_count: colors.len(),
            shape_start: shape_start_buf,
            var_map,
            vars,
            config_bind_group: Default::default(),
            reg_count,
            _config: std::marker::PhantomData,
        })
    }

    /// Helper function to return XYZ variable indices
    fn axes(&self) -> [u32; 3] {
        [Var::X, Var::Y, Var::Z]
            .map(|a| self.var_map.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
    }

    fn config_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> &wgpu::BindGroup {
        self.config_bind_group.get_or_init(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("config bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.shape_start.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.vars.as_entire_binding(),
                    },
                ],
            })
        })
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
            assert_eq!(dynamic_array_offset as usize, std::mem::size_of::<T>());
        } else {
            assert_eq!(span as usize, std::mem::size_of::<T>());
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
