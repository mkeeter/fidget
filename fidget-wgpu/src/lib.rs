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
        // as well, but we special-case them in evaluation.
        let vars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vars"),
            size: u64::try_from(std::mem::size_of::<f32>() * vars.len())
                .unwrap(),
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
pub struct ShapeColorBuffers {
    /// Unified [`VarMap`] object
    var_map: VarMap,

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

    /// Lazily-constructed bind group for the vars array
    vars_bind_group: std::cell::OnceCell<wgpu::BindGroup>,
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

impl ShapeColorBuffers {
    fn new(
        colors: &[ShapeColor<VmShape>],
        device: &wgpu::Device,
        config_size: usize,
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
            shape_start.push(bytecode_data.len());
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
        // as well, but we special-case them in evaluation.
        let vars = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vars"),
            size: u64::try_from(std::mem::size_of::<f32>() * var_map.len())
                .unwrap(),
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
            shape_start: shape_start_buf,
            var_map,
            vars,
            vars_bind_group: Default::default(),
            config_bind_group: Default::default(),
            reg_count,
        })
    }

    /// Helper function to return XYZ variable indices
    fn axes(&self) -> [u32; 3] {
        [Var::X, Var::Y, Var::Z]
            .map(|a| self.var_map.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
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
    use fidget_core::{context::Tree, vm::VmShape};

    #[test]
    fn voxel_render_and_merge() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let voxel_ctx = voxel::Context::new(&gpu);
        let effects_ctx = voxel::effects::Context::new(&gpu);

        let size = 128;
        let image_size = voxel::RenderSize::from(size);
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
        let shape = gpu.shape(&VmShape::from(spheres)).unwrap();

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

    #[test]
    fn pixel_render() {
        // We only run in CI if we're on MacOS (because other runners don't have
        // GPUs and will fail to build the context).
        #[cfg(not(target_os = "macos"))]
        if std::env::var("CI").is_ok() {
            return;
        }

        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let pixel_ctx = pixel::Context::new(&gpu);
        let mut buf = pixel_ctx.buffers();

        let (x, y, _z) = Tree::axes();
        let circle = (x.square() + y.square()).sqrt() - Tree::constant(0.5);
        let shape = gpu.shape(&VmShape::from(circle)).unwrap();

        // Test a variety of image sizes for correctness
        for image_size in [
            pixel::RenderSize::new(64, 64),
            pixel::RenderSize::new(128, 64),
            pixel::RenderSize::new(64, 128),
            pixel::RenderSize::new(27, 51),
        ] {
            let mut pixel_out = pixel_ctx.image_buffer();
            pixel_ctx
                .submit(
                    &shape,
                    &mut buf,
                    Some(&mut pixel_out),
                    &pixel::RenderConfig {
                        image_size,
                        world_to_model: nalgebra::Matrix3::identity(),
                        pixel_perfect: false,
                        z: 0.0,
                    },
                )
                .unwrap();
            let img_out = pixel_ctx.map_image(&mut pixel_out).image();
            assert_eq!(img_out.size(), image_size);

            // Basic circle inside/outside check
            let mat = image_size.screen_to_world();
            for j in 0..image_size.height() {
                for i in 0..image_size.width() {
                    let pos = mat.transform_point(&nalgebra::Point2::new(
                        i as f32, j as f32,
                    ));
                    let p = img_out[(j as usize, i as usize)];
                    let r = (pos.x.powi(2) + pos.y.powi(2)).sqrt();
                    if r < 0.5 {
                        assert!(
                            p.inside(),
                            "pixel should be inside at pixel ({i}, {j}) \
                            (pos {pos}) with radius {r}"
                        );
                    } else {
                        assert!(
                            !p.inside(),
                            "pixel should be outside at pixel ({i}, {j}) \
                            (pos {pos}) with radius {r}"
                        );
                    }
                }
            }
        }
    }

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
}
