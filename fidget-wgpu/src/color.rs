//! Data types for evaluating shape color
//!
//! In this module, only [`ShapeColor`] and [`ShapeColorBuffers`] should be used
//! directly (through direct construction and [`ShapeColorBuffers::new`]
//! respectively).  Other types are generic and are re-exported with a specific
//! paramerization in an `effects` module; e.g. the generic
//! [`ColorWorkspace<C>`] is specialized to
//! [`voxel::effects::ColorWorkspace`](crate::voxel::effects::ColorWorkspace)
use crate::{
    CopyVarsChanged, CopyVarsError, Gpu,
    buf::{BufferSizeError, FlexBuffer, FlexConfigBuffer},
    tag,
    voxel::VarsBufferTag,
};
use fidget_bytecode::{Bytecode, ReservedRegister};
use fidget_core::{
    eval::Function,
    shape::ShapeVars,
    var::{Var, VarMap},
    vm::VmShape,
};

use std::collections::HashMap;
use zerocopy::IntoBytes;

tag!(ShapeStartBufferTag, u32, usize, STORAGE | COPY_DST);

/// Generic workspace for evaluating diffuse color
pub struct ColorWorkspace<C> {
    /// Config and serialized tapes, appended end to end
    config: FlexConfigBuffer<C, u32>,

    /// Start of the channel tapes for each shape
    ///
    /// The high bit indicates whether this is RGB (0) or HSL (1)
    shape_start: FlexBuffer<ShapeStartBufferTag>,

    /// Scratch space to upload variable values
    vars_buf: FlexBuffer<VarsBufferTag>,

    /// Lazily-constructed bind group for the config buffers
    bind_group: std::cell::OnceCell<wgpu::BindGroup>,
}

impl<C> ColorWorkspace<C>
where
    C: zerocopy::IntoBytes + zerocopy::Immutable + Copy,
{
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let config =
            FlexBuffer::new(device, "color config".to_owned(), 4.into())
                .unwrap();
        let shape_start =
            FlexBuffer::new(device, "shape start".to_owned(), 4usize).unwrap();
        let vars_buf =
            FlexBuffer::new(device, "color vars".to_owned(), 4usize).unwrap();
        Self {
            config,
            shape_start,
            vars_buf,
            bind_group: Default::default(),
        }
    }

    pub(crate) fn config_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> &wgpu::BindGroup {
        self.bind_group.get_or_init(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("config bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.config.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.shape_start.bind_active(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.vars_buf.bind_active(),
                    },
                ],
            })
        })
    }

    pub(crate) fn copy_vars(
        &mut self,
        gpu: &Gpu,
        vs: &VarMap,
        vars: &ShapeVars<f32>,
    ) -> Result<(), CopyVarsError> {
        match crate::copy_vars(gpu, vs, vars, &mut self.vars_buf)? {
            CopyVarsChanged::BufferChanged => {
                self.bind_group = Default::default()
            }
            CopyVarsChanged::BufferUnchanged => (),
        }
        Ok(())
    }

    pub(crate) fn copy_tape(
        &mut self,
        gpu: &Gpu,
        bytecode: &[u32],
    ) -> Result<(), BufferSizeError> {
        let r = self
            .config
            .grow_to_fit(&gpu.device, bytecode.len().into())?;
        if !matches!(r, std::cmp::Ordering::Equal) {
            self.bind_group = Default::default();
        }
        let Ok(byte_count) =
            (std::mem::size_of_val(bytecode) as u64).try_into()
        else {
            return Ok(());
        };
        let mut writer = gpu
            .queue
            .write_buffer_with(
                self.config.data(),
                std::mem::size_of::<C>() as u64,
                byte_count,
            )
            .unwrap();
        writer.copy_from_slice(bytecode.as_bytes());
        Ok(())
    }

    pub(crate) fn copy_shape_starts(
        &mut self,
        gpu: &Gpu,
        shape_starts: &[u32],
    ) -> Result<(), BufferSizeError> {
        let r = self
            .shape_start
            .grow_to_fit(&gpu.device, shape_starts.len())?;
        if !matches!(r, std::cmp::Ordering::Equal) {
            self.bind_group = Default::default();
        }
        let Ok(byte_count) =
            (std::mem::size_of_val(shape_starts) as u64).try_into()
        else {
            return Ok(());
        };
        let mut writer = gpu
            .queue
            .write_buffer_with(self.shape_start.data(), 0, byte_count)
            .unwrap();
        writer.copy_from_slice(shape_starts.as_bytes());
        Ok(())
    }

    pub(crate) fn copy_config(&mut self, gpu: &Gpu, config: &C) {
        self.config.write_config(config, &gpu.queue)
    }
}

/// Error type when constructing a [`ShapeColorBuffers`]
#[derive(Debug, thiserror::Error)]
pub enum ShapeColorError {
    /// The shape uses a reserved register
    #[error(transparent)]
    RegisterError(#[from] ReservedRegister),
}

/// Color buffers for rendering a shape's diffuse color
///
/// These buffers live purely on the CPU, so they can be shared between threads.
/// The object may be heavy (and therefore does not implement `Clone`); to share
/// it, wrap it in an `Arc` or similar.
pub struct ShapeColorBuffers {
    /// Unified [`VarMap`] object
    var_map: VarMap,

    /// Number of shapes available
    shape_count: usize,

    /// Index of each shape's start position in the bytecode data
    shape_start: Vec<u32>,

    /// Maximum number of registers used by any tape
    reg_count: u8,

    /// Raw bytecode, which should be copied to the GPU config buffer tail
    bytecode: Vec<u32>,
}

// Accessor methods
impl ShapeColorBuffers {
    pub(crate) fn var_map(&self) -> &VarMap {
        &self.var_map
    }
    pub(crate) fn shape_count(&self) -> usize {
        self.shape_count
    }
    pub(crate) fn shape_start(&self) -> &[u32] {
        &self.shape_start
    }
    pub(crate) fn reg_count(&self) -> u8 {
        self.reg_count
    }
    pub(crate) fn bytecode(&self) -> &[u32] {
        &self.bytecode
    }
}

/// Generic shape color
pub enum ShapeColor<T> {
    /// Red / green / blue channels, in the 0-1 range
    Rgb {
        /// Red component
        r: T,
        /// Green component
        g: T,
        /// Blue component
        b: T,
    },
    /// Hue / saturation / lightness channels, in the 0-1 range
    Hsl {
        /// Hue
        h: T,
        /// Saturation
        s: T,
        /// Lightness
        l: T,
    },
}

impl<T> ShapeColor<T> {
    fn channels(&self) -> [&T; 3] {
        match self {
            ShapeColor::Rgb { r, g, b } => [r, g, b],
            ShapeColor::Hsl { h, s, l } => [h, s, l],
        }
    }
}

impl ShapeColorBuffers {
    /// Builds a new set of buffers for evaluating shape color
    pub fn new(
        colors: &[ShapeColor<VmShape>],
    ) -> Result<Self, ShapeColorError> {
        // Build a single unified variable map, used across all tapes
        let mut var_map = VarMap::new();
        for c in colors {
            for channel in c.channels() {
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
            // Divide by 2 to convert from `u32` to `TapeWord`
            let kind = match c {
                ShapeColor::Rgb { .. } => 0,
                ShapeColor::Hsl { .. } => 1 << 31,
            };
            let index = u32::try_from(bytecode_data.len() / 2).unwrap();
            assert!(
                index & (1 << 31) == 0,
                "you have built more than 2 GiB of shape tapes?!"
            );
            shape_start.push(index | kind);
            for channel in c.channels() {
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

        Ok(Self {
            shape_count: colors.len(),
            shape_start,
            bytecode: bytecode_data,
            var_map,
            reg_count,
        })
    }

    /// Helper function to return XYZ variable indices
    pub(crate) fn axes(&self) -> [u32; 3] {
        [Var::X, Var::Y, Var::Z]
            .map(|a| self.var_map.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
    }
}
