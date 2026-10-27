//! Data types for evaluating shape color
//!
//! In this module, only [`ShapeColor`] can be constructed directly.
//! [`RenderColor`] is constructed by
//! [`Gpu::build_render_color`](crate::Gpu::build_render_color).
use crate::encode_copy_vars;
use fidget_bytecode::{Bytecode, ReservedRegister};
use fidget_core::{
    eval::Function,
    shape::{MissingVar, ShapeVars},
    var::{Var, VarMap},
    vm::VmShape,
};

use std::{collections::HashMap, sync::Arc};
use zerocopy::IntoBytes;

/// Error type when constructing a [`RenderColor`]
#[derive(Debug, thiserror::Error)]
pub enum RenderColorError {
    /// The shape uses a reserved register
    #[error(transparent)]
    RegisterError(#[from] ReservedRegister),
}

/// GPU color buffers for rendering a shape's diffuse color
pub struct RenderColor {
    /// Unified [`VarMap`] object
    vars: Arc<VarMap>,
    /// Number of shapes available
    shape_count: usize,
    /// Buffer for storing variable values
    vars_buf: wgpu::Buffer,
    /// Maximum number of registers used by any tape
    reg_count: u8,
    /// Bind group which binds the tape, starts, and variable buffers
    bind_group: wgpu::BindGroup,
}

// Accessor methods
impl RenderColor {
    pub(crate) fn shape_count(&self) -> usize {
        self.shape_count
    }
    pub(crate) fn reg_count(&self) -> u8 {
        self.reg_count
    }
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Copies variables into a variables buffer
    pub(crate) fn copy_vars(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
        vars: &ShapeVars<f32>,
    ) -> Result<(), MissingVar> {
        encode_copy_vars(encoder, staging, &self.vars, vars, &self.vars_buf)
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

impl RenderColor {
    /// Builds a new set of buffers for evaluating shape color
    pub fn new(
        colors: &[ShapeColor<VmShape>],
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, RenderColorError> {
        // Build a single unified variable map, used across all tapes
        let mut vars = VarMap::new();
        for c in colors {
            for channel in c.channels() {
                let vs = channel.inner().vars();
                for (v, _index) in vs.iter() {
                    vars.insert(v);
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
                            u32::try_from(vars.get(&v).unwrap()).unwrap(),
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

        let shape_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape starts"),
            usage: wgpu::BufferUsages::STORAGE,
            size: (shape_start.len() * std::mem::size_of::<u32>()).max(16)
                as u64,
            mapped_at_creation: true,
        });
        shape_start_buf
            .slice(..)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(shape_start.as_bytes());
        shape_start_buf.unmap();

        let vars_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vars"),
            size: (vars.len() * std::mem::size_of::<f32>())
                .next_multiple_of(16)
                .max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tape = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tape"),
            size: (bytecode_data.len() * std::mem::size_of::<u32>()).max(16)
                as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        tape.slice(..)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(bytecode_data.as_bytes());
        tape.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shape color"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tape.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shape_start_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vars_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            shape_count: colors.len(),
            vars: vars.into(),
            vars_buf,
            reg_count,
            bind_group,
        })
    }

    /// Helper function to return XYZ variable indices
    pub(crate) fn axes(&self) -> [u32; 3] {
        [Var::X, Var::Y, Var::Z]
            .map(|a| self.vars.get(&a).map(|v| v as u32).unwrap_or(u32::MAX))
    }
}
