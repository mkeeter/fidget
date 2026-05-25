//! Shader generation and WGPU-based image rendering
#![warn(missing_docs)]
pub mod voxel;

use heck::ToShoutySnakeCase;

/// Re-export the [`wgpu`] module
pub use wgpu;

/// Returns a set of constant definitions for each opcode
fn opcode_constants() -> String {
    let mut out = String::new();
    for (op, i) in fidget_bytecode::iter_ops() {
        out += &format!("const OP_{}: u32 = {i};\n", op.to_shouty_snake_case());
    }
    out
}

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
