//! Custom types used during evaluation

mod float;
mod grad;
mod interval;
pub use float::FloatExt;
pub use grad::Grad;
pub use interval::Interval;
