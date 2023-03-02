/// We use `xmm4-16` (all caller-saved) for graph variables
pub const REGISTER_LIMIT: u8 = 12;
/// `xmm0` is used for immediates
pub const IMM_REG: u8 = 0;
/// `xmm1-3` are available for use as temporaries.
pub const OFFSET: u8 = 4;

pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;
