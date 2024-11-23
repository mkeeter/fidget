//! Permits for per-thread `W^X` behavior

/// Holding a `WritePermit` allows writes to memory-mapped regions
pub struct WritePermit {
    _marker: std::marker::PhantomData<*const ()>,
}
static_assertions::assert_not_impl_any!(WritePermit: Send);

impl WritePermit {
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(0);
        }

        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl Drop for WritePermit {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(1);
        }
    }
}

#[cfg(target_os = "macos")]
#[link(name = "pthread")]
extern "C" {
    pub fn pthread_jit_write_protect_np(enabled: std::ffi::c_int);
}
