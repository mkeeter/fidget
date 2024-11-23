//! Permits for per-thread `W^X` behavior

use std::cell::Cell;

thread_local! {
    pub static PERMIT_COUNT: Cell<i64> = const { Cell::new(0) };
}

/// Holding a `WritePermit` allows writes to memory-mapped regions
pub struct WritePermit {
    _marker: std::marker::PhantomData<*const ()>,
}
static_assertions::assert_not_impl_any!(WritePermit: Send);

impl WritePermit {
    pub fn new() -> Self {
        let n = PERMIT_COUNT.get();
        if n < 0 {
            panic!("cannot build a WritePermit while ExecPermits are present");
        }
        PERMIT_COUNT.set(n + 1);

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
        let n = PERMIT_COUNT.get();
        assert!(n > 0);

        #[cfg(target_os = "macos")]
        if n == 1 {
            unsafe {
                pthread_jit_write_protect_np(1);
            }
        }

        PERMIT_COUNT.set(n - 1);
    }
}

/// Holding a `ExecutePermit` allows calls to functions in memory-mapped regions
pub struct ExecutePermit {
    _marker: std::marker::PhantomData<*const ()>,
}
static_assertions::assert_not_impl_any!(ExecutePermit: Send);

impl ExecutePermit {
    pub fn new() -> Self {
        let n = PERMIT_COUNT.get();
        if n > 0 {
            panic!("cannot build a ExecPermits while WritePermit are present");
        }
        PERMIT_COUNT.set(n - 1);
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(target_os = "macos")]
#[link(name = "pthread")]
extern "C" {
    pub fn pthread_jit_write_protect_np(enabled: std::ffi::c_int);
}
