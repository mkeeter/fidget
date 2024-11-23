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
            panic!("permit underflow?");
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
///
/// An `ExecutePermit` can only be constructed if no `WritePermits` are extant;
/// the constructor will panic otherwise.
///
/// Note that this isn't 100% foolproof: it's possible to create an
/// `ExecutePermit`, _then_ create a `WritePermit`, making the `ExecutePermit`
/// invalid.  However, this shouldn't be possible from the library's public API.
///
/// (It's possible to fix this by making `ExecutePermit` subtract from a signed
/// `PERMIT_COUNT`, but then you can never build tapes while an evaluator is
/// live on that same thread, which is a heavy restriction)
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
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(target_os = "macos")]
#[link(name = "pthread")]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: std::ffi::c_int);
}
