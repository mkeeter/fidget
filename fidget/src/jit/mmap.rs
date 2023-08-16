pub struct Mmap {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: this is philosophically a `Vec<u8>`, so can be sent to other threads
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Default for Mmap {
    fn default() -> Self {
        Self::empty()
    }
}

impl Mmap {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut::<libc::c_void>(),
            len: 0,
        }
    }

    /// Builds a new `Mmap` that can hold at least `len` bytes.
    ///
    /// If `len == 0`, this will return an `Mmap` of size `PAGE_SIZE`; for a
    /// empty `Mmap` (which makes no system calls), use `Mmap::empty` instead.
    pub fn new(len: usize) -> Result<Self, std::io::Error> {
        let len = (len.max(1) + Self::PAGE_SIZE - 1) / Self::PAGE_SIZE
            * Self::PAGE_SIZE;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                Self::MMAP_PROT,
                Self::MMAP_FLAGS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(Self { ptr, len })
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    /// Writes to the given offset in the memory map
    ///
    /// # Panics
    /// If `index >= self.len`
    #[inline(always)]
    pub fn write(&mut self, index: usize, byte: u8) {
        assert!(index < self.len);
        unsafe {
            *(self.ptr as *mut u8).add(index) = byte;
        }
    }

    /// Treats the memory-mapped data as a slice
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// Returns the inner pointer
    pub fn as_ptr(&self) -> *mut libc::c_void {
        self.ptr
    }
}

#[cfg(target_os = "macos")]
impl Mmap {
    pub const MMAP_PROT: i32 =
        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
    pub const MMAP_FLAGS: i32 =
        libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;
    pub const PAGE_SIZE: usize = 4096;

    /// Invalidates the caches for the first `size` bytes of the mmap
    ///
    /// Note that you will still need to change the global W^X mode before
    /// evaluation, but that's on a per-thread (rather than per-mmap) basis.
    pub fn finalize(&self, size: usize) {
        unsafe {
            macos::sys_icache_invalidate(self.ptr, size);
        }
    }

    /// Modifies the **per-thread** W^X state to allow writing of memory-mapped
    /// regions.
    ///
    /// The fact that this occurs on a per-thread (rather than per-page) basis
    /// is _very strange_, and means this APIs must be used with caution.
    /// Returns a [`macos::ThreadWriteGuard`], which restores execute mode when
    /// dropped.
    pub fn thread_mode_write() -> macos::ThreadWriteGuard {
        unsafe {
            macos::pthread_jit_write_protect_np(0);
        }
        macos::ThreadWriteGuard
    }

    /// Modifies the region's W^X state to allow writing
    ///
    /// This is only relevant on Linux and is a no-op on MacOS
    pub fn make_write(&self) {
        // Nothing to do here
    }
}

#[cfg(target_os = "linux")]
impl Mmap {
    #[cfg(feature = "write-xor-execute")]
    pub const MMAP_PROT: i32 = libc::PROT_READ | libc::PROT_WRITE;

    #[cfg(not(feature = "write-xor-execute"))]
    pub const MMAP_PROT: i32 =
        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;

    pub const MMAP_FLAGS: i32 = libc::MAP_PRIVATE | libc::MAP_ANON;
    pub const PAGE_SIZE: usize = 4096;

    /// Flushes caches and remaps to an executable binding
    ///
    /// The former is a no-op on systems with coherent D/I-caches (i.e. x86);
    /// the latter is a no-op if the `write-xor-execute` feature is not enabled.
    pub fn finalize(&self, size: usize) {
        self.flush_cache(size);

        // This is deliberately done as a cfg! conditional (instead of #[cfg]),
        // so that the code is type-checked even if the feature is disabled.
        if cfg!(feature = "write-xor-execute") {
            unsafe {
                libc::mprotect(
                    self.ptr,
                    size,
                    libc::PROT_READ | libc::PROT_EXEC,
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn flush_cache(&self, size: usize) {
        use std::arch::asm;
        let mut cache_type: usize;
        // Loosely based on code from mono; see mono/mono#3549 for a good
        // writeup and associated PR.
        unsafe {
            asm!(
                "mrs {tmp}, ctr_el0",
                tmp = out(reg) cache_type,
            );
        }
        let icache_line_size = (cache_type & 0xF) << 4;
        let dcache_line_size = ((cache_type >> 16) & 0xF) << 4;

        let mut addr = self.as_ptr() as usize & !(dcache_line_size - 1);
        let end = self.as_ptr() as usize + size;
        while addr < end {
            unsafe {
                asm!(
                    "dc civac, {tmp}",
                    tmp = in(reg) addr,
                );
            }
            addr += dcache_line_size;
        }
        unsafe {
            asm!("dsb ish");
        }

        let mut addr = self.as_ptr() as usize & !(icache_line_size - 1);
        while addr < end {
            unsafe {
                asm!(
                    "ic ivau, {tmp}",
                    tmp = in(reg) addr,
                );
            }
            addr += icache_line_size;
        }
        unsafe {
            asm!("dsb ish", "isb");
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn flush_cache(&self, _size: usize) {
        // Nothing to do here
    }

    /// Modifies the **per-thread** W^X state to allow writing of memory-mapped
    /// regions.
    ///
    /// This is only relevant on macOS and is a no-op on Linux.
    pub fn thread_mode_write() {
        // Nothing to do here
    }

    /// Modifies the region's W^X state to allow writing
    ///
    /// This is a no-op if the `write-xor-execute` feature is not enabled.
    pub fn make_write(&self) {
        if cfg!(feature = "write-xor-execute") {
            unsafe {
                libc::mprotect(
                    self.ptr,
                    self.len,
                    libc::PROT_READ | libc::PROT_WRITE,
                );
            }
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                libc::munmap(self.ptr, self.len as libc::size_t);
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    /// Empty struct which switches the thread to execute mode when dropped
    pub struct ThreadWriteGuard;
    impl Drop for ThreadWriteGuard {
        fn drop(&mut self) {
            unsafe {
                pthread_jit_write_protect_np(1);
            }
        }
    }

    #[link(name = "pthread")]
    extern "C" {
        pub fn pthread_jit_write_protect_np(enabled: libc::c_int);
    }

    #[link(name = "c")]
    extern "C" {
        pub fn sys_icache_invalidate(
            start: *const std::ffi::c_void,
            size: libc::size_t,
        );
    }
}
