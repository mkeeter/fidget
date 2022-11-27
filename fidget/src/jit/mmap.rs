const PAGE_SIZE: usize = 4096;

pub struct Mmap {
    ptr: *mut libc::c_void,
    len: usize,
}

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
        let len = (len.max(1) + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
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

    /// Invalidates the caches for the first `size` bytes of the mmap
    ///
    /// Note that you will still need to change the global W^X mode before
    /// evaluation, but that's on a per-thread (rather than per-mmap) basis.
    pub fn flush(&self, size: usize) {
        unsafe {
            sys_icache_invalidate(self.ptr, size);
        }
    }

    /// Modifies the **per-thread** W^X state to allow writing of memory-mapped
    /// regions.
    ///
    /// The fact that this occurs on a per-thread (rather than per-page) basis
    /// is _very strange_, and means this APIs must be used with caution.
    /// Returns a `WriteGuard`, which restores execute mode when dropped.
    pub fn thread_mode_write() -> WriteGuard {
        unsafe {
            pthread_jit_write_protect_np(0);
        }
        WriteGuard
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

pub struct WriteGuard;
impl Drop for WriteGuard {
    fn drop(&mut self) {
        unsafe {
            pthread_jit_write_protect_np(1);
        }
    }
}

#[link(name = "pthread")]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
}

#[link(name = "c")]
extern "C" {
    fn sys_icache_invalidate(
        start: *const std::ffi::c_void,
        size: libc::size_t,
    );
}
