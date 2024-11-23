#[cfg(target_os = "windows")]
use windows::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
    PAGE_EXECUTE_READWRITE,
};

pub struct Mmap {
    /// Pointer to a memory-mapped region, which may be uninitialized
    ptr: *mut std::ffi::c_void,

    /// Total length of the allocation
    capacity: usize,
}

// SAFETY: this is philosophically a `Vec<u8>`, so can be sent to other threads
unsafe impl Send for Mmap {}

impl Default for Mmap {
    fn default() -> Self {
        Self::empty()
    }
}

impl Mmap {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut::<std::ffi::c_void>(),
            capacity: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Builds a new `Mmap` that can hold at least `len` bytes.
    ///
    /// If `len == 0`, this will return an `Mmap` of size `PAGE_SIZE`; for a
    /// empty `Mmap` (which makes no system calls), use `Mmap::empty` instead.
    #[cfg(not(target_os = "windows"))]
    pub fn new(capacity: usize) -> Result<Self, std::io::Error> {
        let capacity = capacity.max(1).next_multiple_of(Self::PAGE_SIZE);

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                Self::MMAP_PROT,
                Self::MMAP_FLAGS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(Self { ptr, capacity })
        }
    }

    #[cfg(target_os = "windows")]
    pub fn new(capacity: usize) -> Result<Self, std::io::Error> {
        let capacity = capacity.max(1).next_multiple_of(Self::PAGE_SIZE);

        let ptr = unsafe {
            VirtualAlloc(
                None,
                capacity,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            )
        };

        if ptr.is_null() {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(Self { ptr, capacity })
        }
    }

    /// Returns the inner pointer
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Mmap {
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
        #[link(name = "c")]
        extern "C" {
            pub fn sys_icache_invalidate(
                start: *const std::ffi::c_void,
                size: libc::size_t,
            );
        }
        unsafe {
            sys_icache_invalidate(self.ptr, size);
        }
    }
}

#[cfg(target_os = "linux")]
impl Mmap {
    pub const MMAP_PROT: i32 =
        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;

    pub const MMAP_FLAGS: i32 = libc::MAP_PRIVATE | libc::MAP_ANON;
    pub const PAGE_SIZE: usize = 4096;

    /// Flushes caches in preparation for evaluation
    ///
    /// The former is a no-op on systems with coherent D/I-caches (i.e. x86).
    pub fn finalize(&self, size: usize) {
        self.flush_cache(size);
    }
}

#[cfg(target_os = "windows")]
impl Mmap {
    pub const PAGE_SIZE: usize = 4096;

    pub fn finalize(&self, size: usize) {
        self.flush_cache(size);
    }
}

#[cfg(not(target_os = "windows"))]
impl Drop for Mmap {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                libc::munmap(self.ptr, self.capacity as libc::size_t);
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for Mmap {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                let _ = VirtualFree(self.ptr, 0, MEM_RELEASE);
            }
        }
    }
}

use crate::jit::WritePermit;

pub struct MmapWriter {
    mmap: Mmap,

    /// Number of bytes that have been initialized
    ///
    /// This value is conservative and assumes that bytes are written in order
    len: usize,

    _permit: WritePermit,
}

impl From<Mmap> for MmapWriter {
    fn from(mmap: Mmap) -> MmapWriter {
        MmapWriter {
            mmap,
            len: 0,
            _permit: WritePermit::new(),
        }
    }
}

impl MmapWriter {
    /// Pushes a byte to the memmap, resizing if necessary
    pub fn push(&mut self, b: u8) {
        if self.len == self.mmap.capacity {
            let mut next = Mmap::new(self.mmap.capacity * 2).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.mmap.ptr,
                    next.ptr,
                    self.len(),
                );
            }
            std::mem::swap(&mut self.mmap, &mut next);
        }
        unsafe {
            *(self.mmap.ptr as *mut u8).add(self.len) = b;
        }
        self.len += 1;
    }

    /// Finalizes the mmap, invalidating the system icache
    pub fn finalize(self) -> Mmap {
        self.mmap.finalize(self.len);
        self.mmap
    }

    /// Returns the number of bytes written
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the written portion of memory as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.mmap.ptr as *mut u8, self.len)
        }
    }

    /// Returns the inner pointer
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.mmap.ptr
    }
}
