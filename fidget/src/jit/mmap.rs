const PAGE_SIZE: usize = 4096;

pub struct Mmap {
    ptr: *mut libc::c_void,
    len: usize,
}

impl Mmap {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut::<libc::c_void>(),
            len: 0,
        }
    }

    pub fn new(len: usize) -> Result<Self, std::io::Error> {
        let len = (len + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
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

    pub fn len(&self) -> usize {
        self.len
    }

    /// Copies the given slice into the memory-mapped region
    ///
    /// Unlike [`std::slice::copy_from_slice`], this function will allow cases
    /// where `self.len > s.len()`, and will simply copy into a prefix of the
    /// region.
    pub fn copy_from_slice(&mut self, s: &[u8]) {
        unsafe {
            pthread_jit_write_protect_np(0);
            let slice =
                std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len);
            slice[0..s.len()].copy_from_slice(s);
            slice[0..s.len()].reverse();
            slice[0..s.len()].reverse();

            sys_icache_invalidate(self.ptr, s.len());
            pthread_jit_write_protect_np(1);
        }
    }

    pub fn as_ptr(&self) -> *mut libc::c_void {
        self.ptr
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
