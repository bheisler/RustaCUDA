use super::DeviceCopy;
use error::*;
use memory::{cuda_free_locked, cuda_malloc_locked};
use std::mem;
use std::ops;
use std::ptr;
use std::slice;

/*
You should be able to:
- Allocate and deallocate a locked buffer
- Arbitrarily read/write to the elements of that buffer
- Copy data to/from a DeviceBuffer
- Slice the locked buffer
*/

/// Fixed-size host-side buffer in page-locked memory.
///
/// ## Page-locked Memory
///
/// When transferring data to a CUDA device, the driver must first copy the data to a special memory
/// page which is locked to physical memory. Once complete, it can initiate a DMA transfer to copy the
/// data from physical memory to device memory. For some applications, it may be possible to keep
/// the data in page-locked memory, skipping the first step. This can substantially improve data
/// transfer speeds.
///
/// The downside is that using excessive amounts of page-locked memory can degrade system performance
/// by reducing the amount of physical memory available to the system for paging. As a result, it is
/// best used sparingly, to allocate staging areas for data exchange between the host and device.
///
/// The CUDA driver tracks memory regions allocated this way and will automatically optimize
/// memory transfers to or from the device.
#[derive(Debug)]
pub struct LockedBuffer<T: DeviceCopy> {
    buf: *mut T,
    capacity: usize,
}
impl<T: DeviceCopy> LockedBuffer<T> {
    /// Create a new LockedBuffer of length `size` filled with clones of `value`.
    pub fn new(value: &T, size: usize) -> CudaResult<Self> {
        unsafe {
            let mut uninit = LockedBuffer::uninitialized(size)?;
            for x in 0..size {
                *uninit.get_unchecked_mut(x) = value.clone();
            }
            Ok(uninit)
        }
    }

    /// Create a new LockedBuffer with the same size as the given slice, and clone the values into
    /// it.
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = LockedBuffer::uninitialized(slice.len())?;
            for (i, x) in slice.iter().enumerate() {
                *uninit.get_unchecked_mut(i) = x.clone();
            }
            Ok(uninit)
        }
    }

    /// Create a new LockedBuffer without initializing the allocated memory. The caller is
    /// responsible for ensuring that the allocated buffer is initialized.
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let bytes = size
            .checked_mul(mem::size_of::<T>())
            .ok_or(CudaError::InvalidMemoryAllocation)?;

        let ptr: *mut T = if bytes > 0 {
            cuda_malloc_locked(bytes)?
        } else {
            ptr::NonNull::dangling().as_ptr()
        };
        Ok(LockedBuffer {
            buf: ptr as *mut T,
            capacity: size,
        })
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Creates a `LockedBuffer<T>` directly from the raw components of another locked buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `LockedBuffer<T>`
    ///   (at least, it's highly likely to be incorrect if it wasn't).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `LockedBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    pub unsafe fn from_raw_parts(ptr: *mut T, size: usize) -> LockedBuffer<T> {
        LockedBuffer {
            buf: ptr,
            capacity: size,
        }
    }

    // Drop
}

impl<T: DeviceCopy> AsRef<LockedBuffer<T>> for LockedBuffer<T> {
    fn as_ref(&self) -> &LockedBuffer<T> {
        self
    }
}
impl<T: DeviceCopy> AsMut<LockedBuffer<T>> for LockedBuffer<T> {
    fn as_mut(&mut self) -> &mut LockedBuffer<T> {
        self
    }
}
impl<T: DeviceCopy> AsRef<[T]> for LockedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T: DeviceCopy> AsMut<[T]> for LockedBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: DeviceCopy> ops::Deref for LockedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf;
            slice::from_raw_parts(p, self.capacity)
        }
    }
}
impl<T: DeviceCopy> ops::DerefMut for LockedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf;
            slice::from_raw_parts_mut(ptr, self.capacity)
        }
    }
}
impl<T: DeviceCopy> Drop for LockedBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            unsafe {
                cuda_free_locked(self.buf).expect("Failed to deallocate CUDA page-locked memory.");
            }
        }
        self.capacity = 0;
    }
}
