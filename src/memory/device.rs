use memory::DevicePointer;
use memory::DeviceCopy;
use memory::{cuda_malloc, cuda_free};
use error::{CudaResult, CudaError};
use std::mem;
use std::ptr;
use std::ops;
use std::os::raw::c_void;
use cuda_sys::cudart::{
    cudaMemset,
};

/*
TODO:
You should be able to:
- Allocate and deallocate a device buffer
- Copy slices to and from that buffer
- Slice the device buffer into device-slices
- Copy slices to and from device-slices
- length, is_empty of slices and buffer
- memset slices or buffers
    - Should this be gated by some trait? Or just unsafe?
- Split slices just like with regular slices
- copy_host_to_device, copy_device_to_host, copy_device_to_device? Verbose...
- device_slice.copy_from(slice), device_slice.copy_to(mut slice)
    - Is it possible to specialize this? Maybe with some clever trait work I can have 
      device_slice.copy_from(device_slice) and device_slice.copy_from(host_slice) just work.
- Iterate over chunks/chunks_mut/exact_chunks/exact_chunks_mut of a buffer or slice
    - This would be useful in transferring data to the card block-by-block.

You should also be able to:
- Allocate and deallocate a device box
- Copy values to and from that box
- Convert box to and from raw pointers
*/

pub struct

/// Fixed-size device-side buffer. Provides basic access to device memory
/*#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceCopy> {
    buf: DevicePointer<T>,
    capacity: usize
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Create a new DeviceBuffer and fill it with zeroes.
    pub fn zeroed(size: usize) -> CudaResult<Self> {
        let mut uninit = unsafe { DeviceBuffer::uninitialized(size)? };
        uninit.memset(0);
        Ok(uninit)
    }

    /// Create a new DeviceBuffer with the same size as the given slice, and copy the values into 
    /// it.
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        let mut uninit = { DeviceBuffer::uninitialized(slice.len())? };
        uninit.copy_from(slice);
        Ok(uninit)
    }

    /// Create a new DeviceBuffer without initializing the allocated memory. The caller is
    /// responsible for ensuring that the allocated buffer is initialized.
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let bytes = size.checked_mul(mem::size_of::<T>()).ok_or(CudaError::InvalidMemoryAllocation)?;

        let mut ptr = cuda_malloc(bytes)?;
        Ok(DeviceBuffer{ buf : ptr, capacity: size })
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    pub fn as_slice(&self) -> DeviceSlice<T> {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    pub fn as_mut_slice(&mut self) -> DeviceSliceMut<T> {
        self
    }

    /// Creates a `DeviceBuffer<T>` directly from the raw components of another Device buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `DeviceBuffer<T>`
    ///   (at least, it's highly likely to be incorrect if it wasn't).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `DeviceBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    pub unsafe fn from_raw_parts(ptr: *mut T, size: usize) -> DeviceBuffer<T> {
        DeviceBuffer {
            buf: DevicePointer::wrap(ptr),
            capacity: size,
        }
    }
}
impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            cuda_free(self.buf).expect("Failed to deallocate CUDA Device memory.");
        }
        self.capacity = 0;
    }
}
*/