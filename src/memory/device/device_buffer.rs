use crate::error::{CudaError, CudaResult, DropResult, ToResult};
use crate::memory::device::{CopyDestination, DeviceSlice};
use crate::memory::malloc::{cuda_free, cuda_malloc};
use crate::memory::DeviceCopy;
use crate::memory::DevicePointer;
use cuda_sys::cuda;
use std::mem;
use std::ops::{Deref, DerefMut};

use std::ptr;

/// Fixed-size device-side buffer. Provides basic access to device memory.
#[derive(Debug)]
pub struct DeviceBuffer<T> {
    buf: DevicePointer<T>,
    capacity: usize,
}
impl<T> DeviceBuffer<T> {
    /// Allocate a new device buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors:
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety:
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = unsafe { DeviceBuffer::uninitialized(5).unwrap() };
    /// buffer.copy_from(&[0u64, 1, 2, 3, 4]).unwrap();
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let bytes = size
            .checked_mul(mem::size_of::<T>())
            .ok_or(CudaError::InvalidMemoryAllocation)?;

        let ptr = if bytes > 0 {
            cuda_malloc(bytes)?
        } else {
            DevicePointer::wrap(ptr::NonNull::dangling().as_ptr() as *mut T)
        };
        Ok(DeviceBuffer {
            buf: ptr,
            capacity: size,
        })
    }

    /// Allocate a new device buffer large enough to hold `size` `T`'s and fill the contents with
    /// zeroes (`0u8`).
    ///
    /// # Errors:
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety:
    ///
    /// The backing memory is zeroed, which may not be a valid bit-pattern for type `T`. The caller
    /// must ensure either that all-zeroes is a valid bit-pattern for type `T` or that the backing
    /// memory is set to a valid value before it is read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let buffer = unsafe { DeviceBuffer::zeroed(5).unwrap() };
    /// let mut host_values = [1u64, 2, 3, 4, 5];
    /// buffer.copy_to(&mut host_values).unwrap();
    /// assert_eq!([0u64, 0, 0, 0, 0], host_values);
    /// ```
    pub unsafe fn zeroed(size: usize) -> CudaResult<Self> {
        let bytes = size
            .checked_mul(mem::size_of::<T>())
            .ok_or(CudaError::InvalidMemoryAllocation)?;

        let ptr = if bytes > 0 {
            let mut ptr = cuda_malloc(bytes)?;
            cuda::cuMemsetD8_v2(ptr.as_raw_mut() as u64, 0, size * mem::size_of::<T>())
                .to_result()?;
            ptr
        } else {
            DevicePointer::wrap(ptr::NonNull::dangling().as_ptr() as *mut T)
        };
        Ok(DeviceBuffer {
            buf: ptr,
            capacity: size,
        })
    }

    /// Creates a `DeviceBuffer<T>` directly from the raw components of another device buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `DeviceBuffer` or
    /// [`cuda_malloc`](fn.cuda_malloc.html).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `DeviceBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use std::mem;
    /// use rustacuda::memory::*;
    ///
    /// let mut buffer = DeviceBuffer::from_slice(&[0u64; 5]).unwrap();
    /// let ptr = buffer.as_device_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { DeviceBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: DevicePointer<T>, capacity: usize) -> DeviceBuffer<T> {
        DeviceBuffer { buf: ptr, capacity }
    }

    /// Destroy a `DeviceBuffer`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = DeviceBuffer::from_slice(&[10, 20, 30]).unwrap();
    /// match DeviceBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut dev_buf: DeviceBuffer<T>) -> DropResult<DeviceBuffer<T>> {
        if dev_buf.buf.is_null() {
            return Ok(());
        }

        if dev_buf.capacity > 0 && mem::size_of::<T>() > 0 {
            let capacity = dev_buf.capacity;
            let ptr = mem::replace(&mut dev_buf.buf, DevicePointer::null());
            unsafe {
                match cuda_free(ptr) {
                    Ok(()) => {
                        mem::forget(dev_buf);
                        Ok(())
                    }
                    Err(e) => Err((e, DeviceBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}
impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocate a new device buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors:
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = DeviceBuffer::from_slice(&values).unwrap();
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = DeviceBuffer::uninitialized(slice.len())?;
            uninit.copy_from(slice)?;
            Ok(uninit)
        }
    }
}
impl<T> Deref for DeviceBuffer<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &DeviceSlice<T> {
        unsafe {
            DeviceSlice::from_slice(::std::slice::from_raw_parts(
                self.buf.as_raw(),
                self.capacity,
            ))
        }
    }
}
impl<T> DerefMut for DeviceBuffer<T> {
    fn deref_mut(&mut self) -> &mut DeviceSlice<T> {
        unsafe {
            &mut *(::std::slice::from_raw_parts_mut(self.buf.as_raw_mut(), self.capacity)
                as *mut [T] as *mut DeviceSlice<T>)
        }
    }
}
impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            let ptr = mem::replace(&mut self.buf, DevicePointer::null());
            unsafe {
                cuda_free(ptr).expect("Failed to deallocate CUDA Device memory.");
            }
        }
        self.capacity = 0;
    }
}

#[cfg(test)]
mod test_device_buffer {
    use super::*;
    use crate::memory::device::DeviceBox;

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_from_slice_drop() {
        let _context = crate::quick_init().unwrap();
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        drop(buf);
    }

    #[test]
    fn test_copy_to_from_device() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        let buf = DeviceBuffer::from_slice(&start).unwrap();
        buf.copy_to(&mut end).unwrap();
        assert_eq!(start, end);
    }

    #[test]
    fn test_slice() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0];
        let mut buf = DeviceBuffer::from_slice(&[0u64, 0, 0, 0]).unwrap();
        buf.copy_from(&start[0..4]).unwrap();
        buf[0..2].copy_to(&mut end).unwrap();
        assert_eq!(start[0..2], end);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2h_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = [0u64, 1, 2, 3, 4];
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_copy_from_h2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4];
        let mut buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    fn test_copy_device_slice_to_device() {
        let _context = crate::quick_init().unwrap();
        let start = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut mid = DeviceBuffer::from_slice(&[0u64, 0, 0, 0]).unwrap();
        let mut end = DeviceBuffer::from_slice(&[0u64, 0]).unwrap();
        let mut host_end = [0u64, 0];
        start[1..5].copy_to(&mut mid).unwrap();
        end.copy_from(&mid[1..3]).unwrap();
        end.copy_to(&mut host_end).unwrap();
        assert_eq!([2u64, 3], host_end);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_copy_from_d2d_wrong_size() {
        let _context = crate::quick_init().unwrap();
        let mut buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let start = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    fn test_can_create_uninitialized_non_devicecopy_buffers() {
        let _context = crate::quick_init().unwrap();
        unsafe {
            let _box: DeviceBox<Vec<u8>> = DeviceBox::uninitialized().unwrap();
            let buffer: DeviceBuffer<Vec<u8>> = DeviceBuffer::uninitialized(10).unwrap();
            let _slice = &buffer[0..5];
        }
    }
}
