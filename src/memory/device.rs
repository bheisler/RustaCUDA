use cuda_sys::cudart::*;
use error::{CudaResult, ToResult};
use memory::DeviceCopy;
use memory::DevicePointer;
use memory::{cuda_free, cuda_malloc};
use std::fmt::{self, Pointer};
use std::mem;
use std::os::raw::c_void;

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
    - This would be useful in transferring data to the card block-by-block.*/

/// Sealed trait implemented by types which can be the source or destination when copying data
/// to/from the device.
pub trait CopyDestination<O>: ::private::Sealed {
    /// Copy data from `source`. `source` must be the same size that `self` was allocated for.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_from(&mut self, source: &O) -> CudaResult<()>;

    /// Copy data to `dest`. `dest` must be the same size that `self` was allocated for.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_to(&self, dest: &mut O) -> CudaResult<()>;
}

/// A pointer type for heap-allocation in CUDA Device Memory. See the module-level-documentation
/// for more information on device memory.
#[derive(Debug)]
pub struct DeviceBox<T: DeviceCopy> {
    ptr: DevicePointer<T>,
}
impl<T: DeviceCopy> DeviceBox<T> {
    /// Allocate device memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let five = DeviceBox::new(&5).unwrap();
    /// ```
    pub fn new(val: &T) -> CudaResult<Self> {
        let mut dev_box = unsafe { DeviceBox::uninitialized()? };
        dev_box.copy_from(val)?;
        Ok(dev_box)
    }

    /// Allocate device memory, but do not initialize it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety:
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let mut five = unsafe { DeviceBox::uninitialized().unwrap() };
    /// five.copy_from(&5u64).unwrap();
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(DeviceBox {
                ptr: DevicePointer::null(),
            })
        } else {
            let ptr = cuda_malloc(1)?;
            Ok(DeviceBox { ptr })
        }
    }

    /// Constructs a DeviceBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cudaMallocManaged` CUDA API
    /// call.
    ///
    /// # Safety:
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples:
    /// ```
    /// use rustacuda::memory::*;
    /// let x = DeviceBox::new(&5).unwrap();
    /// let ptr = DeviceBox::into_device(x).as_raw_mut();
    /// let x = unsafe { DeviceBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        DeviceBox {
            ptr: DevicePointer::wrap(ptr),
        }
    }

    /// Constructs a DeviceBox from a DevicePointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cudaMallocManaged` CUDA API
    /// call, such as one taken from `DeviceBox::into_device`.
    ///
    /// # Safety:
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples:
    /// ```
    /// use rustacuda::memory::*;
    /// let x = DeviceBox::new(&5).unwrap();
    /// let ptr = DeviceBox::into_device(x);
    /// let x = unsafe { DeviceBox::from_device(ptr) };
    /// ```
    pub unsafe fn from_device(ptr: DevicePointer<T>) -> Self {
        DeviceBox { ptr }
    }

    /// Consumes the DeviceBox, returning the wrapped DevicePointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the DeviceBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new DeviceBox using the `DeviceBox::from_device` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `DeviceBox::into_device(b)` instead of `b.into_device()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples:
    /// ```
    /// use rustacuda::memory::*;
    /// let x = DeviceBox::new(&5).unwrap();
    /// let ptr = DeviceBox::into_device(x);
    /// # unsafe { DeviceBox::from_device(ptr) };
    /// ```
    #[allow(wrong_self_convention)]
    pub fn into_device(mut b: DeviceBox<T>) -> DevicePointer<T> {
        let ptr = mem::replace(&mut b.ptr, DevicePointer::null());
        mem::forget(b);
        ptr
    }
}
impl<T: DeviceCopy> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let ptr = ::std::mem::replace(&mut self.ptr, DevicePointer::null());
            // No choice but to panic if this fails.
            unsafe {
                cuda_free(ptr).expect("Failed to deallocate CUDA memory.");
            }
        }
    }
}
impl<T: DeviceCopy> Pointer for DeviceBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: DeviceCopy> CopyDestination<T> for DeviceBox<T> {
    fn copy_from(&mut self, val: &T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.ptr.as_raw_mut() as *mut c_void,
                    val as *const T as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyHostToDevice,
                ).toResult()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyDeviceToHost,
                ).toResult()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceBox<T>> for DeviceBox<T> {
    fn copy_from(&mut self, val: &DeviceBox<T>) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.ptr.as_raw_mut() as *mut c_void,
                    val.ptr.as_raw() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                ).toResult()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DeviceBox<T>) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val.ptr.as_raw_mut() as *mut c_void,
                    self.ptr.as_raw() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                ).toResult()?
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_device_box {
    use super::*;

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl ::memory::DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_allocate_and_free_device_box() {
        let x = DeviceBox::new(&5u64).unwrap();
        drop(x);
    }

    #[test]
    fn test_device_box_allocates_for_non_zst() {
        let x = DeviceBox::new(&5u64).unwrap();
        let ptr = DeviceBox::into_device(x);
        assert!(!ptr.is_null());
        let _ = unsafe { DeviceBox::from_device(ptr) };
    }

    #[test]
    fn test_device_box_doesnt_allocate_for_zero_sized_type() {
        let x = DeviceBox::new(&ZeroSizedType).unwrap();
        let ptr = DeviceBox::into_device(x);
        assert!(ptr.is_null());
        let _ = unsafe { DeviceBox::from_device(ptr) };
    }

    #[test]
    fn test_into_from_device() {
        let x = DeviceBox::new(&5u64).unwrap();
        let ptr = DeviceBox::into_device(x);
        let _ = unsafe { DeviceBox::from_device(ptr) };
    }

    #[test]
    fn test_copy_host_to_device() {
        let y = 5u64;
        let mut x = DeviceBox::new(&0u64).unwrap();
        x.copy_from(&y).unwrap();
        let mut z = 10u64;
        x.copy_to(&mut z).unwrap();
        assert_eq!(y, z);
    }

    #[test]
    fn test_copy_device_to_host() {
        let x = DeviceBox::new(&5u64).unwrap();
        let mut y = 0u64;
        x.copy_to(&mut y).unwrap();
        assert_eq!(5, y);
    }

    #[test]
    fn test_copy_device_to_device() {
        let x = DeviceBox::new(&5u64).unwrap();
        let mut y = DeviceBox::new(&0u64).unwrap();
        let mut z = DeviceBox::new(&0u64).unwrap();
        x.copy_to(&mut y).unwrap();
        z.copy_from(&y).unwrap();

        let mut h = 0u64;
        z.copy_to(&mut h).unwrap();
        assert_eq!(5, h);
    }
}

/*
/// Fixed-size device-side buffer. Provides basic access to device memory
#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceCopy> {
    buf: DevicePointer<T>,
    capacity: usize
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Create a new DeviceBuffer and fill it with zeroes.
    pub fn zeroed(size: usize) -> CudaResult<Self> {
        let mut uninitialized = unsafe { DeviceBuffer::uninitialized(size)? };
        uninitialized.memset(0);
        Ok(uninitialized)
    }

    /// Create a new DeviceBuffer with the same size as the given slice, and copy the values into 
    /// it.
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        let mut uninitialized = { DeviceBuffer::uninitialized(slice.len())? };
        uninitialized.copy_from(slice);
        Ok(uninitialized)
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
