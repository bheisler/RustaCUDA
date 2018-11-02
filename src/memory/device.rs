use cuda_sys::cudart::*;
use error::{CudaError, CudaResult, ToResult};
use memory::DeviceCopy;
use memory::DevicePointer;
use memory::{cuda_free, cuda_malloc};
use std::fmt::{self, Pointer};
use std::iter::{ExactSizeIterator, FusedIterator};
use std::mem;
use std::ops::{
    Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};
use std::os::raw::c_void;
use std::ptr;
use std::slice::{Chunks as SChunks, ChunksMut as SChunksMut};

/// Sealed trait implemented by types which can be the source or destination when copying data
/// to/from the device or from one device allocation to another.
pub trait CopyDestination<O: ?Sized>: ::private::Sealed {
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

/// Fixed-size device-side slice.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceSlice<T: DeviceCopy>([T]);
// This works by faking a regular slice out of the device raw-pointer and the length and transmuting
// I have no idea if this is safe or not. Probably not, though I can't imagine how the compiler
// could possibly know that the pointer is not de-referenceable. I'm banking that we get proper
// Dynamicaly-sized Types before the compiler authors break this assumption.

impl<T: DeviceCopy> DeviceSlice<T> {
    /// Returns the number of elements in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// assert_eq!(a.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let a : DeviceBuffer<u64> = unsafe { DeviceBuffer::uninitialized(0).unwrap() };
    /// assert!(a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return a raw device-pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else
    /// it will end up pointing to garbage. The caller must also ensure that the pointer is not
    /// dereferenced by the CPU.
    ///
    /// Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// println!("{:p}", a.as_ptr());
    /// ```
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    /// Returns an unsafe mutable device-pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else
    /// it will end up pointing to garbage. The caller must also ensure that the pointer is not
    /// dereferenced by the CPU.
    ///
    /// Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let mut a = DeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// println!("{:p}", a.as_mut_ptr());
    /// ```
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    /// Divides one DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// let (left, right) = buf.split_at(3);
    /// let mut left_host = [0u64, 0, 0];
    /// let mut right_host = [0u64, 0, 0];
    /// left.copy_to(&mut left_host).unwrap();
    /// right.copy_to(&mut right_host).unwrap();
    /// assert_eq!([0u64, 1, 2], left_host);
    /// assert_eq!([3u64, 4, 5], right_host);
    /// ```
    pub fn split_at(&self, mid: usize) -> (&DeviceSlice<T>, &DeviceSlice<T>) {
        let (left, right) = self.0.split_at(mid);
        unsafe {
            (
                DeviceSlice::from_slice(left),
                DeviceSlice::from_slice(right),
            )
        }
    }

    /// Divides one mutable DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let mut buf = DeviceBuffer::from_slice(&[0u64, 0, 0, 0, 0, 0]).unwrap();
    ///
    /// {
    ///     let (left, right) = buf.split_at_mut(3);
    ///     let left_host = [0u64, 1, 2];
    ///     let right_host = [3u64, 4, 5];
    ///     left.copy_from(&left_host).unwrap();
    ///     right.copy_from(&right_host).unwrap();
    /// }
    ///
    /// let mut host_full = [0u64; 6];
    /// buf.copy_to(&mut host_full).unwrap();
    /// assert_eq!([0u64, 1, 2, 3, 4, 5], host_full);
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut DeviceSlice<T>, &mut DeviceSlice<T>) {
        let (left, right) = self.0.split_at_mut(mid);
        unsafe {
            (
                DeviceSlice::from_slice_mut(left),
                DeviceSlice::from_slice_mut(right),
            )
        }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time. The chunks are device
    /// slices and do not overlap. If `chunk_size` does not divide the length of the slice, then the
    /// last chunk will not have length `chunk_size`.
    ///
    /// See `exact_chunks` for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let slice = DeviceBuffer::from_slice(&[1u64, 2, 3, 4, 5]).unwrap();
    /// let mut iter = slice.chunks(2);
    ///
    /// assert_eq!(iter.next().unwrap().len(), 2);
    ///
    /// let mut host_buf = [0u64, 0];
    /// iter.next().unwrap().copy_to(&mut host_buf).unwrap();
    /// assert_eq!([3, 4], host_buf);
    ///
    /// assert_eq!(iter.next().unwrap().len(), 1);
    ///
    /// ```
    pub fn chunks(&self, chunk_size: usize) -> Chunks<T> {
        Chunks(self.0.chunks(chunk_size))
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time. The chunks are
    /// mutable device slices and do not overlap. If `chunk_size` does not divide the length of the
    /// slice, then the last chunk will not have length `chunk_size`.
    ///
    /// See `exact_chunks` for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let mut slice = DeviceBuffer::from_slice(&[0u64, 0, 0, 0, 0]).unwrap();
    /// {
    ///     let mut iter = slice.chunks_mut(2);
    ///
    ///     assert_eq!(iter.next().unwrap().len(), 2);
    ///
    ///     let host_buf = [2u64, 3];
    ///     iter.next().unwrap().copy_from(&host_buf).unwrap();
    ///
    ///     assert_eq!(iter.next().unwrap().len(), 1);
    /// }
    ///
    /// let mut host_buf = [0u64, 0, 0, 0, 0];
    /// slice.copy_to(&mut host_buf).unwrap();
    /// assert_eq!([0u64, 0, 2, 3, 0], host_buf);
    /// ```
    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        ChunksMut(self.0.chunks_mut(chunk_size))
    }

    /// Private function used to transmute a CPU slice (which must have the device pointer as it's
    /// buffer pointer) to a DeviceSlice. Completely unsafe.
    unsafe fn from_slice(slice: &[T]) -> &DeviceSlice<T> {
        &*(slice as *const [T] as *const DeviceSlice<T>)
    }

    /// Private function used to transmute a mutable CPU slice (which must have the device pointer
    /// as it's buffer pointer) to a mutable DeviceSlice. Completely unsafe.
    unsafe fn from_slice_mut(slice: &mut [T]) -> &mut DeviceSlice<T> {
        &mut *(slice as *mut [T] as *mut DeviceSlice<T>)
    }
}

/// An iterator over a `DeviceSlice` in (non-overlapping) chunks (`chunk_size` elements at a time).
///
/// When the slice len is not evenly divided by the chunk size, the last slice of the iteration will
/// be the remainder.
///
/// This struct is created by the `chunks` method on `DeviceSlices`.
#[derive(Debug, Clone)]
pub struct Chunks<'a, T: DeviceCopy + 'a>(SChunks<'a, T>);
impl<'a, T: DeviceCopy> Iterator for Chunks<'a, T> {
    type Item = &'a DeviceSlice<T>;

    fn next(&mut self) -> Option<&'a DeviceSlice<T>> {
        self.0
            .next()
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0
            .nth(n)
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0
            .last()
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }
}
impl<'a, T: DeviceCopy> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a DeviceSlice<T>> {
        self.0
            .next_back()
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }
}
impl<'a, T: DeviceCopy> ExactSizeIterator for Chunks<'a, T> {}
impl<'a, T: DeviceCopy> FusedIterator for Chunks<'a, T> {}

/// An iterator over a `DeviceSlice` in (non-overlapping) mutable chunks (`chunk_size` elements at
/// a time).
///
/// When the slice len is not evenly divided by the chunk size, the last slice of the iteration will
/// be the remainder.
///
/// This struct is created by the `chunks` method on `DeviceSlices`.
#[derive(Debug)]
pub struct ChunksMut<'a, T: DeviceCopy + 'a>(SChunksMut<'a, T>);
impl<'a, T: DeviceCopy> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut DeviceSlice<T>;

    fn next(&mut self) -> Option<&'a mut DeviceSlice<T>> {
        self.0
            .next()
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0
            .nth(n)
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0
            .last()
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }
}
impl<'a, T: DeviceCopy> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut DeviceSlice<T>> {
        self.0
            .next_back()
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }
}
impl<'a, T: DeviceCopy> ExactSizeIterator for ChunksMut<'a, T> {}
impl<'a, T: DeviceCopy> FusedIterator for ChunksMut<'a, T> {}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T: DeviceCopy> Index<$t> for DeviceSlice<T>
            {
                type Output = DeviceSlice<T>;

                fn index(&self, index: $t) -> &Self {
                    unsafe { DeviceSlice::from_slice(self.0.index(index)) }
                }
            }

            impl<T: DeviceCopy> IndexMut<$t> for DeviceSlice<T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self {
                    unsafe { DeviceSlice::from_slice_mut( self.0.index_mut(index)) }
                }
            }
        )*
    }
}
impl_index!{
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> CopyDestination<I> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &I) -> CudaResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.0.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyHostToDevice,
                ).toResult()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut I) -> CudaResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyDeviceToHost,
                ).toResult()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceSlice<T>> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &DeviceSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.0.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                ).toResult()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut DeviceSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = mem::size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                ).toResult()?
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<DeviceBuffer<T>> for DeviceSlice<T> {
    fn copy_from(&mut self, val: &DeviceBuffer<T>) -> CudaResult<()> {
        self.copy_from(val as &DeviceSlice<T>)
    }

    fn copy_to(&self, val: &mut DeviceBuffer<T>) -> CudaResult<()> {
        self.copy_to(val as &mut DeviceSlice<T>)
    }
}

/// Fixed-size device-side buffer. Provides basic access to device memory.
#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceCopy> {
    buf: DevicePointer<T>,
    capacity: usize,
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
    /// use rustacuda::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = DeviceBuffer::from_slice(&values).unwrap();
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = DeviceBuffer::uninitialized(slice.len())?;
            uninit.copy_from(slice).unwrap();
            Ok(uninit)
        }
    }

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
}
impl<T: DeviceCopy> Deref for DeviceBuffer<T> {
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
impl<T: DeviceCopy> DerefMut for DeviceBuffer<T> {
    fn deref_mut(&mut self) -> &mut DeviceSlice<T> {
        unsafe {
            &mut *(::std::slice::from_raw_parts_mut(self.buf.as_raw_mut(), self.capacity)
                as *mut [T] as *mut DeviceSlice<T>)
        }
    }
}
impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            let ptr = ::std::mem::replace(&mut self.buf, DevicePointer::null());
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

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl ::memory::DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_from_slice_drop() {
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        drop(buf);
    }

    #[test]
    fn test_copy_to_from_device() {
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        let buf = DeviceBuffer::from_slice(&start).unwrap();
        buf.copy_to(&mut end).unwrap();
        assert_eq!(start, end);
    }

    #[test]
    fn test_slice() {
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
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = [0u64, 1, 2, 3, 4];
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_copy_from_h2d_wrong_size() {
        let start = [0u64, 1, 2, 3, 4];
        let mut buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    fn test_copy_device_slice_to_device() {
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
        let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_copy_from_d2d_wrong_size() {
        let mut buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let start = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_from(&start);
    }
}
